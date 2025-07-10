import base64
import os
import time
import tempfile
import logging
import re
from typing import List, Dict, Any, Optional
import fitz
import instructor
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from .utils import TokenCounter
from .model import (
    DocumentExtractionResult,
    ContractEntity,
    ExtractedEntity,
    ContractEntitiesResponse,
    BulkFileUploadResponse,
    FileUploadResponse,
    DocumentModel,
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import azure.functions as func

import easyocr
from PIL import Image
import io
import sys

# Fix OpenMP conflict - DEVE SER ANTES de qualquer import que use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    logging.info("Todas as variáveis de ambiente carregadas com sucesso")
except ValueError as e:
    logging.error(f"Erro na configuração: {e}")
    raise

# Configurar encoding para evitar problemas com caracteres especiais
if os.name == 'nt':  # Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suprimir logs verbosos do EasyOCR
os.environ['EASYOCR_MODULE_PATH'] = os.getcwd()
logging.getLogger('easyocr').setLevel(logging.WARNING)

class DocumentEntityExtractor:
    """Extrator de entidades específicas de documentos usando embeddings"""

    def __init__(self):
        self.token_counter = TokenCounter()

        base_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        )

        self.llm = instructor.from_openai(base_client)

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    def create_vector_store_from_chunks(
        self, document_chunks: List[Dict[str, Any]]
    ) -> FAISS:
        """Cria um banco de dados vetorial FAISS a partir dos chunks já processados"""
        logging.info(
            f"Criando banco vetorial com {len(document_chunks)} chunks pré-processados"
        )

        documents = []
        chunk_texts = []

        for chunk in document_chunks:
            chunk_content = base64.b64decode(chunk["content"]).decode("utf-8")
            chunk_texts.append(chunk_content)

            doc = Document(
                page_content=chunk_content,
                metadata={
                    "chunk_id": int(chunk["chunk_id"]),
                    "page": int(chunk["page"]),
                    "tokens": int(chunk["tokens"]),
                },
            )
            documents.append(doc)

        self.token_counter.log_embedding_usage(chunk_texts)

        vector_store = FAISS.from_documents(documents, self.embeddings)

        logging.info(f"Banco vetorial criado com {len(documents)} chunks")
        return vector_store

    def extract_all_entities(
        self, document_chunks: List[Dict[str, Any]], filename: str
    ) -> DocumentExtractionResult:
        """Extrai entidades usando Instructor para garantir JSON válido"""
        start_time = time.time()

        try:
            vector_store = self.create_vector_store_from_chunks(document_chunks)

            combined_query = """
            Contratos financeiros: indexação de juros, spread, taxas, datas de emissão e vencimento, 
            valores nominais, cronogramas de pagamento, atualização monetária IPCA IGPM SELIC, 
            bases de cálculo 252 365 dias, fluxos de amortização, DI CDI pré-fixado
            """

            docs = vector_store.similarity_search_with_score(combined_query, k=20)

            if not docs:
                return self._create_empty_result(filename, start_time)

            context_parts = []
            all_page_refs = set()

            for doc, score in docs:
                context_parts.append(
                    f"Chunk {doc.metadata.get('chunk_id', 'N/A')} (página {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}"
                )
                if "page" in doc.metadata:
                    all_page_refs.add(int(doc.metadata["page"]))

            full_context = "\n\n---\n\n".join(context_parts)

            prompt = f"""
            <role>
            Você é um(a) **analista sênior de contratos financeiros**.  
            Sua tarefa é **LER** o texto abaixo e **DEVOLVER** exatamente **um** objeto
            JSON com os nove campos pedidos, **somente strings** (nunca arrays),
            no formato mostrado depois da lista.
            </role>
            <context>
            {full_context}
            </context>
            <rules>
            REGRAS OBRIGATÓRIAS
            1. Copie o conteúdo **exatamente como está no contrato** – não traduza
            nem reescreva números, índices ou datas.
            2. Se o item não existir, responda **"NÃO ENCONTRADO"**.
            3. Se houver mais de um valor para o mesmo item, una-os em **uma única
            string separada por vírgulas**, mantendo a ordem em que aparecem.
            4. Retorne apenas o JSON válido (sem comentários, sem texto antes ou
            depois).
            </rules>
            <items>
            ITENS QUE DEVEM SER EXTRAÍDOS  
            (***guias de busca*** entre colchetes ajudam a localizar no contrato)

            1. **ATUALIZAÇÃO MONETÁRIA** – índice que corrige o **principal**  
            [palavras-chave: "atualização monetária", "índice de correção",
            "IPCA", "IGP-M", "SELIC", "não haverá atualização"].

            2. **JUROS REMUNERATÓRIOS** – indexador **principal** que corrige os
            **juros** (DI, CDI, taxa prefixada, etc.) 
            VALOR UNICO: EXEMPLO: "DI+" ou "DI" ou "IPCA" ou "IPCA+" ou "CDI" ou "CDI+" ou "SELIC" ou "SELIC+", etc.

            3. **SPREAD FIXO** – percentual adicional **sobre** o indexador principal  
            ["+0,30 %", "acréscimo de 2 % a.a.", "spread"].

            4. **BASE DE CÁLCULO** – metodologia usada nas fórmulas de juros  
            ["252 dias úteis", "365/365", "base ACT/360"].

            5. **DATA EMISSÃO** – data(s) a partir da qual o título passa a vigorar  
            ["Data de Emissão", "Data de Colocação"; se houver séries, todas elas].

            6. **DATA VENCIMENTO** – data(s) final(is) da obrigação correspondente  
            ["Data de Vencimento", "Vencimento Final"; manter mesma ordem de
            emissão].

            7. **VALOR NOMINAL UNITÁRIO** – valor de face por título/cota  
            ["Valor Nominal Unitário", "VNU", "Valor de Face"].

            8. **FLUXOS DE PAGAMENTO DE AMORTIZAÇÃO E JUROS** – **todas** as datas
            que aparecem no cronograma / anexo de pagamentos  
            ["Cronograma de Pagamento", "Anexo XI", "Fluxo de Caixa"].
            → Devolva **todas** em uma única string, separadas por vírgulas,
            no formato DD/MM/AAAA.

            9. **FLUXOS PERCENTUAIS DE AMORTIZAÇÃO E JUROS** – percentuais que
            aparecem na mesma tabela do item 8, na **mesma ordem** das datas  
            [colunas "% Amortização", "Taxa de Amort."].
            → Use vírgula como separador e vírgula decimal (ex.: 0,0000 %).
            </items>
            """

            response = self.llm.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                response_model=ContractEntitiesResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                max_retries=3,
            )

            contract_entities = ContractEntity()
            entities_found = 0

            for field_name, value in response.model_dump().items():
                if value and value.strip() and value.upper() != "NÃO ENCONTRADO":
                    confidence = self._calculate_confidence_from_context(
                        value, full_context
                    )

                    entity = ExtractedEntity(
                        entity_type=field_name,
                        value=value.strip(),
                        confidence=confidence,
                        context=(
                            full_context[:500] + "..."
                            if len(full_context) > 500
                            else full_context
                        ),
                    )
                    setattr(contract_entities, field_name, entity)
                    entities_found += 1
                    logging.info(
                        f"Entidade '{field_name}' encontrada: {value.strip()[:50]}..."
                    )

            processing_time = int((time.time() - start_time) * 1000)

            logging.info(
                f"Extração com Instructor: {entities_found}/9 entidades - JSON sempre válido!"
            )

            return DocumentExtractionResult(
                filename=filename,
                success=True,
                contract_entities=contract_entities,
                processing_time_ms=processing_time,
                token_summary=self.token_counter.get_summary(),
            )

        except Exception as e:
            logging.error(f"Erro na extração com Instructor: {e}")
            return self._create_empty_result(filename, start_time)

    def _calculate_confidence_from_context(self, value: str, context: str) -> float:
        """Calcula confiança baseada na presença do valor no contexto"""
        if not value or not context:
            return 0.5

        if value.lower() in context.lower():
            return 0.9

        keywords = value.lower().split()
        found_keywords = sum(1 for keyword in keywords if keyword in context.lower())

        if found_keywords > 0:
            confidence = min(0.8, 0.5 + (found_keywords / len(keywords)) * 0.3)
            return round(confidence, 3)

        return 0.5

    def _create_empty_result(
        self, filename: str, start_time: float
    ) -> DocumentExtractionResult:
        """Cria um resultado vazio quando nenhum chunk relevante é encontrado"""
        processing_time = int((time.time() - start_time) * 1000)

        return DocumentExtractionResult(
            filename=filename,
            success=False,
            contract_entities=ContractEntity(),
            processing_time_ms=processing_time,
            token_summary=self.token_counter.get_summary(),
            error_message="Nenhum chunk relevante encontrado no documento",
        )

    def create_tabela_faas(
        self, contract_entities: ContractEntity, filename: str
    ) -> List[Dict[str, Any]]:
        """
        Cada row é uma data, tendo Inicio e Vencimento.
        Inicio / Vencimento / Valor nominal / % Amortização / Index / Spread Fixo
        """
        tabela_faas = []

        try:
            logging.info(f"DEBUG: Iniciando criação de tabela FAAS para {filename}")
            logging.info(
                f"DEBUG: fluxos_pagamento existe: {contract_entities.fluxos_pagamento is not None}"
            )
            if contract_entities.fluxos_pagamento:
                logging.info(
                    f"DEBUG: fluxos_pagamento valor: '{contract_entities.fluxos_pagamento.value}'"
                )

            logging.info(
                f"DEBUG: fluxos_percentuais existe: {contract_entities.fluxos_percentuais is not None}"
            )
            if contract_entities.fluxos_percentuais:
                logging.info(
                    f"DEBUG: fluxos_percentuais valor: '{contract_entities.fluxos_percentuais.value}'"
                )

            fluxos_pagamento = self._extract_dates_from_fluxos(
                contract_entities.fluxos_pagamento
            )
            fluxos_percentuais = self._extract_percentages_from_fluxos(
                contract_entities.fluxos_percentuais
            )

            logging.info(f"DEBUG: fluxos_pagamento extraídos: {fluxos_pagamento}")
            logging.info(f"DEBUG: fluxos_percentuais extraídos: {fluxos_percentuais}")

            valor_nominal = self._get_entity_value(
                contract_entities.valor_nominal_unitario, ""
            )
            index_info = self._get_entity_value(
                contract_entities.juros_remuneratorios, ""
            )

            index_completo = self._combine_index_info(index_info)

            if not fluxos_pagamento:
                logging.info(
                    "DEBUG: Nenhum fluxo de pagamento encontrado, criando linha básica"
                )
                data_inicio = self._get_first_emission_date(
                    contract_entities.data_emissao
                )
                data_vencimento = self._get_last_maturity_date(
                    contract_entities.data_vencimento
                )

                if data_inicio or data_vencimento:
                    linha_faas = {
                        "Código": filename,
                        "Início": data_inicio,
                        "Vencimento": data_vencimento,
                        "Valor Nominal": valor_nominal,
                        "Valor Atualizado": "",
                        "% Amort": "100,00%",
                        "Amort. Nominal": "",
                        "Amort. Atual.": "",
                        "Amort. extra.": "",
                        "Remuneração": index_completo,
                        "D": "",
                    }
                    tabela_faas.append(linha_faas)
                    logging.info("DEBUG: Linha básica criada com sucesso")
            else:
                logging.info(
                    f"DEBUG: Criando {len(fluxos_pagamento)} linhas a partir dos fluxos"
                )
                for i, data_pagamento in enumerate(fluxos_pagamento):
                    percentual_amortizacao = (
                        fluxos_percentuais[i] if i < len(fluxos_percentuais) else ""
                    )

                    if i == 0:
                        data_inicio = self._get_first_emission_date(
                            contract_entities.data_emissao
                        )
                    else:
                        data_inicio = fluxos_pagamento[i - 1]

                    linha_faas = {
                        "Código": filename,
                        "Início": data_inicio,
                        "Vencimento": data_pagamento,
                        "Valor Nominal": valor_nominal,
                        "Valor Atualizado": "",
                        "% Amort": percentual_amortizacao,
                        "Amort. Nominal": "",
                        "Amort. Atual.": "",
                        "Amort. extra.": "",
                        "Remuneração": index_completo,
                        "D": "",
                    }

                    tabela_faas.append(linha_faas)

            logging.info(
                f"Tabela FAAS criada com {len(tabela_faas)} linhas para o arquivo {filename}"
            )
            if tabela_faas:
                logging.info(f"Primeira linha da tabela FAAS: {tabela_faas[0]}")
            return tabela_faas

        except Exception as e:
            logging.error(f"Erro ao criar tabela FAAS para {filename}: {e}")
            import traceback

            logging.error(f"Traceback completo: {traceback.format_exc()}")
            return []

    def create_row_faas_resumo(
        self, contract_entities: ContractEntity, filename: str
    ) -> Dict[str, Any]:
        """
        Cada row é um contrato, sendo o inicio a primeira data de emissao, e o vencimento a ultima data de vencimento.
        Nome do arquivo /Inicio / Vencimento / % Amortização / Valor nominal / Index / Spread Fixo
        """
        try:
            data_inicio = self._get_first_emission_date(contract_entities.data_emissao)
            data_vencimento = self._get_last_maturity_date(
                contract_entities.data_vencimento
            )

            index_info = self._get_entity_value(
                contract_entities.juros_remuneratorios, ""
            )
            atualizacao_monetaria = self._get_entity_value(
                contract_entities.atualizacao_monetaria, ""
            )

            index_completo = self._combine_index_info(index_info)

            linha_resumo = {
                "Nome do Arquivo": filename,
                "Fundo": "Agente",
                "Link A": "",
                "Index": index_completo,
                "Aplicação": "",
                "Emissão": data_inicio,
                "Vencimento": data_vencimento,
                "Quantidade": "",
                "PU Mercado": "",
                "PU Custo": "",
                "Saldo": "",
            }

            logging.info(f"Linha de resumo FAAS criada para o arquivo {filename}")
            logging.info(f"Linha de resumo FAAS: {linha_resumo}")
            return linha_resumo

        except Exception as e:
            logging.error(f"Erro ao criar linha de resumo FAAS para {filename}: {e}")
            return {}

    def _extract_dates_from_fluxos(
        self, fluxos_entity: Optional[ExtractedEntity]
    ) -> List[str]:
        """Extrai lista de datas dos fluxos de pagamento"""
        logging.info("DEBUG: _extract_dates_from_fluxos chamado")
        if not fluxos_entity or not fluxos_entity.value:
            logging.info("DEBUG: Nenhum fluxo de pagamento encontrado ou valor vazio")
            return []

        logging.info(f"DEBUG: Valor dos fluxos de pagamento: '{fluxos_entity.value}'")

        date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
        dates = re.findall(date_pattern, fluxos_entity.value)

        logging.info(f"DEBUG: Datas encontradas com regex: {dates}")

        if not dates:
            logging.info(
                "DEBUG: Nenhuma data encontrada com regex, tentando extração flexível"
            )
            parts = fluxos_entity.value.split(",")
            for part in parts:
                part = part.strip()
                flexible_pattern = r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
                found_dates = re.findall(flexible_pattern, part)
                if found_dates:
                    dates.extend(found_dates)

            logging.info(f"DEBUG: Datas encontradas com extração flexível: {dates}")

        return dates

    def _extract_percentages_from_fluxos(
        self, fluxos_entity: Optional[ExtractedEntity]
    ) -> List[str]:
        """Extrai lista de percentuais dos fluxos percentuais"""
        if not fluxos_entity or not fluxos_entity.value:
            return []

        percentages = [p.strip() for p in fluxos_entity.value.split(",")]

        return percentages

    def _get_entity_value(
        self, entity: Optional[ExtractedEntity], default: str = ""
    ) -> str:
        """Extrai valor de uma entidade ou retorna default"""
        if entity and entity.value:
            return entity.value
        return default

    def _get_first_emission_date(
        self, data_emissao_entity: Optional[ExtractedEntity]
    ) -> str:
        """Extrai a primeira data de emissão"""
        if not data_emissao_entity or not data_emissao_entity.value:
            return ""

        dates = data_emissao_entity.value.split(",")
        first_date = dates[0].strip()

        return self._normalize_date_format(first_date)

    def _get_last_maturity_date(
        self, data_vencimento_entity: Optional[ExtractedEntity]
    ) -> str:
        """Extrai a última data de vencimento"""
        if not data_vencimento_entity or not data_vencimento_entity.value:
            return ""

        dates = data_vencimento_entity.value.split(",")
        last_date = dates[-1].strip()

        return self._normalize_date_format(last_date)

    def _normalize_date_format(self, date_str: str) -> str:
        """Normaliza formato de data para DD/MM/YYYY"""
        if not date_str:
            return ""

        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
            return date_str

        months = {
            "janeiro": "01",
            "fevereiro": "02",
            "março": "03",
            "abril": "04",
            "maio": "05",
            "junho": "06",
            "julho": "07",
            "agosto": "08",
            "setembro": "09",
            "outubro": "10",
            "novembro": "11",
            "dezembro": "12",
        }

        for month_name, month_num in months.items():
            if month_name in date_str.lower():
                parts = date_str.split()
                if len(parts) >= 3:
                    day = parts[0].strip()
                    year = parts[-1].strip()
                    return f"{day.zfill(2)}/{month_num}/{year}"

        return date_str

    def _combine_index_info(self, index_info: str) -> str:
        """Combina informações de index e atualização monetária"""
        return index_info

    def _calculate_total_amortization(
        self, fluxos_percentuais_entity: Optional[ExtractedEntity]
    ) -> str:
        """Calcula o total de amortização somando todos os percentuais"""
        if not fluxos_percentuais_entity or not fluxos_percentuais_entity.value:
            return "0,00%"

        try:
            percentages = fluxos_percentuais_entity.value.split(",")
            total = 0.0

            for perc in percentages:
                clean_perc = perc.strip().replace("%", "").replace(",", ".")
                if clean_perc:
                    total += float(clean_perc)

            return f"{total:.2f}%".replace(".", ",")

        except (ValueError, AttributeError):
            return "0,00%"


class DocumentTextExtractorService:
    """Serviço responsável pela extração de texto de arquivos PDF, TXT, DOC e DOCX"""

    def __init__(self, req: func.HttpRequest):
        self.supported_extensions = [".pdf", ".txt", ".doc", ".docx"]
        self.token_counter = TokenCounter()
        self.req = req
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n"],
        )
        
<<<<<<< Updated upstream
        self._ocr_reader = None
        
        self.MAX_PAGES_PER_BATCH = 20
        self.MAX_TOTAL_PAGES_OCR = 100
        self.MAX_FILE_SIZE_MB = 50
        self.OCR_TIMEOUT_PER_PAGE = 15

    def _get_ocr_reader(self):
        """Inicializa o EasyOCR reader apenas quando necessário"""
        if self._ocr_reader is None:
            try:
                logging.info("Inicializando EasyOCR...")
                
                import easyocr
                
                self._ocr_reader = easyocr.Reader(
                    ['pt', 'en'], 
                    gpu=False,
                    verbose=False,
                    download_enabled=True,
                    detector=True,
                    recognizer=True
                )
                
                logging.info("EasyOCR inicializado com sucesso")
                
            except Exception as e:
                logging.error(f"Erro ao inicializar EasyOCR: {str(e)}")
                raise Exception(f"Falha na inicialização do EasyOCR: {str(e)}")
        
        return self._ocr_reader

    def _is_pdf_text_extractable(self, file_content: bytes, filename: str) -> bool:
        """Detecta se um PDF contém texto extraível ou se é uma imagem escaneada"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)
            
            total_text_length = 0
            total_chars_analyzed = 0
            pages_with_text = 0
            pages_with_images = 0
            
            pages_to_analyze = min(3, len(doc))
            
            for page_num in range(pages_to_analyze):
                page = doc.load_page(page_num)
                
                text = page.get_text().strip()
                if text:
                    total_text_length += len(text)
                    pages_with_text += 1
                    
                    non_space_chars = len([c for c in text if not c.isspace()])
                    total_chars_analyzed += non_space_chars
                
                image_list = page.get_images()
                if image_list:
                    pages_with_images += 1
            
            doc.close()
            
            if total_chars_analyzed == 0:
                logging.info(f"{filename}: Nenhum texto extraível encontrado - PDF escaneado")
                return False
            
            avg_chars_per_page = total_chars_analyzed / pages_to_analyze if pages_to_analyze > 0 else 0
            if avg_chars_per_page < 100:
                logging.info(f"{filename}: Texto insuficiente ({avg_chars_per_page:.1f} chars/página) - PDF escaneado")
                return False
            
            if pages_with_images > 0 and avg_chars_per_page < 300:
                logging.info(f"{filename}: PDF com imagens e pouco texto - usando OCR para garantir")
                return False
            
            logging.info(f"{filename}: PDF contém texto extraível ({avg_chars_per_page:.1f} chars/página)")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao verificar se PDF é text-extractable {filename}: {str(e)}")
            return True
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}")

    def _should_use_sampling_strategy(self, total_pages: int, file_size_mb: float) -> bool:
        """Determina se deve usar estratégia de amostragem para documentos grandes"""
        return total_pages > self.MAX_TOTAL_PAGES_OCR or file_size_mb > self.MAX_FILE_SIZE_MB

    def _get_strategic_pages(self, total_pages: int, max_pages: int = 50) -> List[int]:
        """Seleciona páginas estratégicas de documentos grandes"""
        if total_pages <= max_pages:
            return list(range(total_pages))
        
        strategic_pages = []
        
        strategic_pages.extend(range(min(10, total_pages)))
        
        if total_pages > 20:
            middle_start = total_pages // 3
            middle_end = 2 * total_pages // 3
            middle_sample = max_pages - 20
            
            if middle_sample > 0:
                middle_pages = range(middle_start, middle_end)
                step = max(1, len(middle_pages) // middle_sample)
                strategic_pages.extend(middle_pages[::step][:middle_sample])
        
        if total_pages > 10:
            last_pages_start = max(total_pages - 10, max(strategic_pages) + 1)
            strategic_pages.extend(range(last_pages_start, total_pages))
        
        return sorted(list(set(strategic_pages)))

    def extract_text_from_pdf_with_ocr(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """Extrai texto de PDF usando EasyOCR com estratégia otimizada para documentos grandes"""
        documents = []
        temp_path = None
        
        try:
            file_size_mb = len(file_content) / (1024 * 1024)
            logging.info(f"Iniciando OCR: {filename} ({file_size_mb:.1f}MB)")
            
            if file_size_mb > 100:
                logging.warning(f"Arquivo muito grande ({file_size_mb:.1f}MB), pode haver problemas de memória")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)
            total_pages = len(doc)
            
            use_sampling = self._should_use_sampling_strategy(total_pages, file_size_mb)
            
            if use_sampling:
                pages_to_process = self._get_strategic_pages(total_pages, self.MAX_TOTAL_PAGES_OCR)
                logging.info(f"Estratégia: {len(pages_to_process)} páginas estratégicas de {total_pages} total")
            else:
                pages_to_process = list(range(min(self.MAX_TOTAL_PAGES_OCR, total_pages)))
                logging.info(f"Processando: {len(pages_to_process)} de {total_pages} páginas")
            
            try:
                reader = self._get_ocr_reader()
            except Exception as e:
                logging.error(f"Erro ao inicializar OCR: {str(e)}")
                doc.close()
                return self.extract_text_from_pdf_normal(file_content, filename)
            
            total_processed = 0
            start_time = time.time()
            
            for batch_start in range(0, len(pages_to_process), self.MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + self.MAX_PAGES_PER_BATCH, len(pages_to_process))
                batch_pages = pages_to_process[batch_start:batch_end]
                batch_num = batch_start // self.MAX_PAGES_PER_BATCH + 1
                total_batches = (len(pages_to_process) + self.MAX_PAGES_PER_BATCH - 1) // self.MAX_PAGES_PER_BATCH
                
                logging.info(f"Lote {batch_num}/{total_batches}: Páginas {batch_pages[0]+1}-{batch_pages[-1]+1}")
                
                for page_num in batch_pages:
                    try:
                        page_start_time = time.time()
                        
                        page = doc.load_page(page_num)
                        
                        if total_pages > 200:
                            mat = fitz.Matrix(1.5, 1.5)
                        else:
                            mat = fitz.Matrix(2.0, 2.0)
                        
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        
                        if len(img_data) > 5 * 1024 * 1024:
                            mat = fitz.Matrix(1.0, 1.0)
                            pix = page.get_pixmap(matrix=mat)
                            img_data = pix.tobytes("png")
                        
                        try:
                            results = reader.readtext(img_data, detail=0, width_ths=0.7, height_ths=0.7)
                            
                            page_processing_time = time.time() - page_start_time
                            if page_processing_time > self.OCR_TIMEOUT_PER_PAGE:
                                logging.warning(f"Página {page_num + 1} demorou {page_processing_time:.1f}s")
                            
                        except Exception as ocr_error:
                            logging.error(f"OCR falhou na página {page_num + 1}: {str(ocr_error)}")
                            continue
                        
                        if results:
                            filtered_results = [text.strip() for text in results if text.strip() and len(text.strip()) > 2]
                            
                            if filtered_results:
                                page_text = '\n'.join(filtered_results)
                                
                                document = DocumentModel(
                                    page_content=page_text,
                                    page_number=page_num + 1,
                                    source=filename,
                                    file_type="pdf_ocr_strategic" if use_sampling else "pdf_ocr",
                                    character_count=len(page_text),
                                    token_count=self._estimate_tokens(page_text),
                                )
                                documents.append(document)
                                total_processed += 1
                                
                                if total_processed % 10 == 0 or total_processed % 5 == 0:
                                    elapsed_time = time.time() - start_time
                                    avg_time_per_page = elapsed_time / total_processed
                                    estimated_remaining = avg_time_per_page * (len(pages_to_process) - total_processed)
                                    
                                    progress_percent = (total_processed / len(pages_to_process)) * 100
                                    progress_bar = self._create_progress_bar(progress_percent)
                                    
                                    logging.info(f"Progresso: {progress_bar} {total_processed}/{len(pages_to_process)} páginas " +
                                               f"({progress_percent:.1f}%) - {avg_time_per_page:.1f}s/página - " +
                                               f"ETA: {estimated_remaining/60:.1f}min")
                    
                    except Exception as e:
                        logging.error(f"Erro na página {page_num + 1}: {str(e)}")
                        continue
                
                if batch_end < len(pages_to_process):
                    time.sleep(0.5)
            
            doc.close()
            
            total_time = time.time() - start_time
            total_chars = sum(len(doc.page_content) for doc in documents)
            
            logging.info(f"OCR Completo: {len(documents)} páginas, {total_chars:,} caracteres, " +
                        f"{total_time/60:.1f}min total")
            
            if not documents:
                logging.warning(f"Nenhum texto extraído via OCR")
                return self.extract_text_from_pdf_normal(file_content, filename)
=======
        self.ocr_service = OCRService()

    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """Extrai texto de arquivos PDF, detectando automaticamente se precisa de OCR"""
        
        is_text_extractable = self.ocr_service.is_pdf_text_extractable(file_content, filename)
        
        if not is_text_extractable:
            logging.info(f"{filename}: PDF detectado como escaneado, usando EasyOCR")
            return self.ocr_service.extract_text_with_ocr(file_content, filename)
        
        try:
            documents = self.extract_text_from_pdf_normal(file_content, filename)
            
            if not documents:
                logging.info(f"{filename}: Extração normal não retornou texto, tentando OCR")
                return self.ocr_service.extract_text_with_ocr(file_content, filename)
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            avg_chars_per_page = total_chars / len(documents) if documents else 0
            
            if avg_chars_per_page < 50:
                logging.info(f"{filename}: Texto extraído muito escasso ({avg_chars_per_page:.1f} chars/página), tentando OCR")
                try:
                    ocr_documents = self.ocr_service.extract_text_with_ocr(file_content, filename)
                    
                    if ocr_documents:
                        ocr_total_chars = sum(len(doc.page_content) for doc in ocr_documents)
                        if ocr_total_chars > total_chars * 1.5:
                            logging.info(f"{filename}: OCR produziu mais texto ({ocr_total_chars} vs {total_chars}), usando OCR")
                            return ocr_documents
                except Exception:
                    logging.info(f"{filename}: OCR falhou, usando extração normal")
>>>>>>> Stashed changes
            
            return documents
            
        except Exception as e:
<<<<<<< Updated upstream
            logging.error(f"Erro geral na extração OCR: {str(e)}")
            return self.extract_text_from_pdf_normal(file_content, filename)
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}")

    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Cria uma barra de progresso visual"""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
=======
            logging.error(f"Erro na extração normal de {filename}: {str(e)}")
            logging.info(f"Tentando OCR como fallback para {filename}")
            return self.ocr_service.extract_text_with_ocr(file_content, filename)
>>>>>>> Stashed changes

    def extract_text_from_pdf_normal(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """Extração normal de PDF (texto embutido)"""
        documents = []
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)

            logging.info(f"Processando PDF com extração normal: {filename} com {len(doc)} páginas")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()

                if text.strip():
                    document = DocumentModel(
                        page_content=text.strip(),
                        page_number=page_num + 1,
                        source=filename,
                        file_type="pdf",
                        character_count=len(text.strip()),
                        token_count=self._estimate_tokens(text.strip()),
                    )
                    documents.append(document)

            doc.close()
            logging.info(f"Extração normal concluída: {len(documents)} páginas com texto de {filename}")

        except Exception as e:
            logging.error(f"Erro ao extrair texto do PDF {filename}: {str(e)}")
            raise Exception(f"Falha na extração de texto: {str(e)}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}")

        return documents

<<<<<<< Updated upstream
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """Extrai texto de arquivos PDF, detectando automaticamente se precisa de OCR"""
        
        is_text_extractable = self._is_pdf_text_extractable(file_content, filename)
        
        if not is_text_extractable:
            logging.info(f"{filename}: PDF detectado como escaneado, usando EasyOCR")
            return self.extract_text_from_pdf_with_ocr(file_content, filename)
        
        try:
            documents = self.extract_text_from_pdf_normal(file_content, filename)
            
            if not documents:
                logging.info(f"{filename}: Extração normal não retornou texto, tentando OCR")
                return self.extract_text_from_pdf_with_ocr(file_content, filename)
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            avg_chars_per_page = total_chars / len(documents) if documents else 0
            
            if avg_chars_per_page < 50:
                logging.info(f"{filename}: Texto extraído muito escasso ({avg_chars_per_page:.1f} chars/página), tentando OCR")
                ocr_documents = self.extract_text_from_pdf_with_ocr(file_content, filename)
                
                if ocr_documents:
                    ocr_total_chars = sum(len(doc.page_content) for doc in ocr_documents)
                    if ocr_total_chars > total_chars * 1.5:
                        logging.info(f"{filename}: OCR produziu mais texto ({ocr_total_chars} vs {total_chars}), usando OCR")
                        return ocr_documents
            
            return documents
            
        except Exception as e:
            logging.error(f"Erro na extração normal de {filename}: {str(e)}")
            logging.info(f"Tentando OCR como fallback para {filename}")
            return self.extract_text_from_pdf_with_ocr(file_content, filename)

=======
>>>>>>> Stashed changes
    def process_file_upload(self) -> BulkFileUploadResponse:
        """Processa upload de múltiplos arquivos com otimizações para documentos grandes"""
        files_data = self._extract_files_from_request()

        if not files_data:
            raise ValueError("Nenhum arquivo foi enviado")

        processed_files = []
        total_size_mb = sum(len(content) for content, _ in files_data) / (1024 * 1024)
        
        logging.info(f"Iniciando processamento: {len(files_data)} arquivo(s), {total_size_mb:.1f}MB total")
        
        if len(files_data) > 10:
            logging.warning(f"Muitos arquivos ({len(files_data)}), pode haver timeout")
        
        if total_size_mb > 200:
            logging.warning(f"Volume muito grande ({total_size_mb:.1f}MB), pode haver problemas de memória")

        for idx, (file_content, filename) in enumerate(files_data):
            file_size_mb = len(file_content) / (1024 * 1024)
            
            if not self.is_supported_file(filename):
                supported_formats = ", ".join(self.supported_extensions)
                raise ValueError(
                    f"Arquivo {filename}: Apenas arquivos {supported_formats} são suportados"
                )

            logging.info(f"Arquivo [{idx+1}/{len(files_data)}] {filename} ({file_size_mb:.1f}MB)")

            try:
                file_response = self._process_single_file(file_content, filename)
                processed_files.append(file_response)

            except Exception as e:
                logging.error(f"Erro ao processar {filename}: {str(e)}")
                error_response = FileUploadResponse(
                    success=False,
                    filename=filename,
                    file_type=self._get_file_extension(filename).replace(".", ""),
                    documents_count=0,
                    total_characters=0,
                    total_tokens=0,
                    processing_time_ms=0,
                    token_summary=self.token_counter.get_summary(),
                    documents=[],
                    message=f"Erro ao processar arquivo: {str(e)}",
                )
                processed_files.append(error_response)

        success = any(file_resp.success for file_resp in processed_files)
        
        successful_files = sum(1 for f in processed_files if f.success)
        logging.info(f"Processamento finalizado: {successful_files}/{len(files_data)} arquivos processados com sucesso")
        
        response = BulkFileUploadResponse(success=success, files=processed_files)
        return response

<<<<<<< Updated upstream
    def _process_single_file(
        self, file_content: bytes, filename: str
    ) -> FileUploadResponse:
        file_extension = self._get_file_extension(filename)
        
        if file_extension == ".pdf":
            documents = self.extract_text_from_pdf(file_content, filename)
        elif file_extension == ".txt":
            documents = self.extract_text_from_txt(file_content, filename)
        elif file_extension in [".doc", ".docx"]:
            documents = self.extract_text_from_docx(file_content, filename)
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {file_extension}")
            
        all_texts = [doc.page_content for doc in documents]
        all_chunks = self.split_pages(all_texts)
        total_tokens = self.token_counter.log_embedding_usage(all_chunks)
        token_summary = self.token_counter.get_summary()

        chunk_data = []
        for idx, chunk in enumerate(all_chunks):
            chunk_data.append(
                {
                    "chunk_id": idx + 1,
                    "content": base64.b64encode(chunk.encode("utf-8")).decode("utf-8"),
                    "tokens": len(chunk.split()),
                    "page": self._get_chunk_page(idx, len(documents)),
                }
            )

        response = FileUploadResponse(
            success=True,
            filename=filename,
            file_type=file_extension.replace(".", ""),
            documents_count=len(chunk_data),
            total_characters=sum(len(chunk) for chunk in all_chunks),
            total_tokens=total_tokens,
            processing_time_ms=0,
            token_summary=token_summary,
            documents=chunk_data,
        )

        return response

=======
>>>>>>> Stashed changes
    def _get_file_extension(self, filename: str) -> str:
        """Extrai a extensão do arquivo"""
        return os.path.splitext(filename.lower())[1]

    def extract_text_from_txt(
        self, file_content: bytes, filename: str
    ) -> List[DocumentModel]:
        """Extrai texto de arquivos TXT"""
        documents = []
        
        try:
            text = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise Exception("Não foi possível decodificar o arquivo TXT")

            logging.info(f"Processando TXT: {filename}")

            if text.strip():
                document = DocumentModel(
                    page_content=text.strip(),
                    page_number=1,
                    source=filename,
                    file_type="txt",
                    character_count=len(text.strip()),
                    token_count=self._estimate_tokens(text.strip()),
                )
                documents.append(document)

            logging.info(f"Extração TXT concluída: {filename}")

        except Exception as e:
            logging.error(f"Erro ao extrair texto do TXT {filename}: {str(e)}")
            raise Exception(f"Falha na extração de texto: {str(e)}")

        if not documents:
            raise Exception("Nenhum texto foi extraído do arquivo TXT")

        return documents

    def extract_text_from_docx(
        self, file_content: bytes, filename: str
    ) -> List[DocumentModel]:
        """Extrai texto de arquivos DOC e DOCX"""
        documents = []

        try:
            from docx import Document as DocxDocument
            import io

            if filename.lower().endswith('.docx'):
                doc_stream = io.BytesIO(file_content)
                doc = DocxDocument(doc_stream)
                
                full_text = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        full_text.append(paragraph.text.strip())
                
                text = '\n'.join(full_text)
                
            else:
                try:
                    text = file_content.decode('utf-8', errors='ignore')
                except Exception as e:
                    logging.error(f"Erro ao decodificar DOCX/DOC {filename}: {str(e)}")
                    text = file_content.decode('latin-1', errors='ignore')

            logging.info(f"Processando DOCX/DOC: {filename}")

            if text.strip():
                document = DocumentModel(
                    page_content=text.strip(),
                    page_number=1,
                    source=filename,
                    file_type="docx" if filename.lower().endswith('.docx') else "doc",
                    character_count=len(text.strip()),
                    token_count=self._estimate_tokens(text.strip()),
                )
                documents.append(document)

            logging.info(f"Extração DOCX/DOC concluída: {filename}")

        except Exception as e:
            logging.error(f"Erro ao extrair texto do DOCX/DOC {filename}: {str(e)}")
            raise Exception(f"Falha na extração de texto: {str(e)}")

        if not documents:
            raise Exception("Nenhum texto foi extraído do arquivo DOCX/DOC")

        return documents

    def _get_chunk_page(self, chunk_index: int, total_pages: int) -> int:
        if total_pages == 0:
            return 1
        page = (chunk_index % total_pages) + 1
        return page

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def is_supported_file(self, filename: str) -> bool:
        file_extension = self._get_file_extension(filename)
        return file_extension in self.supported_extensions

    def _extract_files_from_request(self) -> List[tuple[bytes, str]]:
        files_data = []

        files = self.req.files
        if files:
            for key, file_item in files.items():
                content = file_item.read()
                filename = file_item.filename or f"arquivo_{key}.pdf"
                files_data.append((content, filename))
            return files_data

        try:
            body_json = self.req.get_json()
            if body_json:
                if "files" in body_json and isinstance(body_json["files"], list):
                    for file_info in body_json["files"]:
                        if "file_content" in file_info:
                            file_content = base64.b64decode(file_info["file_content"])
                            filename = file_info.get("filename", "document.pdf")
                            files_data.append((file_content, filename))
                    return files_data

                elif "file_content" in body_json:
                    file_content = base64.b64decode(body_json["file_content"])
                    filename = body_json.get("filename", "document.pdf")
                    files_data.append((file_content, filename))
                    return files_data

        except Exception as e:
            logging.error(f"Erro ao processar JSON: {str(e)}")

        return files_data

    def split_pages(self, pages: list[str]) -> list[str]:
        joined = "\n".join(pages)
        return self.splitter.split_text(joined)

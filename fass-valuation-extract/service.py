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

class DocumentEntityExtractor:
    """Extrator de entidades específicas de documentos usando embeddings"""

    def __init__(self):
        self.token_counter = TokenCounter()

        # Substitua a configuração do LLM atual por esta:
        base_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        )

        # Aplique o patch do Instructor
        self.llm = instructor.from_openai(base_client)

        # Mantenha os embeddings como estão
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
            Você é um(a) **analista sênior de contratos financeiros**.  
            Sua tarefa é **LER** o texto abaixo e **DEVOLVER** exatamente **um** objeto
            JSON com os nove campos pedidos, **somente strings** (nunca arrays),
            no formato mostrado depois da lista.

            ────────────────────────── CONTEXTO ──────────────────────────
            {full_context}

            REGRAS OBRIGATÓRIAS
            1. Copie o conteúdo **exatamente como está no contrato** – não traduza
            nem reescreva números, índices ou datas.
            2. Se o item não existir, responda **"NÃO ENCONTRADO"**.
            3. Se houver mais de um valor para o mesmo item, una-os em **uma única
            string separada por vírgulas**, mantendo a ordem em que aparecem.
            4. Retorne apenas o JSON válido (sem comentários, sem texto antes ou
            depois).

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
                        page_references=sorted(list(all_page_refs)),
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
                        "% Amort": "100,00%",  # Assumindo amortização total no vencimento
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


class UploadFileService:
    """Serviço responsável pela extração de texto de arquivos PDF, TXT, DOC e DOCX"""

    def __init__(self, req: func.HttpRequest):
        self.supported_extensions = [".pdf", ".txt", ".doc", ".docx"]
        self.token_counter = TokenCounter()
        self.req = req
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_file_upload(self) -> BulkFileUploadResponse:
        files_data = self._extract_files_from_request()

        if not files_data:
            raise ValueError("Nenhum arquivo foi enviado")

        processed_files = []
        all_filenames = []

        for file_content, filename in files_data:
            if not self.is_supported_file(filename):
                supported_formats = ", ".join(self.supported_extensions)
                raise ValueError(
                    f"Arquivo {filename}: Apenas arquivos {supported_formats} são suportados"
                )

            logging.info(
                f"Processando arquivo: {filename}, tamanho: {len(file_content)} bytes"
            )

            try:
                file_response = self._process_single_file(file_content, filename)
                processed_files.append(file_response)
                all_filenames.append(filename)

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

        response = BulkFileUploadResponse(success=success, files=processed_files)

        return response

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
        temp_path = None

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
                except:
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

    def extract_text_from_pdf(
        self, file_content: bytes, filename: str
    ) -> List[DocumentModel]:
        documents = []
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)

            logging.info(f"Processando PDF: {filename} com {len(doc)} páginas")

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
            logging.info(
                f"Extração concluída: {len(documents)} páginas com texto de {filename}"
            )

        except Exception as e:
            logging.error(f"Erro ao extrair texto do PDF {filename}: {str(e)}")
            raise Exception(f"Falha na extração de texto: {str(e)}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}")

        if not documents:
            raise Exception("Nenhum texto foi extraído do PDF")

        return documents

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

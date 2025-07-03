from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import logging as logger
import os
import time
import base64
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from ..models.entites_models import (
    ExtractedEntity,
    ContractEntity,
    DocumentExtractionResult,
)
from ..utils.token_counter import TokenCounter
import re
import instructor
from openai import AzureOpenAI
import json

class ContractEntitiesResponse(BaseModel):

    atualizacao_monetaria: str = Field(
        default="NÃO ENCONTRADO",
        description="Índice que corrige o principal (IPCA, IGP-M, SELIC, etc.)",
    )

    juros_remuneratorios: str = Field(
        default="NÃO ENCONTRADO",
        description="Indexador principal dos juros (DI+, CDI+, IPCA+, etc.)",
    )

    spread_fixo: str = Field(
        default="NÃO ENCONTRADO",
        description="Percentual adicional sobre o indexador principal",
    )

    base_calculo: str = Field(
        default="NÃO ENCONTRADO",
        description="Metodologia de cálculo de juros (252, 365, ACT/360)",
    )

    data_emissao: str = Field(
        default="NÃO ENCONTRADO",
        description="Data(s) de emissão do título no formato DD/MM/AAAA",
    )

    data_vencimento: str = Field(
        default="NÃO ENCONTRADO",
        description="Data(s) de vencimento no formato DD/MM/AAAA",
    )

    valor_nominal_unitario: str = Field(
        default="NÃO ENCONTRADO", description="Valor de face por título/cota"
    )

    fluxos_pagamento: str = Field(
        default="NÃO ENCONTRADO",
        description="Datas do cronograma de pagamentos separadas por vírgula",
    )

    fluxos_percentuais: str = Field(
        default="NÃO ENCONTRADO",
        description="Percentuais de amortização separados por vírgula",
    )


def get_required_env_var(var_name: str, default_value: str = None) -> str:
    value = os.getenv(var_name, default_value)
    if not value:
        logger.error(f"Variável de ambiente obrigatória não encontrada: {var_name}")
        raise ValueError(f"Variável de ambiente {var_name} não configurada")
    return value


try:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    logger.info("Todas as variáveis de ambiente carregadas com sucesso")
except ValueError as e:
    logger.error(f"Erro na configuração: {e}")
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
        logger.info(
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

        logger.info(f"Banco vetorial criado com {len(documents)} chunks")
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
                    confidence = self._calculate_confidence_from_context(value, full_context)
                    
                    entity = ExtractedEntity(
                        entity_type=field_name,
                        value=value.strip(),
                        confidence=confidence,
                        page_references=sorted(list(all_page_refs)),
                        context=(
                            full_context[:500] + "..." if len(full_context) > 500 else full_context
                        ),
                    )
                    setattr(contract_entities, field_name, entity)
                    entities_found += 1
                    logger.info(f"Entidade '{field_name}' encontrada: {value.strip()[:50]}...")

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
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
            logger.error(f"Erro na extração com Instructor: {e}")
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
            logger.info(f"DEBUG: Iniciando criação de tabela FAAS para {filename}")
            logger.info(
                f"DEBUG: fluxos_pagamento existe: {contract_entities.fluxos_pagamento is not None}"
            )
            if contract_entities.fluxos_pagamento:
                logger.info(
                    f"DEBUG: fluxos_pagamento valor: '{contract_entities.fluxos_pagamento.value}'"
                )

            logger.info(
                f"DEBUG: fluxos_percentuais existe: {contract_entities.fluxos_percentuais is not None}"
            )
            if contract_entities.fluxos_percentuais:
                logger.info(
                    f"DEBUG: fluxos_percentuais valor: '{contract_entities.fluxos_percentuais.value}'"
                )

            fluxos_pagamento = self._extract_dates_from_fluxos(
                contract_entities.fluxos_pagamento
            )
            fluxos_percentuais = self._extract_percentages_from_fluxos(
                contract_entities.fluxos_percentuais
            )

            logger.info(f"DEBUG: fluxos_pagamento extraídos: {fluxos_pagamento}")
            logger.info(f"DEBUG: fluxos_percentuais extraídos: {fluxos_percentuais}")

            valor_nominal = self._get_entity_value(
                contract_entities.valor_nominal_unitario, ""
            )
            index_info = self._get_entity_value(
                contract_entities.juros_remuneratorios, ""
            )

            index_completo = self._combine_index_info(index_info)

            if not fluxos_pagamento:
                logger.info(
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
                    logger.info("DEBUG: Linha básica criada com sucesso")
            else:
                logger.info(
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

            logger.info(
                f"Tabela FAAS criada com {len(tabela_faas)} linhas para o arquivo {filename}"
            )
            if tabela_faas:
                logger.info(f"Primeira linha da tabela FAAS: {tabela_faas[0]}")
            return tabela_faas

        except Exception as e:
            logger.error(f"Erro ao criar tabela FAAS para {filename}: {e}")
            import traceback

            logger.error(f"Traceback completo: {traceback.format_exc()}")
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

            logger.info(f"Linha de resumo FAAS criada para o arquivo {filename}")
            logger.info(f"Linha de resumo FAAS: {linha_resumo}")
            return linha_resumo

        except Exception as e:
            logger.error(f"Erro ao criar linha de resumo FAAS para {filename}: {e}")
            return {}

    def _extract_dates_from_fluxos(
        self, fluxos_entity: Optional[ExtractedEntity]
    ) -> List[str]:
        """Extrai lista de datas dos fluxos de pagamento"""
        logger.info("DEBUG: _extract_dates_from_fluxos chamado")
        if not fluxos_entity or not fluxos_entity.value:
            logger.info("DEBUG: Nenhum fluxo de pagamento encontrado ou valor vazio")
            return []

        logger.info(f"DEBUG: Valor dos fluxos de pagamento: '{fluxos_entity.value}'")

        date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
        dates = re.findall(date_pattern, fluxos_entity.value)

        logger.info(f"DEBUG: Datas encontradas com regex: {dates}")

        if not dates:
            logger.info(
                "DEBUG: Nenhuma data encontrada com regex, tentando extração flexível"
            )
            parts = fluxos_entity.value.split(",")
            for part in parts:
                part = part.strip()
                flexible_pattern = r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
                found_dates = re.findall(flexible_pattern, part)
                if found_dates:
                    dates.extend(found_dates)

            logger.info(f"DEBUG: Datas encontradas com extração flexível: {dates}")

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

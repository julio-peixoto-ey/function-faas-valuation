import os
import base64
import re
import json
import logging
import asyncio
import time
from enum import Enum

import fitz
import tiktoken
import tempfile
import numpy as np
import instructor
import concurrent.futures
import azure.functions as func
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class ExtractedEntity:
    """Entidade extraída de um documento"""

    entity_type: str
    value: str
    confidence: float
    page_references: List[int]
    context: Optional[str] = None


@dataclass
class ContractEntity:
    """Entidades específicas de contratos financeiros"""

    atualizacao_monetaria: Optional[ExtractedEntity] = None
    juros_remuneratorios: Optional[ExtractedEntity] = None
    spread_fixo: Optional[ExtractedEntity] = None
    base_calculo: Optional[ExtractedEntity] = None
    data_emissao: Optional[ExtractedEntity] = None
    data_vencimento: Optional[ExtractedEntity] = None
    valor_nominal_unitario: Optional[ExtractedEntity] = None
    fluxos_pagamento: Optional[ExtractedEntity] = None
    fluxos_percentuais: Optional[ExtractedEntity] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    api_calls: int = 0


@dataclass
class CostBreakdown:
    llm_input_cost: float
    llm_output_cost: float
    embedding_cost: float

    @property
    def total_cost(self) -> float:
        return self.llm_input_cost + self.llm_output_cost + self.embedding_cost


@dataclass
class TokenSummary:
    usage: TokenUsage
    cost_breakdown: CostBreakdown

    @property
    def total_tokens(self) -> int:
        return (
            self.usage.input_tokens
            + self.usage.output_tokens
            + self.usage.embedding_tokens
        )

    @property
    def total_cost_usd(self) -> float:
        return self.cost_breakdown.total_cost

    def to_dict(self) -> Dict:
        return {
            "api_calls": self.usage.api_calls,
            "total_tokens": self.total_tokens,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "embedding_tokens": self.usage.embedding_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 4),
            "breakdown": {
                "llm_input_cost": round(self.cost_breakdown.llm_input_cost, 4),
                "llm_output_cost": round(self.cost_breakdown.llm_output_cost, 4),
                "embedding_cost": round(self.cost_breakdown.embedding_cost, 4),
            },
        }


@dataclass
class DocumentExtractionResult:
    """Resultado da extração de um documento"""

    filename: str
    success: bool
    contract_entities: ContractEntity
    processing_time_ms: int
    token_summary: TokenSummary
    error_message: Optional[str] = None


@dataclass
class ExtractionResponse:
    """Resposta da API de extração"""

    success: bool
    documents: List[DocumentExtractionResult]
    total_processing_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "documents": [
                {
                    "filename": doc.filename,
                    "success": doc.success,
                    "contract_entities": self._contract_entities_to_dict(
                        doc.contract_entities
                    ),
                    "processing_time_ms": doc.processing_time_ms,
                    "token_summary": doc.token_summary.to_dict(),
                    "error_message": doc.error_message,
                }
                for doc in self.documents
            ],
            "total_processing_time_ms": self.total_processing_time_ms,
        }

    def _contract_entities_to_dict(self, entities: ContractEntity) -> Dict[str, Any]:
        result = {}
        for field_name in entities.__dataclass_fields__:
            entity = getattr(entities, field_name)
            if entity:
                result[field_name] = {
                    "value": entity.value,
                    "confidence": entity.confidence,
                    "page_references": entity.page_references,
                    "context": entity.context,
                }
            else:
                result[field_name] = None
        return result


class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"


@dataclass
class DocumentModel:
    page_content: str
    page_number: int
    source: str
    file_type: str
    character_count: int
    token_count: int

    @property
    def preview(self) -> str:
        return (
            self.page_content[:200] + "..."
            if len(self.page_content) > 200
            else self.page_content
        )


@dataclass
class FileUploadResponse:
    success: bool
    filename: str
    file_type: str
    documents_count: int
    total_characters: int
    total_tokens: int
    processing_time_ms: int
    token_summary: TokenSummary
    documents: List[Dict[str, Any]]
    message: str = "Arquivo processado com sucesso"

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "message": self.message,
            "filename": self.filename,
            "file_type": self.file_type,
            "documents_count": self.documents_count,
            "total_characters": self.total_characters,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms,
            "cost_summary": {
                "total_cost_usd": self.token_summary.total_cost_usd,
                "total_tokens": self.token_summary.total_tokens,
            },
            "token_summary": self.token_summary.to_dict(),
            "documents": self.documents,
        }


@dataclass
class BulkFileUploadResponse:
    success: bool
    files: List[FileUploadResponse]

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "files": [file.to_dict() for file in self.files],
        }


@dataclass
class ErrorResponse:
    success: bool = False
    error: str = ""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        result = {"success": self.success, "error": self.error}
        if self.error_code:
            result["error_code"] = self.error_code
        if self.details:
            result["details"] = self.details
        return result

    @property
    def total_cost(self) -> float:
        return self.llm_input_cost + self.llm_output_cost + self.embedding_cost


class CustomJSONEncoder(json.JSONEncoder):
    """Encoder JSON personalizado para lidar com tipos numpy e float32"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_dumps(obj, **kwargs):
    """Função auxiliar para serialização JSON segura"""
    kwargs.setdefault("cls", CustomJSONEncoder)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)


def create_error_result(file_response, error_message=""):
    token_counter = TokenCounter()

    return DocumentExtractionResult(
        filename=file_response.filename,
        success=False,
        contract_entities=ContractEntity(),
        processing_time_ms=0,
        token_summary=token_counter.get_summary(),
        error_message=error_message or f"Falha no upload: {file_response.message}",
    )


async def _process_files_parallel(extractor, files, max_workers=3):
    def process_single_file(file_response):
        if not file_response.success:
            return create_error_result(file_response)

        return extractor.extract_all_entities(
            file_response.documents, file_response.filename
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(executor, process_single_file, file_resp)
            for file_resp in files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        extraction_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Erro processando arquivo {files[i].filename}: {result}")
                extraction_results.append(create_error_result(files[i], str(result)))
            else:
                extraction_results.append(result)

        return extraction_results


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function processando requisição")

    try:
        if req.method == "GET":
            return _handle_info_request()
        elif req.method == "POST":
            return _handle_file_upload_and_extraction(req)
        else:
            return _create_error_response("Método não suportado", 405)

    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return _create_error_response(f"Erro interno: {str(e)}", 500)


def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "API para extração de entidades de um arquivo",
        "status": "healthy",
        "timestamp": time.time(),
        "endpoints": {
            "upload_and_extract": "POST /valuation/extract",
            "health": "GET /valuation/extract",
        },
        "extracted_entities": [
            "atualizacao_monetaria",
            "juros_remuneratorios",
            "spread_fixo",
            "base_calculo",
            "data_emissao",
            "data_vencimento",
            "valor_nominal_unitario",
            "fluxos_pagamento",
            "fluxos_percentuais",
        ],
        "connections": {
            "azure_openai": "ok" if os.getenv("AZURE_OPENAI_ENDPOINT") else "missing",
            "embeddings": (
                "ok" if os.getenv("AZURE_EMBEDDING_DEPLOYMENT") else "missing"
            ),
        },
    }

    try:
        extractor = DocumentEntityExtractor()
        info["warm_up"] = "completed"
    except Exception as e:
        info["warm_up"] = f"failed: {str(e)}"

    return func.HttpResponse(
        safe_json_dumps(info), mimetype="application/json", status_code=200
    )


def _handle_file_upload_and_extraction(req: func.HttpRequest) -> func.HttpResponse:
    upload_start_time = time.time()

    upload_file_service = UploadFileService(req)

    try:
        upload_response = upload_file_service.process_file_upload()

        if not upload_response.success:
            return func.HttpResponse(
                safe_json_dumps(upload_response.to_dict()),
                mimetype="application/json",
                status_code=400,
            )

        logging.info(f"Upload concluído com {len(upload_response.files)} arquivo(s)")

        extractor = DocumentEntityExtractor()

        extraction_results = asyncio.run(
            _process_files_parallel(extractor, upload_response.files)
        )

        tabelas_faas = []
        resumo_faas = []

        for extraction_result in extraction_results:
            if extraction_result.success:
                tabela_contrato = extractor.create_tabela_faas(
                    extraction_result.contract_entities, extraction_result.filename
                )
                tabelas_faas.extend(tabela_contrato)
                linha_resumo = extractor.create_row_faas_resumo(
                    extraction_result.contract_entities, extraction_result.filename
                )
                if linha_resumo:
                    resumo_faas.append(linha_resumo)
                successful_entities = sum(
                    1
                    for field_name in extraction_result.contract_entities.__dataclass_fields__
                    if getattr(extraction_result.contract_entities, field_name)
                    is not None
                )
                logging.info(
                    f"Arquivo {extraction_result.filename}: {successful_entities}/9 entidades extraídas"
                )

        total_processing_time = int((time.time() - upload_start_time) * 1000)

        final_response = ExtractionResponse(
            success=any(result.success for result in extraction_results),
            documents=extraction_results,
            total_processing_time_ms=total_processing_time,
        )

        successful_extractions = sum(
            1 for result in extraction_results if result.success
        )
        total_documents = len(extraction_results)

        total_entities_found = 0
        total_entities_possible = len(extraction_results) * 9

        for result in extraction_results:
            if result.success:
                entities_found = sum(
                    1
                    for field_name in result.contract_entities.__dataclass_fields__
                    if getattr(result.contract_entities, field_name) is not None
                )
                total_entities_found += entities_found

        logging.info(
            f"Processamento completo: {successful_extractions}/{total_documents} documentos processados com sucesso"
        )
        logging.info(
            f"Entidades extraídas: {total_entities_found}/{total_entities_possible} em {total_processing_time}ms"
        )
        logging.info(
            f"Tabelas FAAS geradas: {len(tabelas_faas)} linhas detalhadas, {len(resumo_faas)} linhas de resumo"
        )

        response_data = final_response.to_dict()

        response_data["faas_tables"] = {
            "tabela_detalhada": tabelas_faas,
            "tabela_resumo": resumo_faas,
            "total_linhas_detalhadas": len(tabelas_faas),
            "total_contratos": len(resumo_faas),
        }

        try:
            from .write_json import save_json
            json_file_path = save_json(response_data)
            response_data["json_file_path"] = json_file_path
        except Exception:
            pass

        return func.HttpResponse(
            safe_json_dumps(response_data), mimetype="application/json", status_code=200
        )

    except ValueError as e:
        logging.error(f"Erro de validação: {str(e)}")
        return _create_error_response(f"Erro de validação: {str(e)}", 400)
    except Exception as e:
        logging.error(f"Erro ao processar arquivo: {str(e)}")
        return _create_error_response(f"Erro no processamento: {str(e)}", 500)


def _create_error_response(message: str, status_code: int) -> func.HttpResponse:
    error_response = ErrorResponse(error=message)

    return func.HttpResponse(
        safe_json_dumps(error_response.to_dict()),
        mimetype="application/json",
        status_code=status_code,
    )


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
                    logger.info(
                        f"Entidade '{field_name}' encontrada: {value.strip()[:50]}..."
                    )

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


class UploadFileService:
    """Serviço responsável pela extração de texto de arquivos PDF"""

    def __init__(self, req: func.HttpRequest):
        self.supported_extension = ".pdf"
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
                raise ValueError(
                    f"Arquivo {filename}: Apenas arquivos PDF são suportados"
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
                    file_type="pdf",
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
        documents = self.extract_text_from_pdf(file_content, filename)
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
            file_type="pdf",
            documents_count=len(chunk_data),
            total_characters=sum(len(chunk) for chunk in all_chunks),
            total_tokens=total_tokens,
            processing_time_ms=0,
            token_summary=token_summary,
            documents=chunk_data,
        )

        return response

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

            logger.info(f"Processando PDF: {filename} com {len(doc)} páginas")

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
            logger.info(
                f"Extração concluída: {len(documents)} páginas com texto de {filename}"
            )

        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF {filename}: {str(e)}")
            raise Exception(f"Falha na extração de texto: {str(e)}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo temporário: {str(e)}")

        if not documents:
            raise Exception("Nenhum texto foi extraído do PDF")

        return documents

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def is_supported_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.supported_extension)

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


logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.usage = TokenUsage()

        self.PRICES = {
            "gpt4o_input": 0.005,  # $5.00/1M tokens = $0.005/1K tokens
            "gpt4o_input_cached": 0.0025,  # $2.50/1M tokens = $0.0025/1K tokens (futuro)
            "gpt4o_output": 0.020,  # $20.00/1M tokens = $0.020/1K tokens
            "embedding_3_large": 0.00013,  # $0.13/1M tokens = $0.00013/1K tokens
        }

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def log_embedding_usage(self, texts: List[str]) -> int:
        total_tokens = sum(self.count_tokens(text) for text in texts)
        self.usage.embedding_tokens += total_tokens

        logger.info(f"Embedding - {len(texts)} documents, {total_tokens} tokens")
        return total_tokens

    def log_llm_usage(self, input_text: str, output_text: str) -> tuple[int, int]:
        """Registra uso de tokens LLM e retorna (input_tokens, output_tokens)"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)

        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.api_calls += 1

        return input_tokens, output_tokens

    def get_summary(self) -> TokenSummary:
        cost_breakdown = CostBreakdown(
            llm_input_cost=(self.usage.input_tokens / 1000)
            * self.PRICES["gpt4o_input"],
            llm_output_cost=(self.usage.output_tokens / 1000)
            * self.PRICES["gpt4o_output"],
            embedding_cost=(self.usage.embedding_tokens / 1000)
            * self.PRICES["embedding_3_large"],
        )

        summary = TokenSummary(usage=self.usage, cost_breakdown=cost_breakdown)

        logger.info(f"RESUMO DE TOKENS: {summary.to_dict()}")
        return summary

    def reset(self):
        self.usage = TokenUsage()

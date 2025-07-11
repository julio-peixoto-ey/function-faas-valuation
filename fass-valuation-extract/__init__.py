import os
import json
import logging
import asyncio
import time

import numpy as np
import concurrent.futures
import azure.functions as func
from .model import (
    DocumentExtractionResult,
    ContractEntity,
)
from .utils import TokenCounter
from .service import DocumentEntityExtractor, DocumentTextExtractorService
from .model import (
    ExtractionResponse,
    ErrorResponse,
)





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
        error_message=error_message or f"Falha: {file_response.message}",
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
                logging.error(f"Erro {files[i].filename}: {result}")
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
        logging.error(f"Erro na função: {e}")
        return _create_error_response(f"Erro interno: {e}", 500)


def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "API para extração de entidades de documentos",
        "status": "healthy",
        "timestamp": time.time(),
        "supported_formats": ["PDF", "TXT", "DOC", "DOCX"],
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
        info["warm_up"] = f"failed: {e}"

    return func.HttpResponse(
        safe_json_dumps(info), mimetype="application/json", status_code=200
    )


def _handle_file_upload_and_extraction(req: func.HttpRequest) -> func.HttpResponse:
    upload_start_time = time.time()

    upload_file_service = DocumentTextExtractorService(req)

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

        tabelas_por_series = {}  # Estrutura: {"serie_1": [...], "serie_2": [...]}
        resumo_faas = []

        for extraction_result in extraction_results:
            if extraction_result.success:
                # Criar tabelas para cada série (já retorna lista de listas)
                tabelas_series = extractor.create_tabelas_faas_por_series(
                    extraction_result.contract_entities, extraction_result.filename
                )
                
                # Organizar por série
                for idx, tabela_serie in enumerate(tabelas_series):
                    serie_key = f"serie_{idx + 1}"
                    if serie_key not in tabelas_por_series:
                        tabelas_por_series[serie_key] = []
                    tabelas_por_series[serie_key].extend(tabela_serie)
                
                # Criar rows de resumo para cada série
                rows_resumo_series = extractor.create_rows_faas_resumo_por_series(
                    extraction_result.contract_entities, extraction_result.filename
                )
                resumo_faas.extend(rows_resumo_series)
                
                successful_entities = sum(
                    1
                    for field_name in extraction_result.contract_entities.__dataclass_fields__
                    if getattr(extraction_result.contract_entities, field_name)
                    is not None
                )
                
                # Contar o número de séries processadas
                num_series = extractor._get_series_count(extraction_result.contract_entities)
                
                logging.info(
                    f"Arquivo {extraction_result.filename}: {successful_entities}/9 entidades extraídas, {num_series} série(s) processada(s)"
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
            f"Tabelas FAAS geradas: {len(tabelas_por_series)} series, {len(resumo_faas)} linhas de resumo"
        )

        response_data = final_response.to_dict()

        response_data["faas_tables"] = {
            "tabelas_detalhadas": tabelas_por_series,
            "tabela_resumo": resumo_faas,
            "total_series": len(tabelas_por_series),
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


import re
import base64
import os, requests, urllib.parse
import json
import logging
import time
import asyncio
import concurrent.futures
import numpy as np
import azure.functions as func
from .services.upload_file_service import UploadFileService
from .services.extract import DocumentEntityExtractor
from .models.response_models import ErrorResponse
from .models.entites_models import ExtractionResponse, DocumentExtractionResult, ContractEntity


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
    return DocumentExtractionResult(
        filename=file_response.filename,
        success=False,
        contract_entities=ContractEntity(),
        processing_time_ms=0,
        token_summary={"api_calls": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0, "estimated_cost_usd": 0.0, "breakdown": {"llm_input_cost": 0.0, "llm_output_cost": 0.0, "embedding_cost": 0.0}},
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
            "embeddings": "ok" if os.getenv("AZURE_EMBEDDING_DEPLOYMENT") else "missing"
        }
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

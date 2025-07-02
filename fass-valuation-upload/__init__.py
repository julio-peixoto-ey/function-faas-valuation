import azure.functions as func
import json
import logging
import time
import numpy as np
import os, requests, urllib.parse
from .services.upload_file_service import UploadFileService
from .services.extract import DocumentEntityExtractor
from .models.response_models import ErrorResponse
from .models.entites_models import ExtractionResponse
import re

########## Daverse ##########

# constantes corrigidas
FILE_SET   = "new_arquivoses"
PROJ_SET   = "new_valuation_projetoses"
ID_ATTR    = "new_valuation_projetosid"
NAME_ATTR  = "new_name"
LOOKUP_FK  = "_new_projeto_value"

# atributos realmente existentes em Arquivos
FILE_ATTR      = "new_file"        # conteúdo binário
FILE_NAME_ATTR = "new_file_name"   # nome legível
GUID_RE    = re.compile(r"^[0-9a-fA-F\\-]{36}$")

def get_token():
    url = f"https://login.microsoftonline.com/{os.getenv('DATAVERSE_TENANT_ID')}/oauth2/v2.0/token"
    body = {
        "client_id":     os.getenv("CLIENT_ID"),
        "client_secret": os.getenv("CLIENT_SECRET"),
        "grant_type":    "client_credentials",
        "scope":         os.getenv("SCOPE")
    }
    r = requests.post(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    r.raise_for_status()
    print(f"Token -- {r.json()}")
    return r.json()["access_token"]

def _project_guid_from_name(token: str, name: str) -> str | None:
    sanitized = name.replace("'", "''")
    params = {
        "$select": ID_ATTR,
        "$filter": f"{NAME_ATTR} eq '{sanitized}'"
    }
    r = requests.get(
        BASE_URL + PROJ_SET,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
    )
    print(f"Request URL: {BASE_URL + PROJ_SET}, Params: {params}")
    r.raise_for_status()
    rows = r.json().get("value", [])
    return rows[0][ID_ATTR] if rows else None

def list_files_by_project(key: str) -> list[dict]:
    token = get_token()
    guid  = key if GUID_RE.match(key) else _project_guid_from_name(token, key)
    if not guid:
        raise ValueError(f"Projeto '{key}' não encontrado")

    params = {
        "$select": "new_name,new_arquivo",
        "$filter": f"{LOOKUP_FK} eq {guid}",
        "$top": 100
    }
    r = requests.get(
        BASE_URL + FILE_SET,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
    )
    r.raise_for_status()
    return r.json()["value"]

def download_file(row_id: str, target_path: str):
    token = get_token()
    base  = os.getenv("BASE_URL")
    blob  = f"{base}valuation_arquivoses({row_id})/valuation_arquivo?$value"
    r = requests.get(blob, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(r.content)

########## Daverse ##########

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
    kwargs.setdefault('cls', CustomJSONEncoder)
    kwargs.setdefault('ensure_ascii', False)
    return json.dumps(obj, **kwargs)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function processando requisição")
    
    try:
        if req.method == "GET":
            project_guid = req.params.get('project_guid')
            if project_guid:
                return _handle_list_files_request(req)
            else:
                return _handle_info_request()
        elif req.method == "POST":
            return _handle_file_upload_and_extraction(req)
        else:
            return _create_error_response("Método não suportado", 405)
            
    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return _create_error_response(f"Erro interno: {str(e)}", 500)

def _handle_list_files_request(req: func.HttpRequest) -> func.HttpResponse:
    """Lida com a requisição para listar arquivos de um projeto no Dataverse."""
    project_guid = req.params.get('project_guid')
    logging.info(f"Buscando arquivos para o projeto GUID: {project_guid}")
    
    try:
        files = list_files_by_project(project_guid)
        return func.HttpResponse(
            safe_json_dumps(files),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Erro ao listar arquivos do Dataverse: {str(e)}")
        return _create_error_response(f"Erro ao comunicar com o Dataverse: {str(e)}", 500)

def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "File Upload e Extração de Entidades API para Contratos",
        "supported_formats": [".pdf"],
        "endpoints": {
            "upload_and_extract": "POST /valuation/contrato/upload",
            "info": "GET /valuation/contrato/upload"
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
            "fluxos_percentuais"
        ]
    }
    
    return func.HttpResponse(
        safe_json_dumps(info),
        mimetype="application/json",
        status_code=200
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
                status_code=400
            )
        
        logging.info(f"Upload concluído com {len(upload_response.files)} arquivo(s)")
        
        extractor = DocumentEntityExtractor()
        extraction_results = []
        
        tabelas_faas = []
        resumo_faas = []
        
        for file_response in upload_response.files:
            if file_response.success:
                logging.info(f"Extraindo entidades do arquivo: {file_response.filename}")
                
                document_chunks = file_response.documents
                
                extraction_result = extractor.extract_all_entities(
                    document_chunks, 
                    file_response.filename
                )
                extraction_results.append(extraction_result)
                if extraction_result.success:
                    tabela_contrato = extractor.create_tabela_faas(
                        extraction_result.contract_entities, 
                        file_response.filename
                    )
                    tabelas_faas.extend(tabela_contrato)
                    linha_resumo = extractor.create_row_faas_resumo(
                        extraction_result.contract_entities, 
                        file_response.filename
                    )
                    if linha_resumo:
                        resumo_faas.append(linha_resumo)
                successful_entities = sum(
                    1 for field_name in extraction_result.contract_entities.__dataclass_fields__
                    if getattr(extraction_result.contract_entities, field_name) is not None
                )
                logging.info(f"Arquivo {file_response.filename}: {successful_entities}/9 entidades extraídas")
                
            else:
                from .models.entites_models import DocumentExtractionResult, ContractEntity
                error_result = DocumentExtractionResult(
                    filename=file_response.filename,
                    success=False,
                    contract_entities=ContractEntity(),
                    processing_time_ms=0,
                    token_summary=extractor.token_counter.get_summary(),
                    error_message=f"Falha no upload: {file_response.message}"
                )
                extraction_results.append(error_result)
        
        total_processing_time = int((time.time() - upload_start_time) * 1000)
        
        final_response = ExtractionResponse(
            success=any(result.success for result in extraction_results),
            documents=extraction_results,
            total_processing_time_ms=total_processing_time
        )
        
        successful_extractions = sum(1 for result in extraction_results if result.success)
        total_documents = len(extraction_results)
        
        total_entities_found = 0
        total_entities_possible = len(extraction_results) * 9
        
        for result in extraction_results:
            if result.success:
                entities_found = sum(
                    1 for field_name in result.contract_entities.__dataclass_fields__
                    if getattr(result.contract_entities, field_name) is not None
                )
                total_entities_found += entities_found
        
        logging.info(f"Processamento completo: {successful_extractions}/{total_documents} documentos processados com sucesso")
        logging.info(f"Entidades extraídas: {total_entities_found}/{total_entities_possible} em {total_processing_time}ms")
        logging.info(f"Tabelas FAAS geradas: {len(tabelas_faas)} linhas detalhadas, {len(resumo_faas)} linhas de resumo")
        
        response_data = final_response.to_dict()
        
        response_data['faas_tables'] = {
            'tabela_detalhada': tabelas_faas,
            'tabela_resumo': resumo_faas,
            'total_linhas_detalhadas': len(tabelas_faas),
            'total_contratos': len(resumo_faas)
        }
        
        return func.HttpResponse(
            safe_json_dumps(response_data),
            mimetype="application/json",
            status_code=200
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
import re
import base64
import os, requests
import json
import logging
import time
import numpy as np
import azure.functions as func
from typing import Optional

########## Daverse ##########

BASE_URL = os.getenv("BASE_URL")
FILE_SET = "new_arquivoses"
PROJ_SET = "new_valuation_projetoses"

ID_ATTR = "new_valuation_projetosid"
NAME_ATTR = "new_name"
LOOKUP_FK = "_new_projeto_value"

FILE_ID_ATTR = "new_file"
FILE_NAME_ATTR = "new_file_name"
ROW_ID_ATTR = "new_arquivosid"

GUID_RE = re.compile(r"^[0-9a-fA-F\-]{36}$")
FUNCTION_BASE_URL = os.getenv("FUNCTION_BASE_URL")

_cached_token: Optional[str] = None
_token_expires_at: float = 0

def get_cached_token() -> str:
    global _cached_token, _token_expires_at
    
    if _cached_token and time.time() < _token_expires_at:
        logging.info("Usando token Dataverse cacheado")
        return _cached_token
    
    logging.info("Renovando token Dataverse")
    url = f"https://login.microsoftonline.com/{os.getenv('DATAVERSE_TENANT_ID')}/oauth2/v2.0/token"
    body = {
        "client_id": os.getenv("CLIENT_ID"),
        "client_secret": os.getenv("CLIENT_SECRET"),
        "grant_type": "client_credentials",
        "scope": os.getenv("SCOPE"),
    }
    
    r = requests.post(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    r.raise_for_status()
    
    response = r.json()
    _cached_token = response["access_token"]
    _token_expires_at = time.time() + (50 * 60)
    
    return _cached_token

def _project_guid_from_name(token: str, name: str) -> str | None:
    name = name.replace("'", "''")
    params = {"$select": ID_ATTR, "$filter": f"{NAME_ATTR} eq '{name}'"}
    r = requests.get(
        BASE_URL + PROJ_SET,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
    )
    r.raise_for_status()
    rows = r.json()["value"]
    return rows[0][ID_ATTR] if rows else None

def list_files_by_project(key: str) -> tuple[list[dict], str]:
    token = get_cached_token()
    guid = key if GUID_RE.match(key) else _project_guid_from_name(token, key)
    if not guid:
        raise ValueError(f"Projeto '{key}' não encontrado")

    params = {"$filter": f"{LOOKUP_FK} eq {guid}", "$top": 5000}
    r = requests.get(
        BASE_URL + FILE_SET,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
    )
    r.raise_for_status()
    return r.json()["value"], token

def get_binary_list(rows: list[dict], token: str) -> list[dict]:
    binaries, auth = [], {"Authorization": f"Bearer {token}"}

    for row in rows:
        file_guid = row.get(FILE_ID_ATTR)
        if not file_guid:
            continue

        url = f"{BASE_URL}{FILE_SET}({row[ROW_ID_ATTR]})/{FILE_ID_ATTR}/$value"
        resp = requests.get(url, headers=auth)
        resp.raise_for_status()

        binaries.append(
            {
                "file_name": row.get(FILE_NAME_ATTR) or "sem_nome.bin",
                "binary": base64.b64encode(resp.content).decode(),
            }
        )
    return binaries

def _call_extraction_service(files_b64: list) -> dict:
    try:
        extraction_url = f"{FUNCTION_BASE_URL}/valuation/extract"
        
        payload = {
            "files": [
                {
                    "filename": file_data["file_name"],
                    "file_content": file_data["binary"]
                }
                for file_data in files_b64
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        timeout_seconds = min(300, 60 + (len(files_b64) * 30))
        
        response = requests.post(extraction_url, json=payload, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.Timeout:
        logging.error(f"Timeout após {timeout_seconds}s processando {len(files_b64)} arquivos")
        raise
    except Exception as e:
        logging.error(f"Erro ao chamar serviço de extração: {e}")
        raise

def _handle_list_files_request(req: func.HttpRequest) -> func.HttpResponse:
    project_key = req.params.get("project_name")
    logging.info(f"Buscando arquivos para: {project_key}")

    try:
        rows, token = list_files_by_project(project_key)
        files_b64 = get_binary_list(rows, token)

        logging.info(f"Encontrados {len(files_b64)} arquivos, iniciando extração")
        
        extraction_result = _call_extraction_service(files_b64)
        
        return func.HttpResponse(
            safe_json_dumps(extraction_result),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as exc:
        logging.error(f"Falha no processamento: {exc}")
        return _create_error_response("Erro ao processar arquivos", 500)

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
    kwargs.setdefault("cls", CustomJSONEncoder)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function processando requisição")

    try:
        if req.method == "GET":
            project_name = req.params.get("project_name")
            if not project_name:
                return _create_error_response("Project name não fornecido", 400)
            return _handle_list_files_request(req)
        elif req.method == "POST":
            return _handle_info_request()
        else:
            return _create_error_response("Método não suportado", 405)

    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return _create_error_response(f"Erro interno: {str(e)}", 500)

def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "API para obter arquivos binários de um projeto",
        "status": "healthy",
        "timestamp": time.time(),
        "endpoints": {
            "get_binaries": "GET /valuation/get-binaries?project_name=X",
            "health": "POST /valuation/get-binaries",
        },
        "required_params": ["project_name"],
        "connections": {
            "dataverse": "ok" if BASE_URL else "missing",
            "extraction_service": "ok" if FUNCTION_BASE_URL else "missing"
        }
    }

    try:
        token = get_cached_token()
        info["warm_up"] = "completed"
        info["token_status"] = "cached" if _cached_token else "fresh"
    except Exception as e:
        info["warm_up"] = f"failed: {str(e)}"

    return func.HttpResponse(
        safe_json_dumps(info), mimetype="application/json", status_code=200
    )

def _create_error_response(message: str, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        safe_json_dumps({"error": message}),
        mimetype="application/json",
        status_code=status_code,
    )

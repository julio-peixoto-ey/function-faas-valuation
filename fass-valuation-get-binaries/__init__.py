import re
import base64
import os, requests
import json
import logging
import numpy as np
import azure.functions as func

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


def get_token():
    url = f"https://login.microsoftonline.com/{os.getenv('DATAVERSE_TENANT_ID')}/oauth2/v2.0/token"
    body = {
        "client_id": os.getenv("CLIENT_ID"),
        "client_secret": os.getenv("CLIENT_SECRET"),
        "grant_type": "client_credentials",
        "scope": os.getenv("SCOPE"),
    }
    r = requests.post(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    r.raise_for_status()
    return r.json()["access_token"]


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
    """Devolve (lista de rows, token) para consumir depois."""
    token = get_token()
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


def _handle_list_files_request(req: func.HttpRequest) -> func.HttpResponse:
    project_key = req.params.get("project_guid")
    logging.info(f"Buscando arquivos para: {project_key}")

    try:
        rows, token = list_files_by_project(project_key)
        files_b64 = get_binary_list(rows, token)

        return func.HttpResponse(
            safe_json_dumps({"files": files_b64, "total": len(files_b64)}),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as exc:
        logging.error(f"Falha Dataverse: {exc}")
        return _create_error_response("Erro ao comunicar com o Dataverse", 500)


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
            return _handle_info_request()
        elif req.method == "POST":
            project_guid = req.params.get("project_guid")
            if not project_guid:
                return _create_error_response("Project GUID não fornecido", 400)
            return _handle_list_files_request(req)
        else:
            return _create_error_response("Método não suportado", 405)

    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return _create_error_response(f"Erro interno: {str(e)}", 500)


def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "API para obter arquivos binários de um projeto",
        "endpoints": {
            "get_binaries": "POST /valuation/get-binaries",
            "info": "GET /valuation/get-binaries",
        },
        "required_params": ["project_guid"],
    }

    return func.HttpResponse(
        safe_json_dumps(info), mimetype="application/json", status_code=200
    )


def _create_error_response(message: str, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        safe_json_dumps({"error": message}),
        mimetype="application/json",
        status_code=status_code,
    )

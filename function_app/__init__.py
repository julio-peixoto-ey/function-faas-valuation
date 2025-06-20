import azure.functions as func
import json
import logging
import base64
from .services.upload_file_service import UploadFileService
from .utils.token_counter import TokenCounter
from .models.response_models import FileUploadResponse, ErrorResponse

def main(req: func.HttpRequest) -> func.HttpResponse:
    upload_file_service = UploadFileService(req)
    logging.info("Azure Function processando requisição")
    
    try:
        if req.method == "GET":
            return _handle_info_request()
        elif req.method == "POST":
            return _handle_file_upload(req)
        else:
            return upload_file_service._create_error_response("Método não suportado", 405)
            
    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return upload_file_service._create_error_response(f"Erro interno: {str(e)}", 500)

def _handle_info_request() -> func.HttpResponse:
    info = {
        "message": "File Upload API para Power Apps",
        "supported_formats": [".pdf"],
        "endpoints": {
            "upload": "POST /valuation/contrato/upload",
            "info": "GET /valuation/contrato/upload"
        }
    }
    
    return func.HttpResponse(
        json.dumps(info, ensure_ascii=False),
        mimetype="application/json",
        status_code=200
    )

def _handle_file_upload(req: func.HttpRequest) -> func.HttpResponse:
    upload_file_service = UploadFileService(req)
    
    try:
        response = upload_file_service.process_file_upload()
        
        logging.info(f"Processamento concluído: {response.documents_count} páginas, {response.total_tokens} tokens")
        
        return func.HttpResponse(
            json.dumps(response.to_dict(), ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )
        
    except ValueError as e:
        logging.error(f"Erro de validação: {str(e)}")
        return upload_file_service._create_error_response(f"Erro de validação: {str(e)}", 400)
    except Exception as e:
        logging.error(f"Erro ao processar arquivo: {str(e)}")
        return upload_file_service._create_error_response(f"Erro no processamento: {str(e)}", 500)

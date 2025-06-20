import azure.functions as func
import json
import logging
from .services.token_counter import TokenCounter
from .services.file_processor import FileProcessor

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure Function processando requisição")
    
    try:
        if req.method == "GET":
            return func.HttpResponse(
                json.dumps({
                    "message": "File Upload API para Power Apps",
                    "supported_formats": [".pdf", ".docx", ".doc", ".txt"],
                    "endpoints": {
                        "upload": "POST /valuation/contrato/upload",
                        "info": "GET /valuation/contrato/upload"
                    }
                }),
                mimetype="application/json",
                status_code=200
            )
        
        elif req.method == "POST":
            return handle_file_upload_power_apps(req)
        
    except Exception as e:
        logging.error(f"Erro na função: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

def handle_file_upload_power_apps(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info("Processando upload de arquivo do Power Apps")
    
    file_processor = FileProcessor()
    token_counter = TokenCounter()
    
    try:
        files = req.files
        form_data = req.form
        
        logging.info(f"Files recebidos: {list(files.keys()) if files else 'Nenhum'}")
        logging.info(f"Form data: {dict(form_data) if form_data else 'Nenhum'}")
        
        if not files:
            try:
                body_json = req.get_json()
                if body_json and 'file_content' in body_json:
                    import base64
                    file_content = base64.b64decode(body_json['file_content'])
                    filename = body_json.get('filename', 'document.pdf')
                else:
                    return func.HttpResponse(
                        json.dumps({"error": "Nenhum arquivo enviado"}),
                        mimetype="application/json",
                        status_code=400
                    )
            except:
                return func.HttpResponse(
                    json.dumps({"error": "Formato de arquivo não reconhecido"}),
                    mimetype="application/json",
                    status_code=400
                )
        else:
            file_item = None
            for key in files:
                file_item = files[key]
                break
            
            if not file_item:
                return func.HttpResponse(
                    json.dumps({"error": "Arquivo inválido"}),
                    mimetype="application/json",
                    status_code=400
                )
            
            filename = file_item.filename
            file_content = file_item.read()
        
        logging.info(f"Processando arquivo: {filename}, tamanho: {len(file_content)} bytes")
        
        documents = file_processor.extract_text_from_file(file_content, filename)
        
        if not documents:
            return func.HttpResponse(
                json.dumps({"error": "Não foi possível extrair texto do arquivo"}),
                mimetype="application/json",
                status_code=400
            )
        
        all_texts = [doc.page_content for doc in documents]
        total_tokens = token_counter.log_embedding_usage(all_texts)
        
        result = {
            "success": True,
            "message": "Arquivo processado com sucesso",
            "filename": filename,
            "file_type": documents[0].metadata.get("file_type"),
            "documents_count": len(documents),
            "total_characters": sum(len(doc.page_content) for doc in documents),
            "total_tokens": total_tokens,
            "cost_summary": {
                "total_cost_usd": token_counter.get_summary()["estimated_cost_usd"],
                "total_tokens": token_counter.get_summary()["total_tokens"]
            },
            "documents_preview": [
                {
                    "page": doc.metadata.get("page", i+1),
                    "characters": len(doc.page_content),
                    "tokens": token_counter.count_tokens(doc.page_content),
                    "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                for i, doc in enumerate(documents[:3])
            ]
        }
        
        logging.info(f"Processamento concluído: {result['documents_count']} documentos, {result['total_tokens']} tokens")
        
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )
        
    except ValueError as e:
        logging.error(f"Erro de formato: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Formato de arquivo não suportado: {str(e)}"}),
            mimetype="application/json",
            status_code=400
        )
    except Exception as e:
        logging.error(f"Erro ao processar arquivo: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Erro interno: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )
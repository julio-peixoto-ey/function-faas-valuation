import fitz
import tempfile
import os
from typing import List
from ..models.file_models import DocumentModel
import logging
from ..utils.token_counter import TokenCounter
import azure.functions as func
from ..models.response_models import FileUploadResponse, BulkFileUploadResponse
import base64
from ..models.response_models import ErrorResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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

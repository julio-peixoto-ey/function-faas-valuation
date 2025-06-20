import fitz
import tempfile
import os
from typing import List
from ..models.file_models import DocumentModel
import logging
from ..utils.token_counter import TokenCounter
import azure.functions as func
from ..models.response_models import FileUploadResponse
import base64
from ..models.response_models import ErrorResponse
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tiktoken import encoding_for_model

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

    def process_file_upload(self) -> FileUploadResponse:
        file_content, filename = self._extract_file_from_request()

        if not self.is_supported_file(filename):
            raise ValueError("Apenas arquivos PDF são suportados")

        logging.info(
            f"Processando arquivo: {filename}, tamanho: {len(file_content)} bytes"
        )

        documents = self.extract_text_from_pdf(file_content, filename)
        all_texts = [doc.page_content for doc in documents]
        all_chunks = self.split_pages(all_texts)
        total_tokens = self.token_counter.log_embedding_usage(all_chunks)
        token_summary = self.token_counter.get_summary()

        chunk_data = []
        for idx, chunk in enumerate(all_chunks):
            chunk_data.append({
                "chunk_id": idx + 1,
                "content": base64.b64encode(chunk.encode()).decode(),
                "tokens": len(chunk.split()),
                "page": (idx // len(documents)) + 1,
            })

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

    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """
        Extrai texto de um arquivo PDF

        Args:
            file_content: Conteúdo do arquivo PDF em bytes
            filename: Nome do arquivo PDF

        Returns:
            Lista de DocumentModel com o texto extraído por página
        """
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
        """Estimativa simples de tokens baseada na contagem de caracteres"""
        return len(text) // 4

    def is_supported_file(self, filename: str) -> bool:
        """Verifica se o arquivo é um PDF suportado"""
        return filename.lower().endswith(self.supported_extension)

    def _extract_file_from_request(self) -> tuple[bytes, str]:
        files = self.req.files
        if files:
            file_item = next(iter(files.values()))
            return file_item.read(), file_item.filename

        try:
            body_json = self.req.get_json()
            if body_json and "file_content" in body_json:
                file_content = base64.b64decode(body_json["file_content"])
                filename = body_json.get("filename", "document.pdf")
                return file_content, filename
        except Exception:
            pass

        raise ValueError("Nenhum arquivo foi enviado ou formato não reconhecido")

    def split_pages(self, pages: list[str]) -> list[str]:
        """
        Divide as páginas do documento em chunks com sobreposição.

        Args:
            pages: Lista de páginas do documento em formato de texto.

        Returns:
            Lista de chunks do documento.
        """
        joined = "\n".join(pages)
        return self.splitter.split_text(joined)
    
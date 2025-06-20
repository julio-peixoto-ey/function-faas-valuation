import fitz  # PyMuPDF
import tempfile
import os
import time
from docx import Document as DocxDocument
from typing import List
from ..models.file_models import FileUploadRequest, DocumentModel, FileProcessingResult
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    def process_file_upload(self, file_request: FileUploadRequest) -> FileProcessingResult:
        start_time = time.time()
        
        try:
            if f".{file_request.file_extension}" not in self.supported_extensions:
                return FileProcessingResult(
                    success=False,
                    filename=file_request.filename,
                    file_type=file_request.file_extension,
                    documents=[],
                    total_characters=0,
                    total_tokens=0,
                    processing_time_ms=0,
                    error_message=f"Tipo de arquivo não suportado: {file_request.file_extension}"
                )
            
            documents = self._extract_documents(file_request)
            
            total_characters = sum(doc.character_count for doc in documents)
            total_tokens = sum(doc.token_count for doc in documents)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return FileProcessingResult(
                success=True,
                filename=file_request.filename,
                file_type=file_request.file_extension,
                documents=documents,
                total_characters=total_characters,
                total_tokens=total_tokens,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Erro ao processar arquivo: {str(e)}")
            return FileProcessingResult(
                success=False,
                filename=file_request.filename,
                file_type=file_request.file_extension,
                documents=[],
                total_characters=0,
                total_tokens=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )
    
    def _extract_documents(self, file_request: FileUploadRequest) -> List[DocumentModel]:
        if file_request.file_type == file_request.file_type.PDF:
            return self._extract_pdf(file_request)
        elif file_request.file_type in [file_request.file_type.DOCX, file_request.file_type.DOC]:
            return self._extract_docx(file_request)
        elif file_request.file_type == file_request.file_type.TXT:
            return self._extract_txt(file_request)
        else:
            raise ValueError(f"Tipo não suportado: {file_request.file_type}")
    
    def _extract_pdf(self, file_request: FileUploadRequest) -> List[DocumentModel]:
        documents = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_request.file_content)
            temp_path = temp_file.name
        
        try:
            doc = fitz.open(temp_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    document = DocumentModel(
                        page_content=text,
                        page_number=page_num + 1,
                        source=file_request.filename,
                        file_type="pdf",
                        character_count=len(text),
                        token_count=self._estimate_tokens(text)
                    )
                    documents.append(document)
            
            doc.close()
            
        finally:
            os.unlink(temp_path)
        
        return documents
    
    def _extract_docx(self, file_request: FileUploadRequest) -> List[DocumentModel]:
        documents = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(file_request.file_content)
            temp_path = temp_file.name
        
        try:
            doc = DocxDocument(temp_path)
            
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            if full_text:
                text = '\n'.join(full_text)
                document = DocumentModel(
                    page_content=text,
                    page_number=1,
                    source=file_request.filename,
                    file_type="docx",
                    character_count=len(text),
                    token_count=self._estimate_tokens(text)
                )
                documents.append(document)
                
        finally:
            os.unlink(temp_path)
        
        return documents
    
    def _extract_txt(self, file_request: FileUploadRequest) -> List[DocumentModel]:
        try:
            text = file_request.file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = file_request.file_content.decode('latin-1')
        
        if text.strip():
            document = DocumentModel(
                page_content=text,
                page_number=1,
                source=file_request.filename,
                file_type="txt",
                character_count=len(text),
                token_count=self._estimate_tokens(text)
            )
            return [document]
        
        return []
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

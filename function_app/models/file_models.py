from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"

@dataclass
class FileUploadRequest:
    filename: str
    file_content: bytes
    file_size: int
    content_type: Optional[str] = None
    
    @property
    def file_extension(self) -> str:
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ''
    
    @property
    def file_type(self) -> FileType:
        ext = self.file_extension
        if ext == 'pdf':
            return FileType.PDF
        elif ext in ['docx', 'doc']:
            return FileType.DOCX
        elif ext == 'txt':
            return FileType.TXT
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {ext}")

@dataclass
class DocumentModel:
    page_content: str
    page_number: int
    source: str
    file_type: str
    character_count: int
    token_count: int
    
    @property
    def preview(self) -> str:
        return self.page_content[:200] + "..." if len(self.page_content) > 200 else self.page_content

@dataclass
class FileProcessingResult:
    success: bool
    filename: str
    file_type: str
    documents: list[DocumentModel]
    total_characters: int
    total_tokens: int
    processing_time_ms: int
    error_message: Optional[str] = None
    
    @property
    def documents_count(self) -> int:
        return len(self.documents)
    
    @property
    def average_tokens_per_document(self) -> float:
        return self.total_tokens / len(self.documents) if self.documents else 0

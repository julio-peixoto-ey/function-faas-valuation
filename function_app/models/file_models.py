from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"

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

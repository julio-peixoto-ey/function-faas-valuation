from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from .file_models import DocumentModel
from .token_models import TokenSummary

@dataclass
class ApiResponse:
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

@dataclass
class FileUploadResponse:
    success: bool
    filename: str
    file_type: str
    documents_count: int
    total_characters: int
    total_tokens: int
    processing_time_ms: int
    token_summary: TokenSummary
    documents_preview: List[Dict[str, Any]]
    message: str = "Arquivo processado com sucesso"
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'message': self.message,
            'filename': self.filename,
            'file_type': self.file_type,
            'documents_count': self.documents_count,
            'total_characters': self.total_characters,
            'total_tokens': self.total_tokens,
            'processing_time_ms': self.processing_time_ms,
            'cost_summary': {
                'total_cost_usd': self.token_summary.total_cost_usd,
                'total_tokens': self.token_summary.total_tokens
            },
            'token_summary': self.token_summary.to_dict(),
            'documents_preview': self.documents_preview
        }

@dataclass
class ErrorResponse:
    success: bool = False
    error: str = ""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'success': self.success,
            'error': self.error
        }
        if self.error_code:
            result['error_code'] = self.error_code
        if self.details:
            result['details'] = self.details
        return result

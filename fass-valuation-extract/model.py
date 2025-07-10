from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

@dataclass
class ExtractedEntity:
    """Entidade extraída de um documento"""

    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


@dataclass
class ContractEntity:
    """Entidades específicas de contratos financeiros"""

    atualizacao_monetaria: Optional[ExtractedEntity] = None
    juros_remuneratorios: Optional[ExtractedEntity] = None
    spread_fixo: Optional[ExtractedEntity] = None
    base_calculo: Optional[ExtractedEntity] = None
    data_emissao: Optional[ExtractedEntity] = None
    data_vencimento: Optional[ExtractedEntity] = None
    valor_nominal_unitario: Optional[ExtractedEntity] = None
    fluxos_pagamento: Optional[ExtractedEntity] = None
    fluxos_percentuais: Optional[ExtractedEntity] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    api_calls: int = 0


@dataclass
class CostBreakdown:
    llm_input_cost: float
    llm_output_cost: float
    embedding_cost: float

    @property
    def total_cost(self) -> float:
        return self.llm_input_cost + self.llm_output_cost + self.embedding_cost


@dataclass
class TokenSummary:
    usage: TokenUsage
    cost_breakdown: CostBreakdown

    @property
    def total_tokens(self) -> int:
        return (
            self.usage.input_tokens
            + self.usage.output_tokens
            + self.usage.embedding_tokens
        )

    @property
    def total_cost_usd(self) -> float:
        return self.cost_breakdown.total_cost

    def to_dict(self) -> Dict:
        return {
            "api_calls": self.usage.api_calls,
            "total_tokens": self.total_tokens,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "embedding_tokens": self.usage.embedding_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 4),
            "breakdown": {
                "llm_input_cost": round(self.cost_breakdown.llm_input_cost, 4),
                "llm_output_cost": round(self.cost_breakdown.llm_output_cost, 4),
                "embedding_cost": round(self.cost_breakdown.embedding_cost, 4),
            },
        }


@dataclass
class DocumentExtractionResult:
    """Resultado da extração de um documento"""

    filename: str
    success: bool
    contract_entities: ContractEntity
    processing_time_ms: int
    token_summary: TokenSummary
    error_message: Optional[str] = None


@dataclass
class ExtractionResponse:
    """Resposta da API de extração"""

    success: bool
    documents: List[DocumentExtractionResult]
    total_processing_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "documents": [
                {
                    "filename": doc.filename,
                    "success": doc.success,
                    "contract_entities": self._contract_entities_to_dict(
                        doc.contract_entities
                    ),
                    "processing_time_ms": doc.processing_time_ms,
                    "token_summary": doc.token_summary.to_dict(),
                    "error_message": doc.error_message,
                }
                for doc in self.documents
            ],
            "total_processing_time_ms": self.total_processing_time_ms,
        }

    def _contract_entities_to_dict(self, entities: ContractEntity) -> Dict[str, Any]:
        result = {}
        for field_name in entities.__dataclass_fields__:
            entity = getattr(entities, field_name)
            if entity:
                result[field_name] = {
                    "value": entity.value,
                    "confidence": entity.confidence,
                    "context": entity.context,
                }
            else:
                result[field_name] = None
        return result


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
        return (
            self.page_content[:200] + "..."
            if len(self.page_content) > 200
            else self.page_content
        )


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
    documents: List[Dict[str, Any]]
    message: str = "Arquivo processado com sucesso"

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "message": self.message,
            "filename": self.filename,
            "file_type": self.file_type,
            "documents_count": self.documents_count,
            "total_characters": self.total_characters,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms,
            "cost_summary": {
                "total_cost_usd": self.token_summary.total_cost_usd,
                "total_tokens": self.token_summary.total_tokens,
            },
            "token_summary": self.token_summary.to_dict(),
            "documents": self.documents,
        }


@dataclass
class BulkFileUploadResponse:
    success: bool
    files: List[FileUploadResponse]

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "files": [file.to_dict() for file in self.files],
        }


@dataclass
class ErrorResponse:
    success: bool = False
    error: str = ""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        result = {"success": self.success, "error": self.error}
        if self.error_code:
            result["error_code"] = self.error_code
        if self.details:
            result["details"] = self.details
        return result

    @property
    def total_cost(self) -> float:
        return self.llm_input_cost + self.llm_output_cost + self.embedding_cost
    

class ContractEntitiesResponse(BaseModel):

    atualizacao_monetaria: str = Field(
        default="NÃO ENCONTRADO",
        description="Índice que corrige o principal (IPCA, IGP-M, SELIC, etc.)",
    )

    juros_remuneratorios: str = Field(
        default="NÃO ENCONTRADO",
        description="Indexador principal dos juros (DI+, CDI+, IPCA+, etc.)",
    )

    spread_fixo: str = Field(
        default="NÃO ENCONTRADO",
        description="Percentual adicional sobre o indexador principal",
    )

    base_calculo: str = Field(
        default="NÃO ENCONTRADO",
        description="Metodologia de cálculo de juros (252, 365, ACT/360)",
    )

    data_emissao: str = Field(
        default="NÃO ENCONTRADO",
        description="Data(s) de emissão do título no formato DD/MM/AAAA",
    )

    data_vencimento: str = Field(
        default="NÃO ENCONTRADO",
        description="Data(s) de vencimento no formato DD/MM/AAAA",
    )

    valor_nominal_unitario: str = Field(
        default="NÃO ENCONTRADO", description="Valor de face por título/cota"
    )

    fluxos_pagamento: str = Field(
        default="NÃO ENCONTRADO",
        description="Datas do cronograma de pagamentos separadas por vírgula",
    )

    fluxos_percentuais: str = Field(
        default="NÃO ENCONTRADO",
        description="Percentuais de amortização separados por vírgula",
    )
    
class SerieResponse(BaseModel):
    pass

class SeriesResponse(BaseModel):
    series: Dict[str, SerieResponse]
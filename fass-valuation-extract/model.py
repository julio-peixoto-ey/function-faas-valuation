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
class SeriesExtractedEntity:
    """Entidade extraída com suporte para múltiplas séries"""

    entity_type: str
    values: List[str]
    confidences: List[float]
    contexts: List[Optional[str]]

    def __post_init__(self):
        """Garantir que todas as listas tenham o mesmo tamanho"""
        max_len = max(len(self.values), len(self.confidences), len(self.contexts))
        
        # Preencher listas menores com valores padrão
        while len(self.values) < max_len:
            self.values.append("NÃO ENCONTRADO")
        while len(self.confidences) < max_len:
            self.confidences.append(0.0)
        while len(self.contexts) < max_len:
            self.contexts.append(None)

    @property
    def series_count(self) -> int:
        return len(self.values)

    def to_dict(self) -> Dict[str, Any]:
        if self.series_count == 0:
            return None
        
        return {
            "value": self.values,
            "confidence": self.confidences,
            "context": self.contexts,
        }


@dataclass
class ContractEntity:
    """Entidades específicas de contratos financeiros com suporte para múltiplas séries"""

    atualizacao_monetaria: Optional[SeriesExtractedEntity] = None
    juros_remuneratorios: Optional[SeriesExtractedEntity] = None
    spread_fixo: Optional[SeriesExtractedEntity] = None
    base_calculo: Optional[SeriesExtractedEntity] = None
    data_emissao: Optional[SeriesExtractedEntity] = None
    data_vencimento: Optional[SeriesExtractedEntity] = None
    valor_nominal_unitario: Optional[SeriesExtractedEntity] = None
    fluxos_pagamento: Optional[SeriesExtractedEntity] = None
    fluxos_percentuais: Optional[SeriesExtractedEntity] = None


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
                result[field_name] = entity.to_dict()
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
    """Resposta para múltiplas séries de contratos"""

    atualizacao_monetaria: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de índices que corrigem o principal por série (IPCA, IGP-M, SELIC, etc.)",
    )

    juros_remuneratorios: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de indexadores principais dos juros por série (DI+, CDI+, IPCA+, etc.)",
    )

    spread_fixo: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de percentuais adicionais sobre o indexador principal por série",
    )

    base_calculo: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de metodologias de cálculo de juros por série (252, 365, ACT/360)",
    )

    data_emissao: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de datas de emissão do título por série no formato DD/MM/AAAA",
    )

    data_vencimento: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de datas de vencimento por série no formato DD/MM/AAAA",
    )

    valor_nominal_unitario: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de valores de face por título/cota por série"
    )

    fluxos_pagamento: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de datas do cronograma de pagamentos por série separadas por vírgula",
    )

    fluxos_percentuais: List[str] = Field(
        default_factory=lambda: ["NÃO ENCONTRADO"],
        description="Lista de percentuais de amortização por série separados por vírgula",
    )
    
class SerieResponse(BaseModel):
    """Resposta para uma série específica"""
    
    serie_id: str
    atualizacao_monetaria: str = "NÃO ENCONTRADO"
    juros_remuneratorios: str = "NÃO ENCONTRADO"
    spread_fixo: str = "NÃO ENCONTRADO"
    base_calculo: str = "NÃO ENCONTRADO"
    data_emissao: str = "NÃO ENCONTRADO"
    data_vencimento: str = "NÃO ENCONTRADO"
    valor_nominal_unitario: str = "NÃO ENCONTRADO"
    fluxos_pagamento: str = "NÃO ENCONTRADO"
    fluxos_percentuais: str = "NÃO ENCONTRADO"

class SeriesResponse(BaseModel):
    """Resposta para múltiplas séries"""
    
    series: Dict[str, SerieResponse]
    total_series: int = 0
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_series = len(self.series)
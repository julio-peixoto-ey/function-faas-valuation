from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from .token_models import TokenSummary

@dataclass
class ExtractedEntity:
    """Entidade extraída de um documento"""
    entity_type: str
    value: str
    confidence: float
    page_references: List[int]
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
            'success': self.success,
            'documents': [
                {
                    'filename': doc.filename,
                    'success': doc.success,
                    'contract_entities': self._contract_entities_to_dict(doc.contract_entities),
                    'processing_time_ms': doc.processing_time_ms,
                    'token_summary': doc.token_summary.to_dict(),
                    'error_message': doc.error_message
                }
                for doc in self.documents
            ],
            'total_processing_time_ms': self.total_processing_time_ms
        }
    
    def _contract_entities_to_dict(self, entities: ContractEntity) -> Dict[str, Any]:
        result = {}
        for field_name in entities.__dataclass_fields__:
            entity = getattr(entities, field_name)
            if entity:
                result[field_name] = {
                    'value': entity.value,
                    'confidence': entity.confidence,
                    'page_references': entity.page_references,
                    'context': entity.context
                }
            else:
                result[field_name] = None
        return result

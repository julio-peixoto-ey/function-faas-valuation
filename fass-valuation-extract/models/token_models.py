from dataclasses import dataclass
from typing import Dict

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
        return self.usage.input_tokens + self.usage.output_tokens + self.usage.embedding_tokens
    
    @property
    def total_cost_usd(self) -> float:
        return self.cost_breakdown.total_cost
    
    def to_dict(self) -> Dict:
        return {
            'api_calls': self.usage.api_calls,
            'total_tokens': self.total_tokens,
            'input_tokens': self.usage.input_tokens,
            'output_tokens': self.usage.output_tokens,
            'embedding_tokens': self.usage.embedding_tokens,
            'estimated_cost_usd': round(self.total_cost_usd, 4),
            'breakdown': {
                'llm_input_cost': round(self.cost_breakdown.llm_input_cost, 4),
                'llm_output_cost': round(self.cost_breakdown.llm_output_cost, 4),
                'embedding_cost': round(self.cost_breakdown.embedding_cost, 4)
            }
        }

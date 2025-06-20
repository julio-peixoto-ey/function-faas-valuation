import tiktoken
import logging
from typing import List
from ..models.token_models import TokenUsage, CostBreakdown, TokenSummary

logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.usage = TokenUsage()
        
        self.PRICES = {
            'gpt4_input': 0.03,
            'gpt4_output': 0.06,
            'embedding': 0.0001
        }
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def log_embedding_usage(self, texts: List[str]) -> int:
        total_tokens = sum(self.count_tokens(text) for text in texts)
        self.usage.embedding_tokens += total_tokens
        
        logger.info(f"Embedding - {len(texts)} documents, {total_tokens} tokens")
        return total_tokens
    
    def get_summary(self) -> TokenSummary:
        cost_breakdown = CostBreakdown(
            llm_input_cost=(self.usage.input_tokens / 1000) * self.PRICES['gpt4_input'],
            llm_output_cost=(self.usage.output_tokens / 1000) * self.PRICES['gpt4_output'],
            embedding_cost=(self.usage.embedding_tokens / 1000) * self.PRICES['embedding']
        )
        
        summary = TokenSummary(
            usage=self.usage,
            cost_breakdown=cost_breakdown
        )
        
        logger.info(f"RESUMO DE TOKENS: {summary.to_dict()}")
        return summary
    
    def reset(self):
        self.usage = TokenUsage()
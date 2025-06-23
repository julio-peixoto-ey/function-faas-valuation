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
            'gpt4o_input': 0.005,        # $5.00/1M tokens = $0.005/1K tokens
            'gpt4o_input_cached': 0.0025, # $2.50/1M tokens = $0.0025/1K tokens (futuro)
            'gpt4o_output': 0.020,       # $20.00/1M tokens = $0.020/1K tokens
            
            'embedding_3_large': 0.00013, # $0.13/1M tokens = $0.00013/1K tokens
            
        }
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def log_embedding_usage(self, texts: List[str]) -> int:
        total_tokens = sum(self.count_tokens(text) for text in texts)
        self.usage.embedding_tokens += total_tokens
        
        logger.info(f"Embedding - {len(texts)} documents, {total_tokens} tokens")
        return total_tokens
    
    def log_llm_usage(self, input_text: str, output_text: str) -> tuple[int, int]:
        """Registra uso de tokens LLM e retorna (input_tokens, output_tokens)"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.api_calls += 1
        
        return input_tokens, output_tokens
    
    def get_summary(self) -> TokenSummary:
        cost_breakdown = CostBreakdown(
            llm_input_cost=(self.usage.input_tokens / 1000) * self.PRICES['gpt4o_input'],
            llm_output_cost=(self.usage.output_tokens / 1000) * self.PRICES['gpt4o_output'],
            embedding_cost=(self.usage.embedding_tokens / 1000) * self.PRICES['embedding_3_large']
        )
        
        summary = TokenSummary(
            usage=self.usage,
            cost_breakdown=cost_breakdown
        )
        
        logger.info(f"RESUMO DE TOKENS: {summary.to_dict()}")
        return summary
    
    def reset(self):
        self.usage = TokenUsage()
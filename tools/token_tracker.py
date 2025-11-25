"""
Token Tracker - 统计LLM调用的token使用情况
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TokenUsage:
    """单次API调用的token使用"""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # 缓存的输入tokens（如果支持）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def total_tokens(self) -> int:
        """总token数（包括缓存的）"""
        return self.prompt_tokens + self.completion_tokens
    
    @property
    def non_cached_tokens(self) -> int:
        """非缓存的token数（用于计费）"""
        return (self.prompt_tokens - self.cached_tokens) + self.completion_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "non_cached_tokens": self.non_cached_tokens,
            "timestamp": self.timestamp
        }


class TokenTracker:
    """Token使用统计器"""
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self._model_prices: Dict[str, Dict[str, float]] = {
            "gpt-5": {
                "input": 0.625 / 1_000_000,  # $0.625 per 1M tokens
                "cached_input": 0.0625 / 1_000_000,  # $0.0625 per 1M tokens
                "output": 5.00 / 1_000_000  # $5.00 per 1M tokens
            },
            "gpt-5.1": {
                "input": 0.625 / 1_000_000,
                "cached_input": 0.0625 / 1_000_000,
                "output": 5.00 / 1_000_000
            },
            "gpt-5-nano": {
                "input": 0.625 / 1_000_000,  # 假设与gpt-5相同
                "cached_input": 0.0625 / 1_000_000,
                "output": 5.00 / 1_000_000
            },
            "gpt-5-mini": {
                "input": 0.625 / 1_000_000,  # 假设与gpt-5相同
                "cached_input": 0.0625 / 1_000_000,
                "output": 5.00 / 1_000_000
            },
            # 其他模型的默认价格（可以根据需要添加）
            "default": {
                "input": 0.625 / 1_000_000,
                "cached_input": 0.0625 / 1_000_000,
                "output": 5.00 / 1_000_000
            }
        }
    
    def record_usage(self, model: str, prompt_tokens: int = 0, 
                     completion_tokens: int = 0, cached_tokens: int = 0):
        """记录一次token使用"""
        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens
        )
        self.usage_history.append(usage)
        return usage
    
    def get_model_price(self, model: str) -> Dict[str, float]:
        """获取模型价格配置"""
        # 检查精确匹配
        if model in self._model_prices:
            return self._model_prices[model]
        
        # 检查前缀匹配（如 gpt-5-nano 匹配 gpt-5）
        for key, price in self._model_prices.items():
            if key != "default" and model.startswith(key):
                return price
        
        # 返回默认价格
        return self._model_prices["default"]
    
    def calculate_cost(self, usage: TokenUsage) -> float:
        """计算单次调用的成本"""
        prices = self.get_model_price(usage.model)
        
        # 非缓存的输入tokens
        non_cached_input_tokens = usage.prompt_tokens - usage.cached_tokens
        input_cost = non_cached_input_tokens * prices["input"]
        
        # 缓存的输入tokens
        cached_cost = usage.cached_tokens * prices["cached_input"]
        
        # 输出tokens
        output_cost = usage.completion_tokens * prices["output"]
        
        return input_cost + cached_cost + output_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if not self.usage_history:
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_tokens": 0,
                "total_tokens": 0,
                "total_non_cached_tokens": 0,
                "total_cost_usd": 0.0,
                "by_model": {}
            }
        
        total_prompt = sum(u.prompt_tokens for u in self.usage_history)
        total_completion = sum(u.completion_tokens for u in self.usage_history)
        total_cached = sum(u.cached_tokens for u in self.usage_history)
        total_cost = sum(self.calculate_cost(u) for u in self.usage_history)
        
        # 按模型分组统计
        by_model: Dict[str, Dict[str, Any]] = {}
        for usage in self.usage_history:
            if usage.model not in by_model:
                by_model[usage.model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "non_cached_tokens": 0,
                    "cost_usd": 0.0
                }
            
            model_stats = by_model[usage.model]
            model_stats["calls"] += 1
            model_stats["prompt_tokens"] += usage.prompt_tokens
            model_stats["completion_tokens"] += usage.completion_tokens
            model_stats["cached_tokens"] += usage.cached_tokens
            model_stats["total_tokens"] += usage.total_tokens
            model_stats["non_cached_tokens"] += usage.non_cached_tokens
            model_stats["cost_usd"] += self.calculate_cost(usage)
        
        return {
            "total_calls": len(self.usage_history),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": total_cached,
            "total_tokens": total_prompt + total_completion,
            "total_non_cached_tokens": (total_prompt - total_cached) + total_completion,
            "total_cost_usd": total_cost,
            "by_model": by_model
        }
    
    def save_to_file(self, filepath: str):
        """保存统计结果到文件"""
        summary = self.get_summary()
        detailed = {
            "summary": summary,
            "detailed_history": [u.to_dict() for u in self.usage_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 Token Usage Summary")
        print("="*60)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - Prompt Tokens: {summary['total_prompt_tokens']:,}")
        print(f"  - Completion Tokens: {summary['total_completion_tokens']:,}")
        print(f"  - Cached Tokens: {summary['total_cached_tokens']:,}")
        print(f"  - Non-cached Tokens: {summary['total_non_cached_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost_usd']:.4f} USD")
        print("\nBy Model:")
        for model, stats in summary['by_model'].items():
            print(f"  {model}:")
            print(f"    Calls: {stats['calls']}")
            print(f"    Tokens: {stats['total_tokens']:,} (non-cached: {stats['non_cached_tokens']:,})")
            print(f"    Cost: ${stats['cost_usd']:.4f} USD")
        print("="*60 + "\n")


# 全局token tracker实例
_global_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """获取全局token tracker实例"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker


def reset_tracker():
    """重置全局tracker（用于测试）"""
    global _global_tracker
    _global_tracker = None


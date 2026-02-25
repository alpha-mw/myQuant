#!/usr/bin/env python3
"""
统一决策层 - 简化版
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DecisionOutput:
    """决策层输出"""
    ratings: List[Dict] = field(default_factory=list)
    analysis: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class UnifiedDecisionLayer:
    """统一决策层"""
    
    def __init__(self, llm_preference: Optional[List[str]] = None, verbose: bool = True):
        self.llm_preference = llm_preference or ["openai"]
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DecisionLayer] {msg}")
    
    def process(self, ranked_stocks: List[Dict], data_bundle: Any) -> DecisionOutput:
        """处理决策"""
        output = DecisionOutput()
        
        self._log("生成投资建议...")
        
        for stock in ranked_stocks[:5]:
            output.ratings.append({
                'stock': stock.get('code'),
                'rating': '买入' if stock.get('composite_score', 0) > 0 else '持有',
                'score': stock.get('composite_score', 0),
                'reason': '基于多因子模型评分'
            })
        
        output.analysis = f"分析了 {len(ranked_stocks)} 只股票，推荐关注前 5 名"
        self._log(f"生成 {len(output.ratings)} 条投资评级")
        
        return output

"""
OmniDF: AI-Powered Multi-modal DataFrame with Query Optimization.

A research-oriented DataFrame library that supports:
- Traditional relational operations (via Pandas)
- Semantic operations (sem_filter, sem_map, sem_join, sem_dedup)
- Lazy evaluation with DAG-based query planning
- Rule-based and LLM-driven query optimization

Usage:
    import omnidf as odf
    
    # Configure LLM
    odf.settings.configure(oracle_model="gpt-4o-mini")
    
    # Create DataFrame and apply semantic operations
    df = odf.DataFrame(data)
    df = df.sem_filter(user_instruction="...", input_columns=["text"])
    result = df.collect(optimize=True)
"""

from omnidf.dataframe import DataFrame
from omnidf.execution import Executor
from omnidf.optimizer import Optimizer, LLMOptimizer
from omnidf import settings

__version__ = "0.1.0"

__all__ = [
    "DataFrame",
    "Executor",
    "Optimizer",
    "LLMOptimizer",
    "settings",
]

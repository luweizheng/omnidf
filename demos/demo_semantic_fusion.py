"""
Demo: Semantic Filter Fusion Optimization

This demo shows how OmniDF fuses consecutive semantic filters into a single
LLM call, reducing API costs and latency.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import omnidf as odf
from omnidf.optimizer import Optimizer, SemanticFilterFusion


def main():
    # Sample movie data
    movie_data = pd.DataFrame({
        'Title': ['Inception', 'The Dark Knight', 'Interstellar', 'Dunkirk', 'Tenet'],
        'Year': [2010, 2008, 2014, 2017, 2020],
        'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 
                    'Christopher Nolan', 'Christopher Nolan'],
        'Genre': ['Sci-Fi/Action', 'Action/Crime', 'Sci-Fi/Drama', 'War/Drama', 'Sci-Fi/Action']
    })
    
    print("=" * 60)
    print("OmniDF Demo: Semantic Filter Fusion")
    print("=" * 60)
    
    # Build query with consecutive semantic filters
    df = odf.DataFrame(movie_data)
    df = df.filter(predicate="Year > 2010", columns_used=["Year"])
    df = df.semantic_filter(
        user_instruction="The director is Christopher Nolan.",
        input_columns=['Director'],
        model="gpt-4.1"
    )
    df = df.semantic_filter(
        user_instruction="The genre is action movie.",
        input_columns=['Genre'],
        model="gpt-4.1"
    )
    
    print("\n📋 Original Query Plan (Before Optimization):")
    print("-" * 40)
    print(df.explain())
    
    # Apply fusion optimization
    optimizer = Optimizer(rules=[SemanticFilterFusion()])
    optimized_plan = optimizer.optimize(df.plan)
    
    print("\n✨ Optimized Query Plan (After Semantic Filter Fusion):")
    print("-" * 40)
    print(optimized_plan.pretty_print())
    
    print("\n📊 Optimization Benefit:")
    print("-" * 40)
    print("Before: 2 separate LLM calls per row")
    print("After:  1 combined LLM call per row")
    print("        → 50% fewer API calls = Lower cost & latency")


if __name__ == "__main__":
    main()

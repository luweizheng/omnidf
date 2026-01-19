"""
Demo: Semantic Join Decomposition Optimization

This demo shows how OmniDF decomposes semantic joins into semantic map + 
relational join, reducing O(n*m) LLM calls to O(n+m).
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import omnidf as odf
from omnidf.optimizer import Optimizer, SemanticJoinDecomposition


def main():
    # Sample movie reviews data
    reviews_data = pd.DataFrame({
        'review_id': [1, 2, 3, 4, 5],
        'reviewText': [
            'This movie was absolutely fantastic! A masterpiece.',
            'Terrible film, waste of time and money.',
            'Amazing cinematography and brilliant acting.',
            'Boring and predictable plot. Very disappointed.',
            'One of the best movies I have ever seen!'
        ]
    })
    
    print("=" * 60)
    print("OmniDF Demo: Semantic Join Decomposition")
    print("=" * 60)
    
    print("\n📊 Input Data:")
    print(reviews_data)
    
    # Build semantic join query
    reviews = odf.DataFrame(reviews_data)
    
    join_instruction = (
        'These two movie reviews express the same sentiment - '
        'either both are positive or both are negative. '
        'Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'
    )
    
    joined_df = reviews.sem_join(
        reviews,
        join_instruction=join_instruction,
    )
    
    print("\n📋 Original Query Plan (Before Optimization):")
    print("-" * 40)
    print(joined_df.explain())
    
    # Apply decomposition optimization
    optimizer = Optimizer(rules=[SemanticJoinDecomposition()])
    optimized_plan = optimizer.optimize(joined_df.plan)
    
    print("\n✨ Optimized Query Plan (After Decomposition):")
    print("-" * 40)
    print(optimized_plan.pretty_print())
    
    print("\n📊 Optimization Benefit:")
    print("-" * 40)
    print("Before: SemanticJoin requires O(n*m) = 25 LLM calls")
    print("        (comparing every pair of reviews)")
    print()
    print("After:  SemanticMap(left) + SemanticMap(right) + RelationalJoin")
    print("        = O(n+m) = 10 LLM calls")
    print("        (extract sentiment once per review, then join on sentiment)")
    print()
    print("        → 60% fewer LLM calls for this example")
    print("        → For larger datasets, savings grow quadratically!")


if __name__ == "__main__":
    main()

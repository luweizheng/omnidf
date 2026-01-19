"""
Demo: Filter Push Down Optimization

This demo shows how OmniDF optimizes queries by pushing relational filters
before semantic operators, reducing the number of expensive LLM calls.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import omnidf as odf
from omnidf.optimizer import Optimizer, FilterPushDown
from omnidf.visualization import plan_to_dot


def main():
    # Sample real estate data
    estate_data = pd.DataFrame({
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
        'description': [
            'Beautiful house with solar panels on the roof and modern amenities.',
            'Cozy home with a large backyard, perfect for families.',
            'Luxury villa with swimming pool and solar energy system.',
            'Charming cottage with garden and eco-friendly features.',
            'Modern apartment in downtown area with great views.'
        ],
        'price': [1200000, 800000, 2000000, 950000, 1100000]
    })
    
    print("=" * 60)
    print("OmniDF Demo: Filter Push Down Optimization")
    print("=" * 60)
    
    # Build query: semantic filter first, then relational filter
    df = odf.DataFrame(estate_data)
    df = df.semantic_filter(
        user_instruction="Based on the description, the house has solar panels.",
        input_columns=["description"],
        model="gpt-4.1"
    )
    df = df.filter(predicate="price < 1500000", columns_used=["price"])
    
    print("\n📋 Original Query Plan (Before Optimization):")
    print("-" * 40)
    print(df.explain())
    
    # Apply optimization
    optimizer = Optimizer(rules=[FilterPushDown()])
    optimized_plan = optimizer.optimize(df.plan)
    
    print("\n✨ Optimized Query Plan (After Filter Push Down):")
    print("-" * 40)
    print(optimized_plan.pretty_print())
    
    print("\n📊 Optimization Benefit:")
    print("-" * 40)
    print("Before: SemanticFilter processes ALL 5 rows with LLM")
    print("After:  Filter first reduces to 4 rows, then SemanticFilter")
    print("        → Fewer LLM API calls = Lower cost & latency")
    
    # Show JSON representation for LLM optimization
    print("\n📝 JSON Plan (for LLM-based optimization):")
    print("-" * 40)
    print(df.explain(format="json"))


if __name__ == "__main__":
    main()

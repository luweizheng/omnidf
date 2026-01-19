"""
Test cases for Semantic Join Decomposition optimization.

Test Case 3: Decompose semantic join into semantic map + relational join
- Before: SemanticJoin(left, right, "same sentiment")
- After:  Join(SemanticMap(left, "extract sentiment"), SemanticMap(right, "extract sentiment"), on="sentiment")
"""

import pandas as pd
import pytest

import sys
sys.path.insert(0, 'src')

import omnidf as odf
from omnidf.plan import (
    Source, Filter, SemanticFilter, SemanticMap, SemanticJoin, Join, NodeType,
    plan_to_dict, plan_to_json,
)
from omnidf.optimizer import Optimizer, SemanticJoinDecomposition, LLMOptimizer
from omnidf.models import MockLLMClient


class TestSemanticJoinDecomposition:
    """Test semantic join decomposition optimization."""
    
    @pytest.fixture
    def reviews_data(self):
        """Sample movie reviews data."""
        return pd.DataFrame({
            'review_id': [1, 2, 3, 4, 5],
            'reviewText': [
                'This movie was absolutely fantastic! A masterpiece.',
                'Terrible film, waste of time and money.',
                'Amazing cinematography and brilliant acting.',
                'Boring and predictable plot. Very disappointed.',
                'One of the best movies I have ever seen!'
            ]
        })
    
    def test_semantic_join_decomposition_basic(self, reviews_data):
        """
        Test that semantic join is decomposed into semantic map + relational join.
        
        Original query:
            join_instruction = 'These two movie reviews express the same sentiment'
            joined_df = reviews.sem_join(reviews, join_instruction=join_instruction)
        
        Before optimization:
            SemanticJoin(Source, Source, "same sentiment")
        
        After optimization:
            Join(
                SemanticMap(Source, "extract sentiment", output="_left_sentiment"),
                SemanticMap(Source, "extract sentiment", output="_right_sentiment"),
                on=("_left_sentiment", "_right_sentiment")
            )
        
        This reduces O(n*m) LLM calls to O(n+m) calls.
        """
        # Build the query
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
        
        plan_before = joined_df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, SemanticJoin), "Root should be SemanticJoin"
        assert isinstance(plan_before.left, Source), "Left child should be Source"
        assert isinstance(plan_before.right, Source), "Right child should be Source"
        
        print("\n=== Before Optimization ===")
        print(plan_before.pretty_print())
        print("\n=== Plan JSON ===")
        print(plan_to_json(plan_before))
        
        # Apply semantic join decomposition
        optimizer = Optimizer(rules=[SemanticJoinDecomposition()])
        plan_after = optimizer.optimize(plan_before)
        
        print("\n=== After Optimization ===")
        print(plan_after.pretty_print())
        
        # Verify structure after optimization:
        # Root should be Join with SemanticMap children
        assert isinstance(plan_after, Join), f"Root should be Join, got {type(plan_after)}"
        assert isinstance(plan_after.left, SemanticMap), f"Left child should be SemanticMap, got {type(plan_after.left)}"
        assert isinstance(plan_after.right, SemanticMap), f"Right child should be SemanticMap, got {type(plan_after.right)}"
        
        # Verify the semantic maps extract sentiment
        assert "sentiment" in plan_after.left.user_instruction.lower()
        assert "sentiment" in plan_after.right.user_instruction.lower()
        
        # Verify join is on the extracted sentiment columns
        assert "sentiment" in (plan_after.left_on or "") or "sentiment" in (plan_after.on or "")
    
    def test_semantic_join_not_decomposed_for_complex_conditions(self, reviews_data):
        """
        Test that semantic join is NOT decomposed when the condition is too complex.
        """
        reviews = odf.DataFrame(reviews_data)
        
        # A complex join condition that can't be easily decomposed
        join_instruction = (
            'The first review discusses a specific scene that the second review also mentions'
        )
        
        joined_df = reviews.sem_join(
            reviews,
            join_instruction=join_instruction,
        )
        
        plan_before = joined_df.plan
        
        # Apply decomposition rule
        rule = SemanticJoinDecomposition()
        
        # Should not match because the condition doesn't fit decomposable patterns
        assert not rule.matches(plan_before), "Complex conditions should not be decomposed"
    
    def test_decomposition_preserves_semantics(self, reviews_data):
        """
        Test that the decomposed plan preserves the semantic meaning.
        """
        reviews = odf.DataFrame(reviews_data)
        
        join_instruction = 'Reviews have the same sentiment'
        
        joined_df = reviews.sem_join(reviews, join_instruction=join_instruction)
        
        plan_before = joined_df.plan
        
        # Apply optimization
        optimizer = Optimizer(rules=[SemanticJoinDecomposition()])
        plan_after = optimizer.optimize(plan_before)
        
        # Both plans should be executable
        from omnidf.execution import Executor
        
        executor = Executor(use_mock=True)
        
        # Execute original (mock)
        result_before = executor.execute(plan_before)
        executor.clear_cache()
        
        # Execute optimized (mock)
        result_after = executor.execute(plan_after)
        
        # Both should return DataFrames
        assert isinstance(result_before, pd.DataFrame)
        assert isinstance(result_after, pd.DataFrame)
        
        print(f"\nOriginal result shape: {result_before.shape}")
        print(f"Optimized result shape: {result_after.shape}")


class TestSemanticJoinDecompositionPatterns:
    """Test various patterns that can be decomposed."""
    
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Hello', 'World', 'Test']
        })
    
    def test_same_sentiment_pattern(self, data):
        """Test 'same sentiment' pattern is recognized."""
        df = odf.DataFrame(data)
        joined = df.sem_join(df, join_instruction="Reviews have the same sentiment")
        
        rule = SemanticJoinDecomposition()
        assert rule.matches(joined.plan), "'same sentiment' should be decomposable"
    
    def test_same_category_pattern(self, data):
        """Test 'same category' pattern is recognized."""
        df = odf.DataFrame(data)
        joined = df.sem_join(df, join_instruction="Items belong to the same category")
        
        rule = SemanticJoinDecomposition()
        assert rule.matches(joined.plan), "'same category' should be decomposable"
    
    def test_same_topic_pattern(self, data):
        """Test 'same topic' pattern is recognized."""
        df = odf.DataFrame(data)
        joined = df.sem_join(df, join_instruction="Articles discuss the same topic")
        
        rule = SemanticJoinDecomposition()
        assert rule.matches(joined.plan), "'same topic' should be decomposable"
    
    def test_matching_sentiment_pattern(self, data):
        """Test 'matching sentiment' pattern is recognized."""
        df = odf.DataFrame(data)
        joined = df.sem_join(df, join_instruction="Texts have matching sentiment")
        
        rule = SemanticJoinDecomposition()
        assert rule.matches(joined.plan), "'matching sentiment' should be decomposable"


class TestSemanticJoinDecompositionWithLLMOptimizer:
    """Test semantic join decomposition using LLM-based optimizer."""
    
    @pytest.fixture
    def reviews_data(self):
        return pd.DataFrame({
            'review_id': [1, 2, 3, 4, 5],
            'reviewText': [
                'This movie was absolutely fantastic! A masterpiece.',
                'Terrible film, waste of time and money.',
                'Amazing cinematography and brilliant acting.',
                'Boring and predictable plot. Very disappointed.',
                'One of the best movies I have ever seen!'
            ]
        })
    
    @pytest.fixture
    def mock_llm_client(self):
        """Setup mock LLM client for testing."""
        client = odf.settings.configure(use_mock=True)
        return client
    
    def test_llm_optimizer_semantic_join_decomposition(self, reviews_data, mock_llm_client):
        """
        Test that LLM optimizer correctly identifies and applies SemanticJoinDecomposition.
        
        The mock LLM client will analyze the plan and suggest decomposition.
        """
        # Build semantic join query
        reviews = odf.DataFrame(reviews_data)
        
        join_instruction = (
            'These two movie reviews express the same sentiment - '
            'either both are positive or both are negative.'
        )
        
        joined_df = reviews.sem_join(
            reviews,
            join_instruction=join_instruction,
        )
        
        plan_before = joined_df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, SemanticJoin), "Root should be SemanticJoin"
        
        print("\n=== Before LLM Optimization ===")
        print(plan_before.pretty_print())
        
        # Use LLM optimizer
        llm_optimizer = LLMOptimizer(llm_client=mock_llm_client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== After LLM Optimization ===")
        print(plan_after.pretty_print())
        print(f"\nOptimization info: {info}")
        
        # Verify structure after optimization:
        # Root should be Join with SemanticMap children
        assert isinstance(plan_after, Join), f"Root should be Join, got {type(plan_after)}"
        assert isinstance(plan_after.left, SemanticMap), f"Left child should be SemanticMap, got {type(plan_after.left)}"
        assert isinstance(plan_after.right, SemanticMap), f"Right child should be SemanticMap, got {type(plan_after.right)}"
        
        # Verify optimization info
        assert "optimizations" in info or "method" in info
    
    def test_llm_optimizer_preserves_complex_joins(self, reviews_data, mock_llm_client):
        """
        Test that LLM optimizer does NOT decompose complex join conditions.
        """
        reviews = odf.DataFrame(reviews_data)
        
        # Complex condition that shouldn't be decomposed
        join_instruction = 'The first review discusses a scene mentioned in the second review'
        
        joined_df = reviews.sem_join(reviews, join_instruction=join_instruction)
        
        plan_before = joined_df.plan
        
        # Use LLM optimizer
        llm_optimizer = LLMOptimizer(llm_client=mock_llm_client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== Complex Join - After LLM Optimization ===")
        print(plan_after.pretty_print())
        
        # Complex conditions should remain as SemanticJoin
        assert isinstance(plan_after, SemanticJoin), "Complex join should not be decomposed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

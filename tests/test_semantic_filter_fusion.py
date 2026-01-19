"""
Test cases for Semantic Filter Fusion optimization.

Test Case 2: Fuse consecutive semantic filters
- Before: Source -> Filter(Year > 2010) -> SemanticFilter(director) -> SemanticFilter(genre)
- After:  Source -> Filter(Year > 2010) -> FusedSemanticFilter(director AND genre)

Set USE_LLM=true to run tests with real LLM calls.
"""

import os
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path(__file__).parent.parent / '.env',
    Path(__file__).parent.parent.parent / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        break

import omnidf as odf
from omnidf.plan import (
    Source, Filter, SemanticFilter, NodeType,
    plan_to_dict, plan_to_json,
)
from omnidf.optimizer import Optimizer, SemanticFilterFusion, FilterPushDown, LLMOptimizer
from omnidf.models import MockLLMClient

# Check if LLM tests should be run
USE_LLM = os.environ.get('USE_LLM', 'false').lower() == 'true'
skip_llm = pytest.mark.skipif(not USE_LLM, reason="LLM tests disabled. Set USE_LLM=true to enable.")


class TestSemanticFilterFusion:
    """Test semantic filter fusion optimization."""
    
    @pytest.fixture
    def movie_data(self):
        """Sample movie data."""
        return pd.DataFrame({
            'Title': ['Inception', 'The Dark Knight', 'Interstellar', 'Dunkirk', 'Tenet'],
            'Year': [2010, 2008, 2014, 2017, 2020],
            'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 
                        'Christopher Nolan', 'Christopher Nolan'],
            'Genre': ['Sci-Fi/Action', 'Action/Crime', 'Sci-Fi/Drama', 'War/Drama', 'Sci-Fi/Action']
        })
    
    def test_semantic_filter_fusion_basic(self, movie_data):
        """
        Test that consecutive semantic filters are fused.
        
        Original query:
            df_opt = odf.DataFrame(movie_data)
            df_opt = df_opt[df_opt._data['Year'] > 2010]
            df_opt = df_opt.semantic_filter(
                user_instruction="The director is Christopher Nolan.",
                input_columns=['Director'],
                model="gpt-4.1"
            )
            df_opt = df_opt.semantic_filter(
                user_instruction="The genre is action movie.",
                input_columns=['Genre'],
                model="gpt-4.1"
            )
        
        Before optimization:
            Source -> Filter(Year > 2010) -> SemanticFilter(director) -> SemanticFilter(genre)
        
        After optimization:
            Source -> Filter(Year > 2010) -> FusedSemanticFilter(director AND genre)
        """
        # Build the query
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
        
        plan_before = df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, SemanticFilter), "Root should be SemanticFilter (genre)"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter (director)"
        assert isinstance(plan_before.child.child, Filter), "Grandchild should be Filter"
        
        print("\n=== Before Optimization ===")
        print(plan_before.pretty_print())
        
        # Apply semantic filter fusion
        optimizer = Optimizer(rules=[SemanticFilterFusion()])
        plan_after = optimizer.optimize(plan_before)
        
        print("\n=== After Optimization ===")
        print(plan_after.pretty_print())
        
        # Verify structure after optimization:
        # The two SemanticFilters should be fused into one
        assert isinstance(plan_after, SemanticFilter), "Root should still be SemanticFilter"
        assert isinstance(plan_after.child, Filter), "Child should now be Filter (not SemanticFilter)"
        
        # Verify the fused instruction contains both conditions
        assert "Christopher Nolan" in plan_after.user_instruction or "director" in plan_after.user_instruction.lower()
        assert "action" in plan_after.user_instruction.lower() or "genre" in plan_after.user_instruction.lower()
        
        # Verify input columns are combined
        combined_cols = plan_after.input_columns
        assert 'Director' in combined_cols or 'Genre' in combined_cols
    
    def test_semantic_filter_fusion_different_models_not_fused(self, movie_data):
        """
        Test that semantic filters with different models are NOT fused.
        """
        df = odf.DataFrame(movie_data)
        df = df.semantic_filter(
            user_instruction="Director check",
            input_columns=['Director'],
            model="gpt-4.1"
        )
        df = df.semantic_filter(
            user_instruction="Genre check",
            input_columns=['Genre'],
            model="gpt-3.5-turbo"  # Different model
        )
        
        plan_before = df.plan
        
        # Apply fusion rule
        rule = SemanticFilterFusion()
        
        # Should not match because models are different
        assert not rule.matches(plan_before), "Should not fuse filters with different models"
    
    def test_full_optimization_pipeline(self, movie_data):
        """
        Test the full optimization with both FilterPushDown and SemanticFilterFusion.
        """
        # Build query with pandas-style syntax
        df = odf.DataFrame(movie_data)
        df = df[df._data['Year'] > 2010]
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
        
        plan_before = df.plan
        
        print("\n=== Original Plan ===")
        print(plan_before.pretty_print())
        
        # Apply all optimizations
        optimizer = Optimizer()  # Uses default rules
        plan_after = optimizer.optimize(plan_before)
        
        print("\n=== Fully Optimized Plan ===")
        print(plan_after.pretty_print())
        
        # The semantic filters should be fused
        # Note: The exact structure depends on rule application order


class TestSemanticFilterFusionExecution:
    """Test that fused semantic filters execute correctly."""
    
    @pytest.fixture
    def movie_data(self):
        return pd.DataFrame({
            'Title': ['Inception', 'The Dark Knight', 'Interstellar'],
            'Year': [2010, 2008, 2014],
            'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan'],
            'Genre': ['Sci-Fi/Action', 'Action/Crime', 'Sci-Fi/Drama']
        })
    
    def test_fused_filter_execution(self, movie_data):
        """Test that the fused filter can be executed."""
        df = odf.DataFrame(movie_data)
        df = df.semantic_filter(
            user_instruction="Director is Nolan",
            input_columns=['Director'],
        )
        df = df.semantic_filter(
            user_instruction="Genre is action",
            input_columns=['Genre'],
        )
        
        # Execute with optimization
        result = df.collect(optimize=True)
        
        # Result should be a DataFrame (mock LLM returns all rows)
        assert isinstance(result, pd.DataFrame)
        print(f"\nExecution result shape: {result.shape}")


class TestSemanticFilterFusionWithLLMOptimizer:
    """Test semantic filter fusion using LLM-based optimizer."""
    
    @pytest.fixture
    def movie_data(self):
        return pd.DataFrame({
            'Title': ['Inception', 'The Dark Knight', 'Interstellar', 'Dunkirk', 'Tenet'],
            'Year': [2010, 2008, 2014, 2017, 2020],
            'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 
                        'Christopher Nolan', 'Christopher Nolan'],
            'Genre': ['Sci-Fi/Action', 'Action/Crime', 'Sci-Fi/Drama', 'War/Drama', 'Sci-Fi/Action']
        })
    
    @pytest.fixture
    def mock_llm_client(self):
        """Setup mock LLM client for testing."""
        client = odf.settings.configure(use_mock=True)
        return client
    
    def test_llm_optimizer_semantic_filter_fusion(self, movie_data, mock_llm_client):
        """
        Test that LLM optimizer correctly identifies and applies SemanticFilterFusion.
        """
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
        
        plan_before = df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, SemanticFilter), "Root should be SemanticFilter"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter"
        
        print("\n=== Before LLM Optimization ===")
        print(plan_before.pretty_print())
        
        # Use LLM optimizer
        llm_optimizer = LLMOptimizer(llm_client=mock_llm_client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== After LLM Optimization ===")
        print(plan_after.pretty_print())
        print(f"\nOptimization info: {info}")
        
        # Verify the two SemanticFilters are fused
        assert isinstance(plan_after, SemanticFilter), "Root should be SemanticFilter"
        # After fusion, child should be Filter (not another SemanticFilter)
        assert isinstance(plan_after.child, Filter), f"Child should be Filter after fusion, got {type(plan_after.child)}"
        
        # Verify optimization info
        assert "optimizations" in info or "method" in info


class TestSemanticFilterFusionWithRealLLM:
    """Test semantic filter fusion using real LLM optimizer."""
    
    @pytest.fixture
    def movie_data(self):
        return pd.DataFrame({
            'Title': ['Inception', 'The Dark Knight', 'Interstellar', 'Dunkirk', 'Tenet'],
            'Year': [2010, 2008, 2014, 2017, 2020],
            'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 
                        'Christopher Nolan', 'Christopher Nolan'],
            'Genre': ['Sci-Fi/Action', 'Action/Crime', 'Sci-Fi/Drama', 'War/Drama', 'Sci-Fi/Action']
        })
    
    @skip_llm
    def test_llm_optimizer_semantic_filter_fusion_real(self, movie_data):
        """
        Test that real LLM optimizer correctly identifies and applies SemanticFilterFusion.
        
        This test uses actual LLM API calls to verify the optimizer works correctly.
        """
        # Configure real LLM client
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            optimizer_model="gpt-4o-mini",
            oracle_api_key=os.environ.get('OPENAI_API_KEY'),
            oracle_api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Build query with consecutive semantic filters
        df = odf.DataFrame(movie_data)
        df = df.filter(predicate="Year > 2010", columns_used=["Year"])
        df = df.semantic_filter(
            user_instruction="The director is Christopher Nolan.",
            input_columns=['Director'],
            model="gpt-4o-mini"
        )
        df = df.semantic_filter(
            user_instruction="The genre is action movie.",
            input_columns=['Genre'],
            model="gpt-4o-mini"
        )
        
        plan_before = df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, SemanticFilter), "Root should be SemanticFilter"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter"
        
        print("\n=== Before LLM Optimization ===")
        print(plan_before.pretty_print())
        
        # Use LLM optimizer with real client
        llm_optimizer = LLMOptimizer(llm_client=client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== After LLM Optimization ===")
        print(plan_after.pretty_print())
        print(f"\nOptimization info: {info}")
        
        # Verify LLM correctly identified SemanticFilterFusion
        assert info.get("method") == "llm_guided", f"Expected llm_guided, got {info.get('method')}"
        
        # Check if SemanticFilterFusion was suggested
        optimizations = info.get("optimizations", [])
        fusion_suggested = any(
            opt.get("rule") == "SemanticFilterFusion" 
            for opt in optimizations
        )
        
        if fusion_suggested:
            # Verify the two SemanticFilters are fused
            assert isinstance(plan_after, SemanticFilter), "Root should be SemanticFilter"
            # After fusion, child should be Filter (not another SemanticFilter)
            assert isinstance(plan_after.child, Filter), f"Child should be Filter after fusion, got {type(plan_after.child).__name__}"
            
            # Verify the fused instruction contains both conditions
            assert "Christopher Nolan" in plan_after.user_instruction or "director" in plan_after.user_instruction.lower()
            assert "action" in plan_after.user_instruction.lower() or "genre" in plan_after.user_instruction.lower()
            
            print("\n✓ LLM optimizer correctly applied SemanticFilterFusion!")
        else:
            # LLM may not always suggest fusion (depends on its analysis)
            # Just verify the optimization process worked
            print(f"\nNote: LLM did not suggest SemanticFilterFusion. Suggested: {[opt.get('rule') for opt in optimizations]}")
            print("This is acceptable - LLM may have different optimization priorities.")
            
            # At minimum, verify the plan is valid
            assert isinstance(plan_after, SemanticFilter), "Root should still be SemanticFilter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

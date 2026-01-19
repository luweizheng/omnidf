"""
Test cases for Filter Push Down optimization.

Test Case 1: Push relational filter before semantic filter
- Before: Source -> SemanticFilter -> Filter(price < 1500000)
- After:  Source -> Filter(price < 1500000) -> SemanticFilter

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
from omnidf.optimizer import Optimizer, FilterPushDown, LLMOptimizer
from omnidf.models import MockLLMClient

# Check if LLM tests should be run
USE_LLM = os.environ.get('USE_LLM', 'false').lower() == 'true'
skip_llm = pytest.mark.skipif(not USE_LLM, reason="LLM tests disabled. Set USE_LLM=true to enable.")


class TestFilterPushDown:
    """Test filter push down optimization."""
    
    @pytest.fixture
    def estate_data(self):
        """Sample real estate data."""
        return pd.DataFrame({
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
    
    def test_filter_pushdown_basic(self, estate_data):
        """
        Test that relational filter is pushed before semantic filter.
        
        Original query:
            df = odf.DataFrame(estate_data).semantic_filter(
                user_instruction="Based on the description, the house has solar panels.",
                input_columns=["description"],
                model="gpt-4.1"
            )
            df = df[df._data['price'] < 1500000]
        
        Before optimization:
            Source -> SemanticFilter -> Filter(price < 1500000)
        
        After optimization:
            Source -> Filter(price < 1500000) -> SemanticFilter
        """
        # Build the query (without optimization)
        df = odf.DataFrame(estate_data)
        df = df.semantic_filter(
            user_instruction="Based on the description, the house has solar panels.",
            input_columns=["description"],
            model="gpt-4.1"
        )
        
        # Apply relational filter using the filter method with explicit columns
        df = df.filter(
            predicate="price < 1500000",
            columns_used=["price"]
        )
        
        # Get the unoptimized plan
        plan_before = df.plan
        
        # Verify structure before optimization:
        # Root should be Filter, child should be SemanticFilter
        assert isinstance(plan_before, Filter), "Root should be Filter"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter"
        assert isinstance(plan_before.child.child, Source), "Grandchild should be Source"
        
        print("\n=== Before Optimization ===")
        print(plan_before.pretty_print())
        
        # Apply optimization
        optimizer = Optimizer(rules=[FilterPushDown()])
        plan_after = optimizer.optimize(plan_before)
        
        print("\n=== After Optimization ===")
        print(plan_after.pretty_print())
        
        # Verify structure after optimization:
        # Root should be SemanticFilter, child should be Filter
        assert isinstance(plan_after, SemanticFilter), f"Root should be SemanticFilter, got {type(plan_after)}"
        assert isinstance(plan_after.child, Filter), f"Child should be Filter, got {type(plan_after.child)}"
        assert isinstance(plan_after.child.child, Source), f"Grandchild should be Source, got {type(plan_after.child.child)}"
        
        # Verify the filter predicate is preserved
        assert plan_after.child.predicate == "price < 1500000"
        
        # Verify the semantic filter instruction is preserved
        assert "solar panels" in plan_after.user_instruction
    
    def test_filter_not_pushed_when_depends_on_semantic_output(self, estate_data):
        """
        Test that filter is NOT pushed when it depends on semantic map output.
        
        If we have:
            Source -> SemanticMap(output='sentiment') -> Filter(sentiment == 'positive')
        
        The filter cannot be pushed because 'sentiment' column doesn't exist before SemanticMap.
        """
        df = odf.DataFrame(estate_data)
        df = df.semantic_map(
            user_instruction="Extract sentiment",
            output_column="sentiment",
            input_columns=["description"],
            model="gpt-4.1"
        )
        
        # Filter on the output column of semantic map
        df = df.filter(
            predicate="sentiment == 'positive'",
            columns_used=["sentiment"]  # This column is produced by SemanticMap
        )
        
        plan_before = df.plan
        
        # The FilterPushDown rule should NOT match because filter depends on semantic output
        rule = FilterPushDown()
        
        # Check if rule matches - it should match the pattern but we need to verify
        # the optimization doesn't break semantics
        # In our current implementation, we check if filter uses output_column
        from omnidf.plan import SemanticMap
        assert isinstance(plan_before, Filter)
        assert isinstance(plan_before.child, SemanticMap)
        
        # The rule should not match because filter uses the output column
        assert not rule.matches(plan_before), "Filter should not be pushed when it uses semantic output column"
    
    def test_plan_serialization(self, estate_data):
        """Test that plans can be serialized to JSON for LLM optimization."""
        df = odf.DataFrame(estate_data)
        df = df.semantic_filter(
            user_instruction="Has solar panels",
            input_columns=["description"],
        )
        df = df.filter(predicate="price < 1500000", columns_used=["price"])
        
        # Serialize to dict
        plan_dict = plan_to_dict(df.plan)
        
        assert "root_id" in plan_dict
        assert "nodes" in plan_dict
        assert len(plan_dict["nodes"]) == 3  # Source, SemanticFilter, Filter
        
        # Serialize to JSON
        plan_json = plan_to_json(df.plan)
        assert isinstance(plan_json, str)
        
        print("\n=== Plan JSON ===")
        print(plan_json)


class TestFilterPushDownWithPandasSyntax:
    """Test filter push down with pandas-style syntax."""
    
    @pytest.fixture
    def estate_data(self):
        return pd.DataFrame({
            'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'description': [
                'House with solar panels.',
                'Cozy home.',
                'Villa with solar energy.',
            ],
            'price': [1200000, 800000, 2000000]
        })
    
    def test_pandas_style_filter(self, estate_data):
        """
        Test the exact syntax from the user's example:
        
        df_no_opt = odf.DataFrame(estate_data).semantic_filter(
            user_instruction="Based on the description, the house has solar panels.",
            input_columns=["description"],
            model="gpt-4.1"
        )
        df_no_opt = df_no_opt[df_no_opt._data['price'] < 1500000]
        """
        # This is the exact user syntax
        df = odf.DataFrame(estate_data).semantic_filter(
            user_instruction="Based on the description, the house has solar panels.",
            input_columns=["description"],
            model="gpt-4.1"
        )
        
        # Using pandas-style boolean indexing
        # Note: df._data gives access to the source data for predicate building
        df = df[df._data['price'] < 1500000]
        
        plan = df.plan
        
        # Verify the plan structure
        assert isinstance(plan, Filter), "Root should be Filter"
        assert isinstance(plan.child, SemanticFilter), "Child should be SemanticFilter"
        
        print("\n=== Plan with Pandas Syntax ===")
        print(plan.pretty_print())
        
        # Apply optimization
        optimizer = Optimizer(rules=[FilterPushDown()])
        optimized = optimizer.optimize(plan)
        
        print("\n=== Optimized Plan ===")
        print(optimized.pretty_print())
        
        # After optimization, SemanticFilter should be on top
        assert isinstance(optimized, SemanticFilter), "Optimized root should be SemanticFilter"
        assert isinstance(optimized.child, Filter), "Optimized child should be Filter"


class TestFilterPushDownWithLLMOptimizer:
    """Test filter push down using LLM-based optimizer."""
    
    @pytest.fixture
    def estate_data(self):
        return pd.DataFrame({
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
    
    @pytest.fixture
    def mock_llm_client(self):
        """Setup mock LLM client for testing."""
        client = odf.settings.configure(use_mock=True)
        return client
    
    def test_llm_optimizer_filter_pushdown(self, estate_data, mock_llm_client):
        """
        Test that LLM optimizer correctly identifies and applies FilterPushDown.
        
        The mock LLM client will analyze the plan and suggest FilterPushDown.
        """
        # Build query with filter after semantic filter
        df = odf.DataFrame(estate_data)
        df = df.semantic_filter(
            user_instruction="Based on the description, the house has solar panels.",
            input_columns=["description"],
            model="gpt-4.1"
        )
        df = df.filter(predicate="price < 1500000", columns_used=["price"])
        
        plan_before = df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, Filter), "Root should be Filter"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter"
        
        print("\n=== Before LLM Optimization ===")
        print(plan_before.pretty_print())
        
        # Use LLM optimizer
        llm_optimizer = LLMOptimizer(llm_client=mock_llm_client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== After LLM Optimization ===")
        print(plan_after.pretty_print())
        print(f"\nOptimization info: {info}")
        
        # Verify structure after optimization:
        # Root should be SemanticFilter, child should be Filter
        assert isinstance(plan_after, SemanticFilter), f"Root should be SemanticFilter, got {type(plan_after)}"
        assert isinstance(plan_after.child, Filter), f"Child should be Filter, got {type(plan_after.child)}"
        
        # Verify optimization info
        assert "optimizations" in info or "method" in info
    
    def test_llm_optimizer_analysis(self, estate_data, mock_llm_client):
        """
        Test that LLM optimizer can analyze a plan and provide suggestions.
        """
        df = odf.DataFrame(estate_data)
        df = df.semantic_filter(
            user_instruction="Has solar panels",
            input_columns=["description"],
        )
        df = df.filter(predicate="price < 1500000", columns_used=["price"])
        
        llm_optimizer = LLMOptimizer(llm_client=mock_llm_client)
        analysis = llm_optimizer.get_optimization_analysis(df.plan)
        
        print("\n=== LLM Optimization Analysis ===")
        print(analysis)
        
        # Should have analysis or optimizations
        assert "analysis" in analysis or "optimizations" in analysis


class TestFilterPushDownWithRealLLM:
    """Test filter push down using real LLM optimizer."""
    
    @pytest.fixture
    def estate_data(self):
        return pd.DataFrame({
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
    
    @skip_llm
    def test_llm_optimizer_filter_pushdown_real(self, estate_data):
        """
        Test that real LLM optimizer correctly identifies and applies FilterPushDown.
        
        This test uses actual LLM API calls to verify the optimizer works correctly.
        """
        # Configure real LLM client
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            optimizer_model="gpt-4o-mini",
            oracle_api_key=os.environ.get('OPENAI_API_KEY'),
            oracle_api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Build query with filter after semantic filter
        df = odf.DataFrame(estate_data)
        df = df.semantic_filter(
            user_instruction="Based on the description, the house has solar panels.",
            input_columns=["description"],
            model="gpt-4o-mini"
        )
        df = df.filter(predicate="price < 1500000", columns_used=["price"])
        
        plan_before = df.plan
        
        # Verify structure before optimization
        assert isinstance(plan_before, Filter), "Root should be Filter"
        assert isinstance(plan_before.child, SemanticFilter), "Child should be SemanticFilter"
        
        print("\n=== Before LLM Optimization ===")
        print(plan_before.pretty_print())
        
        # Use LLM optimizer with real client
        llm_optimizer = LLMOptimizer(llm_client=client)
        plan_after, info = llm_optimizer.optimize(plan_before)
        
        print("\n=== After LLM Optimization ===")
        print(plan_after.pretty_print())
        print(f"\nOptimization info: {info}")
        
        # Verify LLM correctly identified FilterPushDown
        assert info.get("method") == "llm_guided", f"Expected llm_guided, got {info.get('method')}"
        
        # Verify structure after optimization:
        # Root should be SemanticFilter, child should be Filter (filter pushed down)
        assert isinstance(plan_after, SemanticFilter), f"Root should be SemanticFilter after pushdown, got {type(plan_after).__name__}"
        assert isinstance(plan_after.child, Filter), f"Child should be Filter after pushdown, got {type(plan_after.child).__name__}"
        
        # Verify the filter predicate is preserved
        assert plan_after.child.predicate == "price < 1500000"
        
        print("\n✓ LLM optimizer correctly applied FilterPushDown!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

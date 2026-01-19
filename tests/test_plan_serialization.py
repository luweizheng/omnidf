"""
Test cases for plan serialization and LLM-friendly representation.

These tests verify that plans can be serialized to JSON and reconstructed,
which is essential for LLM-based query optimization.
"""

import json
import pandas as pd
import pytest

import sys
sys.path.insert(0, 'src')

import omnidf as odf
from omnidf.plan import (
    Source, Filter, SemanticFilter, SemanticMap, SemanticJoin, Join,
    plan_to_dict, plan_from_dict, plan_to_json, plan_from_json,
)
from omnidf.optimizer import Optimizer, LLMOptimizer


class TestPlanSerialization:
    """Test plan serialization to/from JSON."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30],
            'description': ['Engineer', 'Designer']
        })
    
    def test_simple_plan_to_dict(self, sample_data):
        """Test serializing a simple plan to dictionary."""
        df = odf.DataFrame(sample_data)
        df = df.filter(predicate="age > 25", columns_used=["age"])
        
        plan_dict = plan_to_dict(df.plan)
        
        assert "root_id" in plan_dict
        assert "nodes" in plan_dict
        assert len(plan_dict["nodes"]) == 2  # Source + Filter
        
        # Check node types
        node_types = {n["type"] for n in plan_dict["nodes"]}
        assert "source" in node_types
        assert "filter" in node_types
    
    def test_complex_plan_to_dict(self, sample_data):
        """Test serializing a complex plan with semantic operators."""
        df = odf.DataFrame(sample_data)
        df = df.semantic_filter(
            user_instruction="Person is technical",
            input_columns=["description"]
        )
        df = df.filter(predicate="age > 25", columns_used=["age"])
        
        plan_dict = plan_to_dict(df.plan)
        
        assert len(plan_dict["nodes"]) == 3  # Source + SemanticFilter + Filter
        
        # Verify children references
        root_node = next(n for n in plan_dict["nodes"] if n["node_id"] == plan_dict["root_id"])
        assert len(root_node["children"]) == 1
    
    def test_plan_roundtrip(self, sample_data):
        """Test that plan can be serialized and deserialized."""
        df = odf.DataFrame(sample_data)
        df = df.semantic_filter(
            user_instruction="Technical person",
            input_columns=["description"],
            model="gpt-4.1"
        )
        df = df.filter(predicate="age > 25", columns_used=["age"])
        
        original_plan = df.plan
        
        # Serialize
        plan_dict = plan_to_dict(original_plan)
        
        # Deserialize
        reconstructed_plan = plan_from_dict(plan_dict)
        
        # Verify structure
        assert type(reconstructed_plan) == type(original_plan)
        assert len(reconstructed_plan.children) == len(original_plan.children)
        
        # Verify attributes
        if isinstance(original_plan, Filter):
            assert reconstructed_plan.predicate == original_plan.predicate
    
    def test_json_roundtrip(self, sample_data):
        """Test JSON serialization roundtrip."""
        df = odf.DataFrame(sample_data)
        df = df.semantic_map(
            user_instruction="Extract role",
            output_column="role",
            input_columns=["description"]
        )
        
        original_plan = df.plan
        
        # To JSON
        json_str = plan_to_json(original_plan)
        assert isinstance(json_str, str)
        
        # Parse JSON to verify it's valid
        parsed = json.loads(json_str)
        assert "root_id" in parsed
        
        # From JSON
        reconstructed = plan_from_json(json_str)
        assert isinstance(reconstructed, SemanticMap)
        assert reconstructed.output_column == "role"
    
    def test_join_plan_serialization(self, sample_data):
        """Test serialization of join plans."""
        df1 = odf.DataFrame(sample_data)
        df2 = odf.DataFrame(pd.DataFrame({
            'name': ['Alice', 'Charlie'],
            'score': [95, 85]
        }))
        
        joined = df1.merge(df2, on='name', how='inner')
        
        plan_dict = plan_to_dict(joined.plan)
        
        # Should have 3 nodes: 2 Sources + 1 Join
        assert len(plan_dict["nodes"]) == 3
        
        # Verify join node has 2 children
        join_node = next(n for n in plan_dict["nodes"] if n["type"] == "join")
        assert len(join_node["children"]) == 2
        assert join_node["on"] == "name"
        assert join_node["how"] == "inner"


class TestLLMFriendlyFormat:
    """Test that the serialization format is suitable for LLM consumption."""
    
    @pytest.fixture
    def complex_plan_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Hello', 'World', 'Test'],
            'category': ['A', 'B', 'A']
        })
    
    def test_plan_is_human_readable(self, complex_plan_data):
        """Test that serialized plan is human/LLM readable."""
        df = odf.DataFrame(complex_plan_data)
        df = df.filter(predicate="category == 'A'", columns_used=["category"])
        df = df.semantic_filter(
            user_instruction="Text is a greeting",
            input_columns=["text"]
        )
        df = df.semantic_map(
            user_instruction="Translate to Spanish",
            output_column="spanish_text",
            input_columns=["text"]
        )
        
        json_str = plan_to_json(df.plan, indent=2)
        
        print("\n=== LLM-Friendly Plan Representation ===")
        print(json_str)
        
        # Verify key information is present
        assert "semantic_filter" in json_str
        assert "semantic_map" in json_str
        assert "filter" in json_str
        assert "Text is a greeting" in json_str
        assert "Translate to Spanish" in json_str
    
    def test_optimization_suggestions_format(self, complex_plan_data):
        """Test that optimization suggestions are clear for LLM."""
        df = odf.DataFrame(complex_plan_data)
        df = df.semantic_filter(
            user_instruction="Important text",
            input_columns=["text"]
        )
        df = df.filter(predicate="category == 'A'", columns_used=["category"])
        
        optimizer = Optimizer()
        suggestions = optimizer.get_optimization_suggestions(df.plan)
        
        print("\n=== Optimization Suggestions ===")
        for s in suggestions:
            print(f"- {s['rule']}: {s['description']}")
        
        # Should suggest filter pushdown
        assert any(s['rule'] == 'FilterPushDown' for s in suggestions)


class TestLLMOptimizer:
    """Test the LLM optimizer interface."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'text': ['Hello', 'World'],
            'value': [1, 2]
        })
    
    def test_llm_optimizer_prompt_generation(self, sample_data):
        """Test that LLM optimizer generates appropriate prompts."""
        df = odf.DataFrame(sample_data)
        df = df.semantic_filter(
            user_instruction="Positive text",
            input_columns=["text"]
        )
        df = df.filter(predicate="value > 1", columns_used=["value"])
        
        llm_optimizer = LLMOptimizer()
        prompt = llm_optimizer.get_plan_prompt(df.plan)
        
        print("\n=== LLM Optimization Prompt ===")
        print(prompt[:500] + "...")
        
        # Verify prompt contains key elements
        assert "query optimizer" in prompt.lower()
        assert "semantic" in prompt.lower()
        assert "FilterPushDown" in prompt
    
    def test_llm_optimizer_with_mock(self, sample_data):
        """Test that LLM optimizer works with mock client."""
        # Configure mock LLM client
        odf.settings.configure(use_mock=True)
        
        df = odf.DataFrame(sample_data)
        df = df.semantic_filter(
            user_instruction="Positive text",
            input_columns=["text"]
        )
        df = df.filter(predicate="value > 1", columns_used=["value"])
        
        llm_optimizer = LLMOptimizer()
        optimized_plan, info = llm_optimizer.optimize(df.plan)
        
        # With mock client, should use llm_guided method
        assert info["method"] == "llm_guided"
        assert "optimizations" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

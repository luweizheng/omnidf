"""Test cases for Semantic Operators using DataFrame API.

Tests semantic operators (sem_filter, sem_map, sem_join, sem_dedup) with both:
- Mock mode: For fast unit testing without LLM calls
- LLM mode: For integration testing with real LLM service

Set USE_LLM=true environment variable to run LLM tests.
Requires OPENAI_API_KEY and OPENAI_API_BASE in .env file.
"""

import os
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv

# Load environment variables - try multiple paths
env_paths = [
    Path(__file__).parent.parent.parent.parent / '.env',  # /omnidf/.env
    Path(__file__).parent.parent / '.env',  # /omnidf/omnidf/.env
    Path.cwd().parent / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        break

import omnidf as odf
from omnidf.execution import Executor
from omnidf.optimizer import Optimizer


# Check if LLM tests should be run
USE_LLM = os.environ.get('USE_LLM', 'false').lower() == 'true'
skip_llm = pytest.mark.skipif(not USE_LLM, reason="LLM tests disabled. Set USE_LLM=true to enable.")


class TestSemanticFilter:
    """Test sem_filter using DataFrame API."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': [
                'I love this product! It is amazing.',
                'Terrible experience, would not recommend.',
                'Great quality and fast shipping.',
                'Waste of money, very disappointed.',
                'Excellent service, will buy again!'
            ],
            'category': ['electronics', 'clothing', 'electronics', 'home', 'electronics']
        })
    
    def test_sem_filter_mock(self, sample_data):
        """Test sem_filter with mock implementation."""
        # Create DataFrame and apply semantic filter
        df = odf.DataFrame(sample_data)
        df = df.sem_filter(
            user_instruction="The review expresses positive sentiment",
            input_columns=["text"],
        )
        
        # Execute with mock
        result = df.collect(optimize=True)
        
        # Mock returns all rows
        assert len(result) == len(sample_data)
        assert isinstance(result, pd.DataFrame)
    
    @skip_llm
    def test_sem_filter_llm(self, sample_data):
        """Test sem_filter with real LLM calls."""
        # Configure LLM using settings API
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Create DataFrame and apply semantic filter
        df = odf.DataFrame(sample_data)
        df = df.sem_filter(
            user_instruction="The review expresses positive sentiment",
            input_columns=["text"],
            model="gpt-4o-mini",
        )
        
        # Execute with LLM
        executor = Executor(llm_client=client, use_mock=False)
        result = executor.execute(df.plan)
        
        print(f"\n=== Semantic Filter LLM Result ===")
        print(f"Original rows: {len(sample_data)}")
        print(f"Filtered rows: {len(result)}")
        print(result)
        
        # Should filter to positive reviews (ids 1, 3, 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)


class TestSemanticMap:
    """Test sem_map using DataFrame API."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': [
                'I love this product! It is amazing.',
                'Terrible experience, would not recommend.',
                'Great quality and fast shipping.',
            ]
        })
    
    def test_sem_map_mock(self, sample_data):
        """Test sem_map with mock implementation."""
        # Create DataFrame and apply semantic map
        df = odf.DataFrame(sample_data)
        df = df.sem_map(
            user_instruction="Extract the sentiment (positive, negative, or neutral)",
            input_columns=["text"],
            output_column="sentiment",
        )
        
        # Execute with mock
        result = df.collect(optimize=True)
        
        # Mock adds column with placeholder value
        assert 'sentiment' in result.columns
        assert len(result) == len(sample_data)
    
    @skip_llm
    def test_sem_map_llm(self, sample_data):
        """Test sem_map with real LLM calls."""
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Create DataFrame and apply semantic map
        df = odf.DataFrame(sample_data)
        df = df.sem_map(
            user_instruction="Extract the sentiment (positive, negative, or neutral)",
            input_columns=["text"],
            output_column="sentiment",
            model="gpt-4o-mini",
        )
        
        # Execute with LLM
        executor = Executor(llm_client=client, use_mock=False)
        result = executor.execute(df.plan)
        
        print(f"\n=== Semantic Map LLM Result ===")
        print(result[['text', 'sentiment']])
        
        assert 'sentiment' in result.columns
        assert len(result) == len(sample_data)
        # Check that sentiments are extracted
        for sentiment in result['sentiment']:
            assert sentiment.lower() in ['positive', 'negative', 'neutral'] or \
                   any(s in sentiment.lower() for s in ['positive', 'negative', 'neutral'])


class TestSemanticJoin:
    """Test sem_join using DataFrame API."""
    
    @pytest.fixture
    def reviews_data(self):
        return pd.DataFrame({
            'review_id': [1, 2, 3],
            'text': [
                'Amazing product, highly recommend!',
                'Terrible quality, waste of money.',
                'Great value for the price.',
            ]
        })
    
    def test_sem_join_mock(self, reviews_data):
        """Test sem_join with mock implementation."""
        # Create DataFrames and apply semantic join
        left_df = odf.DataFrame(reviews_data)
        right_df = odf.DataFrame(reviews_data)
        
        joined_df = left_df.sem_join(
            right_df,
            join_instruction="Reviews express the same sentiment",
        )
        
        # Execute with mock
        result = joined_df.collect(optimize=True)
        
        # Mock returns cross join
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(reviews_data) ** 2
    
    @skip_llm
    def test_sem_join_llm(self, reviews_data):
        """Test sem_join with real LLM calls."""
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Create DataFrames and apply semantic join
        left_df = odf.DataFrame(reviews_data)
        right_df = odf.DataFrame(reviews_data)
        
        joined_df = left_df.sem_join(
            right_df,
            join_instruction="Reviews express the same sentiment",
            model="gpt-4o-mini",
        )
        
        # Execute with LLM
        executor = Executor(llm_client=client, use_mock=False)
        result = executor.execute(joined_df.plan)
        
        print(f"\n=== Semantic Join LLM Result ===")
        print(f"Left rows: {len(reviews_data)}")
        print(f"Right rows: {len(reviews_data)}")
        print(f"Joined rows: {len(result)}")
        print(result)
        
        assert isinstance(result, pd.DataFrame)
        # Should join reviews with same sentiment (positive: 1,3; negative: 2)
        # Expected: (1,1), (1,3), (3,1), (3,3), (2,2) = 5 rows


class TestSemanticDedup:
    """Test sem_dedup using DataFrame API."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'text': [
                'Great product, highly recommend!',
                'Excellent item, would buy again!',  # Semantically similar to 1
                'Terrible quality, waste of money.',
                'Very disappointed, not worth it.',  # Semantically similar to 3
            ]
        })
    
    def test_sem_dedup_mock(self, sample_data):
        """Test sem_dedup with mock implementation."""
        # Create DataFrame and apply semantic dedup
        df = odf.DataFrame(sample_data)
        df = df.sem_dedup(
            user_instruction="Reviews express the same opinion",
            input_columns=["text"],
        )
        
        # Execute with mock
        result = df.collect(optimize=True)
        
        # Mock uses pandas drop_duplicates
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)
    
    @skip_llm
    def test_sem_dedup_llm(self, sample_data):
        """Test sem_dedup with real LLM calls."""
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Create DataFrame and apply semantic dedup
        df = odf.DataFrame(sample_data)
        df = df.sem_dedup(
            user_instruction="Reviews express the same opinion or sentiment",
            input_columns=["text"],
            model="gpt-4o-mini",
        )
        
        # Execute with LLM
        executor = Executor(llm_client=client, use_mock=False)
        result = executor.execute(df.plan)
        
        print(f"\n=== Semantic Dedup LLM Result ===")
        print(f"Original rows: {len(sample_data)}")
        print(f"After dedup: {len(result)}")
        print(result)
        
        assert isinstance(result, pd.DataFrame)
        # Should remove semantic duplicates (keep 1 or 2, keep 3 or 4)
        assert len(result) <= len(sample_data)


class TestEndToEndPipeline:
    """End-to-end tests using the full DataFrame API."""
    
    @pytest.fixture
    def product_reviews(self):
        return pd.DataFrame({
            'product_id': [1, 1, 2, 2, 3],
            'review': [
                'Love this phone! Great camera and battery life.',
                'Excellent device, worth every penny.',
                'Terrible laptop, crashes constantly.',
                'Worst purchase ever, avoid this product.',
                'Good tablet for the price, decent performance.',
            ],
            'rating': [5, 5, 1, 1, 4]
        })
    
    def test_filter_then_map_mock(self, product_reviews):
        """Test filter + semantic map pipeline with mock."""
        df = odf.DataFrame(product_reviews)
        
        # Filter to high ratings
        df = df.filter(predicate="rating >= 4", columns_used=["rating"])
        
        # Extract sentiment
        df = df.sem_map(
            user_instruction="Extract sentiment (positive/negative/neutral)",
            input_columns=["review"],
            output_column="sentiment",
        )
        
        # Execute with mock
        result = df.collect(optimize=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sentiment' in result.columns
        # Should have 3 rows (rating >= 4)
        assert len(result) == 3
        print(f"\n=== Filter + Map Mock Result ===")
        print(result)
    
    def test_chained_semantic_ops_mock(self, product_reviews):
        """Test chaining multiple semantic operations with mock."""
        df = odf.DataFrame(product_reviews)
        
        # First filter by semantic condition
        df = df.sem_filter(
            user_instruction="The review mentions product quality",
            input_columns=["review"],
        )
        
        # Then extract sentiment
        df = df.sem_map(
            user_instruction="Extract sentiment (positive/negative/neutral)",
            input_columns=["review"],
            output_column="sentiment",
        )
        
        # Execute with mock
        result = df.collect(optimize=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sentiment' in result.columns
        print(f"\n=== Chained Semantic Ops Mock Result ===")
        print(result)
    
    @skip_llm
    def test_filter_then_map_llm(self, product_reviews):
        """Test filter + semantic map pipeline with real LLM."""
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        df = odf.DataFrame(product_reviews)
        
        # Filter to high ratings
        df = df.filter(predicate="rating >= 4", columns_used=["rating"])
        
        # Extract sentiment
        df = df.sem_map(
            user_instruction="Extract sentiment (positive/negative/neutral)",
            input_columns=["review"],
            output_column="sentiment",
            model="gpt-4o-mini",
        )
        
        # Execute with LLM
        executor = Executor(llm_client=client, use_mock=False)
        optimizer = Optimizer()
        optimized_plan = optimizer.optimize(df.plan)
        result = executor.execute(optimized_plan)
        
        print(f"\n=== Filter + Map LLM Result ===")
        print(result)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sentiment' in result.columns
        # High rating reviews should have positive sentiment
        for sentiment in result['sentiment']:
            assert 'positive' in sentiment.lower() or sentiment.lower() == 'positive'
    
    @skip_llm
    def test_sem_join_with_optimization_llm(self, product_reviews):
        """Test semantic join with optimization using real LLM."""
        client = odf.settings.configure(
            oracle_model="gpt-4o-mini",
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
        )
        
        # Create two DataFrames
        df1 = odf.DataFrame(product_reviews[['product_id', 'review']])
        df2 = odf.DataFrame(product_reviews[['product_id', 'review']])
        
        # Semantic join on same sentiment
        joined = df1.sem_join(
            df2,
            join_instruction="Reviews express the same sentiment",
            model="gpt-4o-mini",
        )
        
        # Execute with LLM and optimization
        executor = Executor(llm_client=client, use_mock=False)
        optimizer = Optimizer()
        optimized_plan = optimizer.optimize(joined.plan)
        result = executor.execute(optimized_plan)
        
        print(f"\n=== Semantic Join with Optimization LLM Result ===")
        print(f"Joined rows: {len(result)}")
        print(result)
        
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    # Run with: USE_LLM=true python -m pytest tests/test_semantic_operators.py -v
    pytest.main([__file__, "-v"])

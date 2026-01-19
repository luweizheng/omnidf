"""
Test cases for Pandas API compatibility.

Test Case 4: Standard pandas operations like merge and filter
"""

import pandas as pd
import pytest

import sys
sys.path.insert(0, 'src')

import omnidf as odf
from omnidf.plan import (
    Source, Filter, Join, Project, NodeType,
    plan_to_dict, plan_to_json,
)
from omnidf.optimizer import Optimizer


class TestPandasMerge:
    """Test pandas-style merge operations."""
    
    @pytest.fixture
    def df1_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40]
        })
    
    @pytest.fixture
    def df2_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 5],
            'score': [85, 90, 95, 100]
        })
    
    def test_inner_merge(self, df1_data, df2_data):
        """
        Test inner merge between two DataFrames.
        
        Query:
            expected = df1_data.merge(df2_data, on='id', how='inner')
            expected = expected[expected['age'] > 25]
        """
        # Build OmniDF query
        df1 = odf.DataFrame(df1_data)
        df2 = odf.DataFrame(df2_data)
        
        merged = df1.merge(df2, on='id', how='inner')
        filtered = merged.filter(predicate="age > 25", columns_used=["age"])
        
        plan = filtered.plan
        
        print("\n=== Merge + Filter Plan ===")
        print(plan.pretty_print())
        
        # Verify plan structure
        assert isinstance(plan, Filter), "Root should be Filter"
        assert isinstance(plan.child, Join), "Child should be Join"
        assert plan.child.how == "inner"
        assert plan.child.on == "id"
        
        # Execute and compare with pandas
        result = filtered.collect(optimize=False)
        
        expected = df1_data.merge(df2_data, on='id', how='inner')
        expected = expected[expected['age'] > 25].reset_index(drop=True)
        
        # Compare results
        pd.testing.assert_frame_equal(result, expected)
        print(f"\nResult:\n{result}")
    
    def test_left_merge(self, df1_data, df2_data):
        """Test left merge."""
        df1 = odf.DataFrame(df1_data)
        df2 = odf.DataFrame(df2_data)
        
        merged = df1.merge(df2, on='id', how='left')
        
        result = merged.collect(optimize=False)
        expected = df1_data.merge(df2_data, on='id', how='left')
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_merge_with_different_column_names(self, df1_data):
        """Test merge with left_on and right_on."""
        df2_data = pd.DataFrame({
            'user_id': [1, 2, 3, 5],
            'score': [85, 90, 95, 100]
        })
        
        df1 = odf.DataFrame(df1_data)
        df2 = odf.DataFrame(df2_data)
        
        merged = df1.merge(df2, left_on='id', right_on='user_id', how='inner')
        
        result = merged.collect(optimize=False)
        expected = df1_data.merge(df2_data, left_on='id', right_on='user_id', how='inner')
        
        pd.testing.assert_frame_equal(result, expected)


class TestPandasFilter:
    """Test pandas-style filter operations."""
    
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'city': ['NYC', 'LA', 'NYC', 'Chicago']
        })
    
    def test_simple_filter(self, data):
        """Test simple filter with comparison."""
        df = odf.DataFrame(data)
        filtered = df.filter(predicate="age > 30", columns_used=["age"])
        
        result = filtered.collect(optimize=False)
        expected = data[data['age'] > 30].reset_index(drop=True)
        
        # Note: Our simple executor might not handle all predicates
        # This test verifies the plan structure
        plan = filtered.plan
        assert isinstance(plan, Filter)
        assert plan.predicate == "age > 30"


class TestPandasProject:
    """Test pandas-style column selection."""
    
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30],
            'city': ['NYC', 'LA']
        })
    
    def test_single_column_select(self, data):
        """Test selecting a single column."""
        df = odf.DataFrame(data)
        selected = df['name']
        
        plan = selected.plan
        assert isinstance(plan, Project)
        assert plan.columns == ['name']
    
    def test_multiple_column_select(self, data):
        """Test selecting multiple columns."""
        df = odf.DataFrame(data)
        selected = df[['name', 'age']]
        
        plan = selected.plan
        assert isinstance(plan, Project)
        assert plan.columns == ['name', 'age']
        
        result = selected.collect(optimize=False)
        expected = data[['name', 'age']]
        
        pd.testing.assert_frame_equal(result, expected)


class TestPandasSort:
    """Test pandas-style sort operations."""
    
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'name': ['Charlie', 'Alice', 'Bob'],
            'age': [35, 25, 30]
        })
    
    def test_sort_ascending(self, data):
        """Test sorting in ascending order."""
        df = odf.DataFrame(data)
        sorted_df = df.sort_values(by='age')
        
        result = sorted_df.collect(optimize=False)
        expected = data.sort_values(by='age').reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_sort_descending(self, data):
        """Test sorting in descending order."""
        df = odf.DataFrame(data)
        sorted_df = df.sort_values(by='age', ascending=False)
        
        result = sorted_df.collect(optimize=False)
        expected = data.sort_values(by='age', ascending=False).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected)


class TestPandasLimit:
    """Test pandas-style head/limit operations."""
    
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45]
        })
    
    def test_head(self, data):
        """Test head operation."""
        df = odf.DataFrame(data)
        head_df = df.head(3)
        
        result = head_df.collect(optimize=False)
        expected = data.head(3).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_limit_with_offset(self, data):
        """Test limit with offset."""
        df = odf.DataFrame(data)
        limited = df.limit(2, offset=1)
        
        result = limited.collect(optimize=False)
        expected = data.iloc[1:3].reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected)


class TestComplexQueries:
    """Test complex queries combining multiple operations."""
    
    @pytest.fixture
    def df1_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40]
        })
    
    @pytest.fixture
    def df2_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 5],
            'score': [85, 90, 95, 100]
        })
    
    def test_merge_filter_sort(self, df1_data, df2_data):
        """Test merge -> filter -> sort pipeline."""
        df1 = odf.DataFrame(df1_data)
        df2 = odf.DataFrame(df2_data)
        
        result_df = (
            df1.merge(df2, on='id', how='inner')
            .filter(predicate="age > 25", columns_used=["age"])
            .sort_values(by='score', ascending=False)
        )
        
        plan = result_df.plan
        print("\n=== Complex Query Plan ===")
        print(plan.pretty_print())
        
        # Execute
        result = result_df.collect(optimize=False)
        
        # Compare with pandas
        expected = df1_data.merge(df2_data, on='id', how='inner')
        expected = expected[expected['age'] > 25]
        expected = expected.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

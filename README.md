# OmniDF

**AI-Powered Multi-modal DataFrame with Query Optimization**

OmniDF is a DataFrame library that combines traditional relational operations with semantic (AI-powered) operators. It features lazy evaluation with DAG-based query planning and supports both rule-based and LLM-driven query optimization.

## Features

- **Semantic Operators**: `sem_filter`, `sem_map`, `sem_join`, `sem_dedup` powered by LLMs
- **Lazy Evaluation**: Operations build a DAG; execution happens only on `collect()`
- **Pandas-Compatible API**: Familiar syntax for relational operations
- **Query Optimization**: 
  - Rule-based: FilterPushDown, SemanticMapFusion, SemanticFilterFusion, SemanticJoinDecomposition
  - LLM-driven: Plans serializable to JSON for AI-based optimization

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Example

```python
import pandas as pd
import omnidf as odf

# Configure LLM model (optional - defaults to mock client)
odf.settings.configure(
    oracle_model="gpt-4o-mini",
    oracle_api_key="your-api-key",  # Or set OPENAI_API_KEY env var
)

# Create DataFrame
data = pd.DataFrame({
    'description': ['House with solar panels', 'Cozy apartment', 'Villa with pool'],
    'price': [1200000, 800000, 2000000]
})

df = odf.DataFrame(data)

# Apply semantic filter (LLM-powered)
df = df.semantic_filter(
    user_instruction="The property has eco-friendly features",
    input_columns=["description"]
)

# Apply relational filter
df = df.filter(predicate="price < 1500000", columns_used=["price"])

# View the query plan
print(df.explain())

# Execute with optimization (using collect or execute)
result = df.collect(optimize=True)
# or alternatively:
# result = df.execute(optimize=True)
```

### Use AI Models

```python
import omnidf as odf

# Simple configuration
odf.settings.configure(
    oracle_model="gpt-4o-mini",
    oracle_api_key="your-api-key",
)

# Advanced configuration (different APIs for oracle/proxy)
odf.settings.configure(
    oracle_model="gpt-4o",
    oracle_api_key="cloud-key",
    oracle_api_base="https://api.openai.com/v1",
    proxy_model="internal-llm",
    proxy_api_key="enterprise-key",
    proxy_api_base="https://internal.company.com/v1",
)
```

## Query Optimization Examples

### 1. Filter Push Down

Push relational filters before semantic operators to reduce LLM calls:

```python
# Before: Source -> SemanticFilter -> Filter(price < 1500000)
# After:  Source -> Filter(price < 1500000) -> SemanticFilter

df = odf.DataFrame(estate_data)
df = df.semantic_filter(
    user_instruction="Has solar panels",
    input_columns=["description"]
)
df = df.filter(predicate="price < 1500000", columns_used=["price"])

# The optimizer will push the price filter before the semantic filter
result = df.collect(optimize=True)
```

### 2. Semantic Filter Fusion

Combine consecutive semantic filters into a single LLM call:

```python
# Before: SemanticFilter(director) -> SemanticFilter(genre)
# After:  FusedSemanticFilter(director AND genre)

df = odf.DataFrame(movie_data)
df = df.filter(predicate="Year > 2010", columns_used=["Year"])
df = df.semantic_filter(
    user_instruction="The director is Christopher Nolan",
    input_columns=['Director']
)
df = df.semantic_filter(
    user_instruction="The genre is action movie",
    input_columns=['Genre']
)

result = df.collect(optimize=True)
```

### 3. Semantic Join Decomposition

Convert O(n*m) semantic joins to O(n+m) operations:

```python
# Before: SemanticJoin(left, right, "same sentiment")
# After:  Join(SemanticMap(left, extract), SemanticMap(right, extract))

reviews = odf.DataFrame(reviews_data)
joined = reviews.sem_join(
    reviews,
    join_instruction="Reviews express the same sentiment"
)

result = joined.collect(optimize=True)
```

### DataFrame Methods

**Relational Operations:**
- `filter(predicate, columns_used)` - Filter rows
- `select(*columns)` - Select columns
- `merge(right, on, how)` - Join DataFrames
- `join(right, on, how)` - Alias for merge
- `groupby(by).agg(aggregations)` - Aggregate
- `sort_values(by, ascending)` - Sort
- `head(n)` / `limit(n, offset)` - Limit rows

**Semantic Operations:**
- `semantic_filter(user_instruction, input_columns, model)` - LLM-based filtering
- `sem_filter(user_instruction, input_columns, model)` - Alias for semantic_filter
- `semantic_map(user_instruction, output_column, input_columns, model)` - LLM-based transformation
- `sem_map(user_instruction, output_column, input_columns, model)` - Alias for semantic_map
- `semantic_join(right, join_instruction, model)` - LLM-based join
- `sem_join(right, join_instruction, model)` - Alias for semantic_join
- `semantic_dedup(user_instruction, input_columns, model)` - LLM-based deduplication
- `sem_dedup(user_instruction, input_columns, model)` - Alias for semantic_dedup

**Execution:**
- `collect(optimize=True)` - Execute and return pandas DataFrame
- `execute(optimize=True)` - Alias for collect
- `explain(format="tree"|"json")` - Show query plan
- `to_plan_dict()` - Get plan as dictionary for LLM optimization
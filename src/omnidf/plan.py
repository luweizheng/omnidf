"""
Plan nodes for the OmniDF query DAG.

Design Principles:
1. Immutable nodes - transformations create new nodes
2. Each node has a unique ID for tracking and optimization
3. Nodes are serializable to JSON for LLM-based optimization
4. Clear separation between relational and semantic operators
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


class NodeType(Enum):
    """Types of plan nodes."""
    # Data sources
    SOURCE = "source"
    
    # Relational operators
    FILTER = "filter"
    PROJECT = "project"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    
    # Semantic operators
    SEMANTIC_FILTER = "semantic_filter"
    SEMANTIC_MAP = "semantic_map"
    SEMANTIC_JOIN = "semantic_join"
    SEMANTIC_DEDUP = "semantic_dedup"


@dataclass
class PlanNode(ABC):
    """
    Base class for all plan nodes in the query DAG.
    
    Each node represents an operation in the query plan.
    Nodes are immutable - transformations create new nodes.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return the type of this node."""
        pass
    
    @property
    @abstractmethod
    def children(self) -> List[PlanNode]:
        """Return child nodes (inputs to this operation)."""
        pass
    
    @abstractmethod
    def with_children(self, new_children: List[PlanNode]) -> PlanNode:
        """Create a new node with different children (for plan transformations)."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary for JSON/LLM representation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> PlanNode:
        """Deserialize node from dictionary."""
        pass
    
    def is_relational(self) -> bool:
        """Check if this is a relational (non-semantic) operator."""
        return self.node_type in {
            NodeType.SOURCE, NodeType.FILTER, NodeType.PROJECT,
            NodeType.JOIN, NodeType.AGGREGATE, NodeType.SORT, NodeType.LIMIT
        }
    
    def is_semantic(self) -> bool:
        """Check if this is a semantic operator."""
        return self.node_type in {
            NodeType.SEMANTIC_FILTER, NodeType.SEMANTIC_MAP,
            NodeType.SEMANTIC_JOIN, NodeType.SEMANTIC_DEDUP
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id})"
    
    def pretty_print(self, indent: int = 0) -> str:
        """Pretty print the plan tree."""
        lines = [" " * indent + self._node_str()]
        for child in self.children:
            lines.append(child.pretty_print(indent + 2))
        return "\n".join(lines)
    
    @abstractmethod
    def _node_str(self) -> str:
        """String representation of this node (without children)."""
        pass


@dataclass
class Source(PlanNode):
    """
    Source node - represents input data.
    
    Can hold either:
    - A pandas DataFrame directly
    - A reference to external data (file path, table name, etc.)
    """
    data: Optional[pd.DataFrame] = None
    data_ref: Optional[str] = None  # File path or table reference
    schema: Optional[Dict[str, str]] = None  # Column name -> type
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SOURCE
    
    @property
    def children(self) -> List[PlanNode]:
        return []
    
    def with_children(self, new_children: List[PlanNode]) -> Source:
        assert len(new_children) == 0, "Source node has no children"
        return Source(
            node_id=self.node_id,
            data=self.data,
            data_ref=self.data_ref,
            schema=self.schema
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "data_ref": self.data_ref,
            "schema": self.schema,
            "has_data": self.data is not None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Source:
        return Source(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            data_ref=data.get("data_ref"),
            schema=data.get("schema"),
        )
    
    def _node_str(self) -> str:
        if self.data_ref:
            return f"Source[{self.node_id}](ref={self.data_ref})"
        elif self.data is not None:
            return f"Source[{self.node_id}](rows={len(self.data)}, cols={list(self.data.columns)})"
        return f"Source[{self.node_id}](empty)"


@dataclass
class Filter(PlanNode):
    """
    Relational filter node.
    
    Supports predicate expressions that can be evaluated on pandas DataFrames.
    """
    child: PlanNode = None
    predicate: str = ""  # String representation of the predicate
    predicate_expr: Any = None  # Callable or expression object
    columns_used: List[str] = field(default_factory=list)  # Columns referenced in predicate
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.FILTER
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> Filter:
        assert len(new_children) == 1, "Filter node has exactly one child"
        return Filter(
            node_id=self.node_id,
            child=new_children[0],
            predicate=self.predicate,
            predicate_expr=self.predicate_expr,
            columns_used=self.columns_used,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "predicate": self.predicate,
            "columns_used": self.columns_used,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Filter:
        return Filter(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            predicate=data.get("predicate", ""),
            columns_used=data.get("columns_used", []),
        )
    
    def _node_str(self) -> str:
        return f"Filter[{self.node_id}]({self.predicate})"


@dataclass
class SemanticFilter(PlanNode):
    """
    Semantic filter node - uses LLM to filter rows based on natural language instruction.
    """
    child: PlanNode = None
    user_instruction: str = ""
    input_columns: List[str] = field(default_factory=list)
    model: str = "gpt-4.1"
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SEMANTIC_FILTER
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> SemanticFilter:
        assert len(new_children) == 1, "SemanticFilter node has exactly one child"
        return SemanticFilter(
            node_id=self.node_id,
            child=new_children[0],
            user_instruction=self.user_instruction,
            input_columns=self.input_columns,
            model=self.model,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "user_instruction": self.user_instruction,
            "input_columns": self.input_columns,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> SemanticFilter:
        return SemanticFilter(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            user_instruction=data.get("user_instruction", ""),
            input_columns=data.get("input_columns", []),
            model=data.get("model", "gpt-4.1"),
        )
    
    def _node_str(self) -> str:
        return f"SemanticFilter[{self.node_id}]('{self.user_instruction[:30]}...', cols={self.input_columns})"


@dataclass
class SemanticMap(PlanNode):
    """
    Semantic map node - uses LLM to transform/extract information from rows.
    """
    child: PlanNode = None
    user_instruction: str = ""
    input_columns: List[str] = field(default_factory=list)
    output_column: str = ""
    model: str = "gpt-4.1"
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SEMANTIC_MAP
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> SemanticMap:
        assert len(new_children) == 1, "SemanticMap node has exactly one child"
        return SemanticMap(
            node_id=self.node_id,
            child=new_children[0],
            user_instruction=self.user_instruction,
            input_columns=self.input_columns,
            output_column=self.output_column,
            model=self.model,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "user_instruction": self.user_instruction,
            "input_columns": self.input_columns,
            "output_column": self.output_column,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> SemanticMap:
        return SemanticMap(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            user_instruction=data.get("user_instruction", ""),
            input_columns=data.get("input_columns", []),
            output_column=data.get("output_column", ""),
            model=data.get("model", "gpt-4.1"),
        )
    
    def _node_str(self) -> str:
        return f"SemanticMap[{self.node_id}]('{self.user_instruction[:30]}...' -> {self.output_column})"


@dataclass
class SemanticJoin(PlanNode):
    """
    Semantic join node - uses LLM to determine if rows from two tables should be joined.
    """
    left: PlanNode = None
    right: PlanNode = None
    join_instruction: str = ""
    model: str = "gpt-4.1"
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SEMANTIC_JOIN
    
    @property
    def children(self) -> List[PlanNode]:
        result = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result
    
    def with_children(self, new_children: List[PlanNode]) -> SemanticJoin:
        assert len(new_children) == 2, "SemanticJoin node has exactly two children"
        return SemanticJoin(
            node_id=self.node_id,
            left=new_children[0],
            right=new_children[1],
            join_instruction=self.join_instruction,
            model=self.model,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "join_instruction": self.join_instruction,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> SemanticJoin:
        return SemanticJoin(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            left=children[0] if len(children) > 0 else None,
            right=children[1] if len(children) > 1 else None,
            join_instruction=data.get("join_instruction", ""),
            model=data.get("model", "gpt-4.1"),
        )
    
    def _node_str(self) -> str:
        return f"SemanticJoin[{self.node_id}]('{self.join_instruction[:40]}...')"


@dataclass
class SemanticDedup(PlanNode):
    """
    Semantic deduplication node - uses LLM to identify duplicate rows.
    """
    child: PlanNode = None
    user_instruction: str = ""
    input_columns: List[str] = field(default_factory=list)
    model: str = "gpt-4.1"
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SEMANTIC_DEDUP
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> SemanticDedup:
        assert len(new_children) == 1, "SemanticDedup node has exactly one child"
        return SemanticDedup(
            node_id=self.node_id,
            child=new_children[0],
            user_instruction=self.user_instruction,
            input_columns=self.input_columns,
            model=self.model,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "user_instruction": self.user_instruction,
            "input_columns": self.input_columns,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> SemanticDedup:
        return SemanticDedup(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            user_instruction=data.get("user_instruction", ""),
            input_columns=data.get("input_columns", []),
            model=data.get("model", "gpt-4.1"),
        )
    
    def _node_str(self) -> str:
        return f"SemanticDedup[{self.node_id}]('{self.user_instruction[:30]}...')"


@dataclass
class Join(PlanNode):
    """
    Relational join node.
    """
    left: PlanNode = None
    right: PlanNode = None
    on: Optional[Union[str, List[str]]] = None
    left_on: Optional[Union[str, List[str]]] = None
    right_on: Optional[Union[str, List[str]]] = None
    how: str = "inner"  # inner, left, right, outer, cross
    suffixes: Tuple[str, str] = ("_x", "_y")
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.JOIN
    
    @property
    def children(self) -> List[PlanNode]:
        result = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result
    
    def with_children(self, new_children: List[PlanNode]) -> Join:
        assert len(new_children) == 2, "Join node has exactly two children"
        return Join(
            node_id=self.node_id,
            left=new_children[0],
            right=new_children[1],
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            how=self.how,
            suffixes=self.suffixes,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "on": self.on,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "how": self.how,
            "suffixes": list(self.suffixes),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Join:
        return Join(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            left=children[0] if len(children) > 0 else None,
            right=children[1] if len(children) > 1 else None,
            on=data.get("on"),
            left_on=data.get("left_on"),
            right_on=data.get("right_on"),
            how=data.get("how", "inner"),
            suffixes=tuple(data.get("suffixes", ("_x", "_y"))),
        )
    
    def _node_str(self) -> str:
        join_cols = self.on or f"({self.left_on}, {self.right_on})"
        return f"Join[{self.node_id}]({self.how} on {join_cols})"


@dataclass
class Project(PlanNode):
    """
    Projection node - select specific columns.
    """
    child: PlanNode = None
    columns: List[str] = field(default_factory=list)
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.PROJECT
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> Project:
        assert len(new_children) == 1, "Project node has exactly one child"
        return Project(
            node_id=self.node_id,
            child=new_children[0],
            columns=self.columns,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "columns": self.columns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Project:
        return Project(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            columns=data.get("columns", []),
        )
    
    def _node_str(self) -> str:
        return f"Project[{self.node_id}]({self.columns})"


@dataclass
class Aggregate(PlanNode):
    """
    Aggregation node - group by and aggregate.
    """
    child: PlanNode = None
    group_by: List[str] = field(default_factory=list)
    aggregations: Dict[str, str] = field(default_factory=dict)  # column -> agg_func
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.AGGREGATE
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> Aggregate:
        assert len(new_children) == 1, "Aggregate node has exactly one child"
        return Aggregate(
            node_id=self.node_id,
            child=new_children[0],
            group_by=self.group_by,
            aggregations=self.aggregations,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "group_by": self.group_by,
            "aggregations": self.aggregations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Aggregate:
        return Aggregate(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            group_by=data.get("group_by", []),
            aggregations=data.get("aggregations", {}),
        )
    
    def _node_str(self) -> str:
        return f"Aggregate[{self.node_id}](by={self.group_by}, aggs={self.aggregations})"


@dataclass
class Sort(PlanNode):
    """
    Sort node - order by columns.
    """
    child: PlanNode = None
    by: List[str] = field(default_factory=list)
    ascending: Union[bool, List[bool]] = True
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.SORT
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> Sort:
        assert len(new_children) == 1, "Sort node has exactly one child"
        return Sort(
            node_id=self.node_id,
            child=new_children[0],
            by=self.by,
            ascending=self.ascending,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "by": self.by,
            "ascending": self.ascending,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Sort:
        return Sort(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            by=data.get("by", []),
            ascending=data.get("ascending", True),
        )
    
    def _node_str(self) -> str:
        return f"Sort[{self.node_id}](by={self.by}, asc={self.ascending})"


@dataclass
class Limit(PlanNode):
    """
    Limit node - take first n rows.
    """
    child: PlanNode = None
    n: int = 10
    offset: int = 0
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.LIMIT
    
    @property
    def children(self) -> List[PlanNode]:
        return [self.child] if self.child else []
    
    def with_children(self, new_children: List[PlanNode]) -> Limit:
        assert len(new_children) == 1, "Limit node has exactly one child"
        return Limit(
            node_id=self.node_id,
            child=new_children[0],
            n=self.n,
            offset=self.offset,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "node_id": self.node_id,
            "n": self.n,
            "offset": self.offset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], children: List[PlanNode]) -> Limit:
        return Limit(
            node_id=data.get("node_id", str(uuid.uuid4())[:8]),
            child=children[0] if children else None,
            n=data.get("n", 10),
            offset=data.get("offset", 0),
        )
    
    def _node_str(self) -> str:
        return f"Limit[{self.node_id}](n={self.n}, offset={self.offset})"


# Registry for deserializing nodes from dict
NODE_REGISTRY: Dict[str, type] = {
    NodeType.SOURCE.value: Source,
    NodeType.FILTER.value: Filter,
    NodeType.SEMANTIC_FILTER.value: SemanticFilter,
    NodeType.SEMANTIC_MAP.value: SemanticMap,
    NodeType.SEMANTIC_JOIN.value: SemanticJoin,
    NodeType.SEMANTIC_DEDUP.value: SemanticDedup,
    NodeType.JOIN.value: Join,
    NodeType.PROJECT.value: Project,
    NodeType.AGGREGATE.value: Aggregate,
    NodeType.SORT.value: Sort,
    NodeType.LIMIT.value: Limit,
}


def plan_to_dict(plan: PlanNode) -> Dict[str, Any]:
    """
    Serialize entire plan tree to dictionary.
    
    This format is designed for LLM consumption:
    - Flat list of nodes with references by ID
    - Clear node types and parameters
    - Easy to modify and reconstruct
    """
    nodes = []
    
    def visit(node: PlanNode):
        node_dict = node.to_dict()
        node_dict["children"] = [child.node_id for child in node.children]
        nodes.append(node_dict)
        for child in node.children:
            visit(child)
    
    visit(plan)
    
    return {
        "root_id": plan.node_id,
        "nodes": nodes,
    }


def plan_from_dict(data: Dict[str, Any]) -> PlanNode:
    """
    Deserialize plan tree from dictionary.
    
    Reconstructs the plan from LLM-modified representation.
    """
    nodes_by_id = {node["node_id"]: node for node in data["nodes"]}
    built_nodes: Dict[str, PlanNode] = {}
    
    def build_node(node_id: str) -> PlanNode:
        if node_id in built_nodes:
            return built_nodes[node_id]
        
        node_data = nodes_by_id[node_id]
        node_type = node_data["type"]
        node_class = NODE_REGISTRY[node_type]
        
        # Build children first
        children = [build_node(child_id) for child_id in node_data.get("children", [])]
        
        # Build this node
        node = node_class.from_dict(node_data, children)
        built_nodes[node_id] = node
        return node
    
    return build_node(data["root_id"])


def plan_to_json(plan: PlanNode, indent: int = 2) -> str:
    """Serialize plan to JSON string."""
    return json.dumps(plan_to_dict(plan), indent=indent)


def plan_from_json(json_str: str) -> PlanNode:
    """Deserialize plan from JSON string."""
    return plan_from_dict(json.loads(json_str))

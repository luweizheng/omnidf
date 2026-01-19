"""
Visualization tools for OmniDF query plans.

Provides DAG visualization using graphviz.
"""

from __future__ import annotations

from typing import Optional

from omnidf.plan import PlanNode, NodeType


def plan_to_dot(plan: PlanNode) -> str:
    """
    Convert a query plan to DOT format for graphviz.
    
    Args:
        plan: The query plan to visualize
    
    Returns:
        DOT format string
    """
    lines = ["digraph QueryPlan {"]
    lines.append("  rankdir=BT;")  # Bottom to top
    lines.append("  node [shape=box, style=filled];")
    
    visited = set()
    
    def get_node_color(node: PlanNode) -> str:
        """Get color based on node type."""
        if node.is_semantic():
            return "#FFB6C1"  # Light pink for semantic ops
        elif node.node_type == NodeType.SOURCE:
            return "#90EE90"  # Light green for source
        elif node.node_type == NodeType.FILTER:
            return "#ADD8E6"  # Light blue for filter
        elif node.node_type == NodeType.JOIN:
            return "#DDA0DD"  # Plum for join
        else:
            return "#F0E68C"  # Khaki for others
    
    def get_node_label(node: PlanNode) -> str:
        """Get a concise label for the node."""
        return node._node_str().replace('"', '\\"')
    
    def visit(node: PlanNode):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        
        color = get_node_color(node)
        label = get_node_label(node)
        lines.append(f'  "{node.node_id}" [label="{label}", fillcolor="{color}"];')
        
        for child in node.children:
            lines.append(f'  "{child.node_id}" -> "{node.node_id}";')
            visit(child)
    
    visit(plan)
    lines.append("}")
    
    return "\n".join(lines)


def visualize_plan(
    plan: PlanNode,
    filename: Optional[str] = None,
    format: str = "png",
) -> Optional[str]:
    """
    Visualize a query plan using graphviz.
    
    Args:
        plan: The query plan to visualize
        filename: Output filename (without extension). If None, returns DOT string.
        format: Output format (png, pdf, svg, etc.)
    
    Returns:
        DOT string if filename is None, else the output filepath
    """
    dot_str = plan_to_dot(plan)
    
    if filename is None:
        return dot_str
    
    try:
        import graphviz
        
        graph = graphviz.Source(dot_str)
        output_path = graph.render(filename, format=format, cleanup=True)
        return output_path
    except ImportError:
        print("graphviz package not installed. Returning DOT string.")
        return dot_str


def compare_plans(
    before: PlanNode,
    after: PlanNode,
    filename: Optional[str] = None,
) -> str:
    """
    Create a side-by-side comparison of two plans.
    
    Args:
        before: Original plan
        after: Optimized plan
        filename: Output filename (optional)
    
    Returns:
        DOT string for the comparison
    """
    lines = ["digraph PlanComparison {"]
    lines.append("  rankdir=BT;")
    lines.append("  node [shape=box, style=filled];")
    
    # Create subgraphs for before and after
    lines.append("  subgraph cluster_before {")
    lines.append('    label="Before Optimization";')
    lines.append("    style=dashed;")
    
    visited = set()
    
    def visit_before(node: PlanNode):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        
        label = node._node_str().replace('"', '\\"')
        lines.append(f'    "before_{node.node_id}" [label="{label}", fillcolor="#FFE4E1"];')
        
        for child in node.children:
            lines.append(f'    "before_{child.node_id}" -> "before_{node.node_id}";')
            visit_before(child)
    
    visit_before(before)
    lines.append("  }")
    
    lines.append("  subgraph cluster_after {")
    lines.append('    label="After Optimization";')
    lines.append("    style=dashed;")
    
    visited.clear()
    
    def visit_after(node: PlanNode):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        
        label = node._node_str().replace('"', '\\"')
        lines.append(f'    "after_{node.node_id}" [label="{label}", fillcolor="#E0FFE0"];')
        
        for child in node.children:
            lines.append(f'    "after_{child.node_id}" -> "after_{node.node_id}";')
            visit_after(child)
    
    visit_after(after)
    lines.append("  }")
    
    lines.append("}")
    
    dot_str = "\n".join(lines)
    
    if filename:
        try:
            import graphviz
            graph = graphviz.Source(dot_str)
            graph.render(filename, format="png", cleanup=True)
        except ImportError:
            pass
    
    return dot_str

"""
graph_state.py — Node/Edge dataclasses and GraphState container.

The GraphState is the single source of truth passed between the pipeline
stages and serialised to disk for session persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Node:
    """Represents a single concept node in the knowledge graph.

    Attributes:
        id: Unique identifier (UUID4 hex string).
        label: Short human-readable concept name.
        description: Longer explanation extracted from the source text.
        source_paper: Filename of the PDF this concept came from.
        page_refs: List of page numbers where this concept appears.
        confidence: Float in [0, 1] reflecting extraction confidence.
        embedding: Dense vector representation (list of floats), may be None
            until the deduplication stage populates it.
        x: Horizontal position in the rendered canvas (pixels).
        y: Vertical position in the rendered canvas (pixels).
        vx: Horizontal velocity for physics simulation.
        vy: Vertical velocity for physics simulation.
        pinned: When True the physics engine will not move this node.
        selected: Whether this node is currently highlighted/selected.
        expanded: Whether the zoom_node sub-graph is currently visible.
        metadata: Arbitrary extra key/value pairs for future extensibility.
    """

    label: str
    description: str = ""
    source_paper: str = ""
    page_refs: List[int] = field(default_factory=list)
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    pinned: bool = False
    selected: bool = False
    expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the Node to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "source_paper": self.source_paper,
            "page_refs": self.page_refs,
            "confidence": self.confidence,
            "embedding": self.embedding,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "pinned": self.pinned,
            "selected": self.selected,
            "expanded": self.expanded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Deserialise a Node from a dictionary produced by :meth:`to_dict`."""
        node = cls(
            label=data["label"],
            description=data.get("description", ""),
            source_paper=data.get("source_paper", ""),
            page_refs=data.get("page_refs", []),
            confidence=data.get("confidence", 1.0),
            embedding=data.get("embedding"),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            vx=data.get("vx", 0.0),
            vy=data.get("vy", 0.0),
            vz=data.get("vz", 0.0),
            pinned=data.get("pinned", False),
            selected=data.get("selected", False),
            expanded=data.get("expanded", False),
            metadata=data.get("metadata", {}),
        )
        node.id = data.get("id", node.id)
        return node


@dataclass
class Edge:
    """Represents a directed relationship between two concept nodes.

    Attributes:
        source_id: ID of the source Node.
        target_id: ID of the target Node.
        relation: Short label describing the relationship type (e.g. "causes",
            "extends", "contradicts").
        weight: Numeric strength of the relationship in [0, 1].
        source_paper: PDF filename this edge was derived from.
        confidence: Extraction confidence in [0, 1].
        id: Unique identifier for this edge.
        metadata: Arbitrary extra key/value pairs.
    """

    source_id: str
    target_id: str
    relation: str = "related_to"
    weight: float = 1.0
    source_paper: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the Edge to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "source_paper": self.source_paper,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """Deserialise an Edge from a dictionary produced by :meth:`to_dict`."""
        edge = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=data.get("relation", "related_to"),
            weight=data.get("weight", 1.0),
            source_paper=data.get("source_paper", ""),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
        edge.id = data.get("id", edge.id)
        return edge


class GraphState:
    """Mutable container for the full knowledge-graph state.

    Nodes and edges are stored in insertion-ordered dictionaries keyed by their
    ``id`` fields so that lookups, updates and deletions are O(1).

    Example::

        gs = GraphState()
        n1 = Node(label="Attention Mechanism")
        n2 = Node(label="Transformer")
        gs.add_node(n1)
        gs.add_node(n2)
        gs.add_edge(Edge(source_id=n1.id, target_id=n2.id, relation="part_of"))
    """

    def __init__(self) -> None:
        """Initialise an empty graph."""
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self.session_name: str = ""
        self.source_papers: List[str] = []

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a Node to the graph, overwriting any existing node with the same id.

        Args:
            node: The Node instance to insert.
        """
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges that reference it.

        Args:
            node_id: ID of the node to remove.

        Raises:
            KeyError: If no node with ``node_id`` exists.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph.")
        del self._nodes[node_id]
        dangling = [e_id for e_id, e in self._edges.items()
                    if e.source_id == node_id or e.target_id == node_id]
        for e_id in dangling:
            del self._edges[e_id]

    def get_node(self, node_id: str) -> Optional[Node]:
        """Return the Node with the given id, or None if absent.

        Args:
            node_id: The UUID hex string identifying the node.
        """
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> List[Node]:
        """Return all nodes as an ordered list."""
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """Add an Edge to the graph.

        Args:
            edge: The Edge instance to insert.

        Raises:
            ValueError: If either endpoint node id does not exist in the graph.
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node '{edge.source_id}' does not exist.")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node '{edge.target_id}' does not exist.")
        self._edges[edge.id] = edge

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge by id.

        Args:
            edge_id: ID of the edge to remove.

        Raises:
            KeyError: If no edge with ``edge_id`` exists.
        """
        if edge_id not in self._edges:
            raise KeyError(f"Edge '{edge_id}' not found in graph.")
        del self._edges[edge_id]

    @property
    def edges(self) -> List[Edge]:
        """Return all edges as an ordered list."""
        return list(self._edges.values())

    def neighbors(self, node_id: str) -> List[Tuple[str, Edge]]:
        """Return (neighbour_node_id, edge) pairs for all edges touching *node_id*.

        Args:
            node_id: The node whose neighbourhood to query.
        """
        result = []
        for edge in self._edges.values():
            if edge.source_id == node_id:
                result.append((edge.target_id, edge))
            elif edge.target_id == node_id:
                result.append((edge.source_id, edge))
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire graph to a JSON-compatible dictionary."""
        return {
            "session_name": self.session_name,
            "source_papers": self.source_papers,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphState":
        """Reconstruct a GraphState from a serialised dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.
        """
        gs = cls()
        gs.session_name = data.get("session_name", "")
        gs.source_papers = data.get("source_papers", [])
        for nd in data.get("nodes", []):
            gs.add_node(Node.from_dict(nd))
        for ed in data.get("edges", []):
            edge = Edge.from_dict(ed)
            # Silently skip edges whose endpoints were pruned during export.
            if edge.source_id in gs._nodes and edge.target_id in gs._nodes:
                gs._edges[edge.id] = edge
        return gs

    def __repr__(self) -> str:
        return (f"GraphState(nodes={len(self._nodes)}, "
                f"edges={len(self._edges)}, "
                f"session='{self.session_name}')")

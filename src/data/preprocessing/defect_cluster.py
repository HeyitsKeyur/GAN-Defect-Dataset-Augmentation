"""Defect grouping and spatial cluster formation for multi-defect crop extraction.

Implements:
- Spatial clustering using Euclidean distance between defect centers or bounding-box distance.
- Connected-component cluster formation.
- Individual defect crop mode vs. Multi-defect clustered crop mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from src.data.preprocessing.crop_geometry import BoundingBox


@dataclass
class DefectCluster:
    """A cluster of one or more defects that will share a crop window."""
    cluster_id: int
    boxes: list[BoundingBox] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def bounding_extent(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) enclosing all boxes in the cluster."""
        if not self.boxes:
            return 0.0, 0.0, 0.0, 0.0
        xmins = [b.xmin for b in self.boxes]
        ymins = [b.ymin for b in self.boxes]
        xmaxs = [b.xmax for b in self.boxes]
        ymaxs = [b.ymax for b in self.boxes]
        return min(xmins), min(ymins), max(xmaxs), max(ymaxs)

    @property
    def center(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bounding_extent
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    @property
    def extent_dimensions(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bounding_extent
        return (xmax - xmin), (ymax - ymin)


def box_distance(b1: BoundingBox, b2: BoundingBox) -> float:
    """Calculate Euclidean distance between defect centers."""
    c1_x, c1_y = b1.center
    c2_x, c2_y = b2.center
    return math.hypot(c1_x - c2_x, c1_y - c2_y)


def cluster_defects(
    boxes: Sequence[BoundingBox],
    cluster_distance_threshold: float = 128.0,
    mode: str = "cluster",  # "individual" | "cluster"
) -> list[DefectCluster]:
    """Group bounding boxes into clusters based on spatial proximity.

    If mode == "individual": Each box forms its own singleton DefectCluster.
    If mode == "cluster": Boxes whose centers are within cluster_distance_threshold
                          are merged into a connected-component cluster.
    """
    if not boxes:
        return []

    if mode == "individual" or cluster_distance_threshold <= 0:
        return [DefectCluster(cluster_id=i, boxes=[box]) for i, box in enumerate(boxes)]

    n = len(boxes)
    # Adjacency matrix for connected components
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = box_distance(boxes[i], boxes[j])
            if dist <= cluster_distance_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters: list[DefectCluster] = []
    cluster_id = 0

    for i in range(n):
        if not visited[i]:
            component: list[int] = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            cluster_boxes = [boxes[idx] for idx in sorted(component)]
            clusters.append(DefectCluster(cluster_id=cluster_id, boxes=cluster_boxes))
            cluster_id += 1

    return clusters

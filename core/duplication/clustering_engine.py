"""
Clustering engine for grouping similar code blocks
Supports hierarchical clustering and similarity-based grouping
"""

import asyncio
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

from .types import DuplicationGroup, DuplicateBlock, DuplicationType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Cluster:
    """Represents a cluster of similar code blocks"""
    cluster_id: str
    center_block: DuplicateBlock
    blocks: List[DuplicateBlock]
    similarity_score: float
    cluster_type: DuplicationType
    radius: float
    density: float

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate cluster metrics"""
        if not self.blocks:
            self.density = 0.0
            return

        # Calculate density based on average similarity to center
        similarities = []
        for block in self.blocks:
            if block.block_id != self.center_block.block_id:
                # Simplified similarity calculation
                similarity = self._calculate_block_similarity(block, self.center_block)
                similarities.append(similarity)

        self.density = sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_block_similarity(self, block1: DuplicateBlock, block2: DuplicateBlock) -> float:
        """Calculate similarity between two blocks"""
        # Use content hash similarity as proxy
        if block1.content_hash == block2.content_hash:
            return 1.0

        # Calculate Jaccard similarity of file paths (for structural similarity)
        path_parts1 = set(block1.file_path.split('/'))
        path_parts2 = set(block2.file_path.split('/'))

        if not path_parts1 and not path_parts2:
            return 0.0

        intersection = len(path_parts1.intersection(path_parts2))
        union = len(path_parts1.union(path_parts2))

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary"""
        return {
            "cluster_id": self.cluster_id,
            "center_block": self.center_block.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "similarity_score": self.similarity_score,
            "cluster_type": self.cluster_type.value,
            "radius": self.radius,
            "density": self.density,
            "block_count": len(self.blocks),
            "file_count": len(set(block.file_path for block in self.blocks))
        }


class ClusteringEngine:
    """Clustering engine for duplicate code groups"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize clustering engine with configuration"""
        self.config = config or {}
        self.clustering_algorithm = config.get('algorithm', 'hierarchical')
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.max_cluster_size = config.get('max_cluster_size', 10)
        self.min_cluster_size = config.get('min_cluster_size', 2)
        self.distance_metric = config.get('distance_metric', 'jaccard')
        self.linkage_method = config.get('linkage_method', 'average')

    async def cluster_duplicate_groups(self, groups: List[DuplicationGroup]) -> List[DuplicationGroup]:
        """
        Cluster duplicate groups to reduce redundancy and find patterns

        Args:
            groups: List of duplication groups to cluster

        Returns:
            List of clustered duplication groups
        """
        if not groups:
            logger.info("No groups to cluster")
            return []

        logger.info(f"Starting clustering of {len(groups)} duplicate groups")

        try:
            # Flatten all blocks
            all_blocks = []
            for group in groups:
                all_blocks.extend(group.blocks)

            # Cluster blocks by duplication type
            clustered_groups = []

            # Group by duplication type first
            type_groups = defaultdict(list)
            for block in all_blocks:
                type_groups[block.duplication_type].append(block)

            # Cluster each type group
            for dup_type, blocks in type_groups.items():
                if len(blocks) < self.min_cluster_size:
                    # Create single group for small collections
                    if len(blocks) >= 2:
                        group = DuplicationGroup(
                            group_id=f"{dup_type.value}_simple_{hash(str(blocks)) % 10000:04d}",
                            duplication_type=dup_type,
                            similarity_threshold=self.similarity_threshold,
                            blocks=blocks
                        )
                        clustered_groups.append(group)
                    continue

                # Perform clustering
                clusters = await self._cluster_blocks(blocks, dup_type)

                # Convert clusters to duplication groups
                for cluster in clusters:
                    if len(cluster.blocks) >= self.min_cluster_size:
                        group = DuplicationGroup(
                            group_id=cluster.cluster_id,
                            duplication_type=cluster.cluster_type,
                            similarity_threshold=cluster.similarity_score,
                            blocks=cluster.blocks
                        )
                        clustered_groups.append(group)

            logger.info(f"Clustering completed. Created {len(clustered_groups)} clustered groups")
            return clustered_groups

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return groups  # Return original groups on failure

    async def _cluster_blocks(self, blocks: List[DuplicateBlock], dup_type: DuplicationType) -> List[Cluster]:
        """Cluster blocks using specified algorithm"""
        if self.clustering_algorithm == 'hierarchical':
            return await self._hierarchical_clustering(blocks, dup_type)
        elif self.clustering_algorithm == 'kmeans':
            return await self._kmeans_clustering(blocks, dup_type)
        elif self.clustering_algorithm == 'dbscan':
            return await self._dbscan_clustering(blocks, dup_type)
        else:
            logger.warning(f"Unknown clustering algorithm: {self.clustering_algorithm}")
            return await self._simple_clustering(blocks, dup_type)

    async def _hierarchical_clustering(self, blocks: List[DuplicateBlock], dup_type: DuplicationType) -> List[Cluster]:
        """Hierarchical clustering of blocks"""
        if len(blocks) < 2:
            return []

        # Initialize each block as its own cluster
        clusters = []
        for i, block in enumerate(blocks):
            cluster = Cluster(
                cluster_id=f"cluster_{i}",
                center_block=block,
                blocks=[block],
                similarity_score=1.0,
                cluster_type=dup_type,
                radius=0.0,
                density=1.0
            )
            clusters.append(cluster)

        # Build distance matrix
        distance_matrix = self._build_distance_matrix(blocks)

        # Perform agglomerative clustering
        while len(clusters) > 1:
            # Find most similar clusters
            best_i, best_j, best_similarity = -1, -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(
                        clusters[i], clusters[j], distance_matrix
                    )
                    if similarity > best_similarity:
                        best_i, best_j, best_similarity = i, j, similarity

            # Check if we should merge
            if best_similarity < self.similarity_threshold:
                break

            # Merge clusters
            merged_cluster = self._merge_clusters(
                clusters[best_i], clusters[best_j], best_similarity
            )

            # Remove old clusters and add merged one
            if best_j > best_i:
                del clusters[best_j]
                del clusters[best_i]
            else:
                del clusters[best_i]
                del clusters[best_j]

            clusters.append(merged_cluster)

            # Limit cluster size
            if len(clusters) <= self.max_cluster_size:
                break

        return clusters

    async def _kmeans_clustering(self, blocks: List[DuplicateBlock], dup_type: DuplicationType) -> List[Cluster]:
        """K-means clustering of blocks"""
        if len(blocks) < self.min_cluster_size:
            return []

        # Determine number of clusters (simplified)
        k = min(max(2, len(blocks) // 5), 10)  # 2-10 clusters

        # Initialize centroids (random selection)
        import random
        centroids = random.sample(blocks, min(k, len(blocks)))

        clusters = []
        for i, centroid in enumerate(centroids):
            cluster = Cluster(
                cluster_id=f"kmeans_cluster_{i}",
                center_block=centroid,
                blocks=[centroid],
                similarity_score=1.0,
                cluster_type=dup_type,
                radius=0.0,
                density=1.0
            )
            clusters.append(cluster)

        # K-means iterations
        max_iterations = 10
        for iteration in range(max_iterations):
            # Assign blocks to nearest centroid
            new_clusters = defaultdict(list)

            for block in blocks:
                best_cluster = None
                best_similarity = -1

                for cluster in clusters:
                    similarity = self._calculate_block_similarity(
                        block, cluster.center_block
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster

                if best_cluster:
                    new_clusters[best_cluster.cluster_id].append(block)

            # Update clusters
            changed = False
            for cluster in clusters:
                old_center = cluster.center_block
                cluster_blocks = new_clusters.get(cluster.cluster_id, [])

                if cluster_blocks:
                    # Find new centroid (block with highest average similarity)
                    new_center = self._find_centroid(cluster_blocks)
                    if new_center.block_id != old_center.block_id:
                        changed = True

                    cluster.center_block = new_center
                    cluster.blocks = cluster_blocks
                    cluster._calculate_metrics()

            if not changed:
                break

        # Filter small clusters
        return [c for c in clusters if len(c.blocks) >= self.min_cluster_size]

    async def _dbscan_clustering(self, blocks: List[DuplicateBlock], dup_type: DuplicationType) -> List[Cluster]:
        """DBSCAN clustering of blocks"""
        if len(blocks) < self.min_cluster_size:
            return []

        epsilon = 1.0 - self.similarity_threshold  # Convert similarity to distance
        min_pts = max(2, self.min_cluster_size)

        visited = set()
        clusters = []
        noise = []

        for block in blocks:
            if block.block_id in visited:
                continue

            visited.add(block.block_id)

            # Find neighbors
            neighbors = self._find_neighbors(block, blocks, epsilon)

            if len(neighbors) < min_pts:
                noise.append(block)
                continue

            # Create new cluster
            cluster_id = f"dbscan_cluster_{len(clusters)}"
            cluster_blocks = [block]

            # Expand cluster
            i = 0
            while i < len(neighbors):
                neighbor = neighbors[i]
                neighbor_id = neighbor.block_id

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_neighbors = self._find_neighbors(neighbor, blocks, epsilon)

                    if len(new_neighbors) >= min_pts:
                        neighbors.extend(new_neighbors)

                if neighbor not in cluster_blocks:
                    cluster_blocks.append(neighbor)

                i += 1

            # Create cluster if it's large enough
            if len(cluster_blocks) >= self.min_cluster_size:
                center_block = self._find_centroid(cluster_blocks)
                cluster = Cluster(
                    cluster_id=cluster_id,
                    center_block=center_block,
                    blocks=cluster_blocks,
                    similarity_score=1.0 - epsilon,
                    cluster_type=dup_type,
                    radius=epsilon,
                    density=len(cluster_blocks) / len(blocks)
                )
                clusters.append(cluster)

        return clusters

    async def _simple_clustering(self, blocks: List[DuplicateBlock], dup_type: DuplicationType) -> List[Cluster]:
        """Simple content-based clustering"""
        # Group by content hash
        hash_groups = defaultdict(list)
        for block in blocks:
            hash_groups[block.content_hash].append(block)

        clusters = []
        for i, (content_hash, hash_blocks) in enumerate(hash_groups.items()):
            if len(hash_blocks) >= self.min_cluster_size:
                center_block = self._find_centroid(hash_blocks)
                cluster = Cluster(
                    cluster_id=f"simple_cluster_{i}_{content_hash[:8]}",
                    center_block=center_block,
                    blocks=hash_blocks,
                    similarity_score=1.0,
                    cluster_type=dup_type,
                    radius=0.0,
                    density=1.0
                )
                clusters.append(cluster)

        return clusters

    def _build_distance_matrix(self, blocks: List[DuplicateBlock]) -> List[List[float]]:
        """Build distance matrix between blocks"""
        n = len(blocks)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_block_similarity(blocks[i], blocks[j])
                distance = 1.0 - similarity  # Convert similarity to distance
                matrix[i][j] = distance
                matrix[j][i] = distance

        return matrix

    def _calculate_block_similarity(self, block1: DuplicateBlock, block2: DuplicateBlock) -> float:
        """Calculate similarity between two blocks"""
        # Check for exact match
        if block1.content_hash == block2.content_hash:
            return 1.0

        # Similarity based on file path structure
        path_similarity = self._calculate_path_similarity(block1.file_path, block2.file_path)

        # Similarity based on block size
        size_similarity = 1.0 - abs(block1.line_count - block2.line_count) / max(block1.line_count, block2.line_count, 1)

        # Combined similarity
        return (path_similarity * 0.4 + size_similarity * 0.3 + block1.similarity_score * 0.3)

    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between file paths"""
        parts1 = set(path1.split('/'))
        parts2 = set(path2.split('/'))

        if not parts1 and not parts2:
            return 1.0

        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))

        return intersection / union if union > 0 else 0.0

    def _calculate_cluster_similarity(self, cluster1: Cluster, cluster2: Cluster,
                                     distance_matrix: List[List[float]]) -> float:
        """Calculate similarity between two clusters"""
        if self.linkage_method == 'single':
            # Minimum distance (maximum similarity)
            max_similarity = 0.0
            for block1 in cluster1.blocks:
                for block2 in cluster2.blocks:
                    similarity = self._calculate_block_similarity(block1, block2)
                    max_similarity = max(max_similarity, similarity)
            return max_similarity

        elif self.linkage_method == 'complete':
            # Maximum distance (minimum similarity)
            min_similarity = 1.0
            for block1 in cluster1.blocks:
                for block2 in cluster2.blocks:
                    similarity = self._calculate_block_similarity(block1, block2)
                    min_similarity = min(min_similarity, similarity)
            return min_similarity

        else:  # average (default)
            # Average similarity
            total_similarity = 0.0
            count = 0

            for block1 in cluster1.blocks:
                for block2 in cluster2.blocks:
                    similarity = self._calculate_block_similarity(block1, block2)
                    total_similarity += similarity
                    count += 1

            return total_similarity / count if count > 0 else 0.0

    def _merge_clusters(self, cluster1: Cluster, cluster2: Cluster, similarity: float) -> Cluster:
        """Merge two clusters"""
        all_blocks = cluster1.blocks + cluster2.blocks
        new_center = self._find_centroid(all_blocks)

        return Cluster(
            cluster_id=f"merged_{cluster1.cluster_id}_{cluster2.cluster_id}",
            center_block=new_center,
            blocks=all_blocks,
            similarity_score=similarity,
            cluster_type=cluster1.cluster_type,
            radius=max(cluster1.radius, cluster2.radius),
            density=(cluster1.density + cluster2.density) / 2
        )

    def _find_centroid(self, blocks: List[DuplicateBlock]) -> DuplicateBlock:
        """Find centroid block from a list of blocks"""
        if len(blocks) == 1:
            return blocks[0]

        best_block = blocks[0]
        best_score = -1

        for candidate in blocks:
            total_similarity = 0.0
            for block in blocks:
                if block.block_id != candidate.block_id:
                    similarity = self._calculate_block_similarity(candidate, block)
                    total_similarity += similarity

            avg_similarity = total_similarity / (len(blocks) - 1)
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_block = candidate

        return best_block

    def _find_neighbors(self, block: DuplicateBlock, blocks: List[DuplicateBlock],
                        epsilon: float) -> List[DuplicateBlock]:
        """Find neighbors within epsilon distance"""
        neighbors = []

        for other_block in blocks:
            if other_block.block_id != block.block_id:
                similarity = self._calculate_block_similarity(block, other_block)
                distance = 1.0 - similarity
                if distance <= epsilon:
                    neighbors.append(other_block)

        return neighbors

    def get_clustering_statistics(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """Get clustering statistics"""
        if not clusters:
            return {
                "total_clusters": 0,
                "total_blocks": 0,
                "avg_cluster_size": 0,
                "avg_similarity": 0,
                "avg_density": 0
            }

        total_blocks = sum(len(cluster.blocks) for cluster in clusters)
        avg_cluster_size = total_blocks / len(clusters)
        avg_similarity = sum(cluster.similarity_score for cluster in clusters) / len(clusters)
        avg_density = sum(cluster.density for cluster in clusters) / len(clusters)

        # Size distribution
        size_distribution = defaultdict(int)
        for cluster in clusters:
            size_category = "small" if len(cluster.blocks) <= 3 else "medium" if len(cluster.blocks) <= 7 else "large"
            size_distribution[size_category] += 1

        return {
            "total_clusters": len(clusters),
            "total_blocks": total_blocks,
            "avg_cluster_size": avg_cluster_size,
            "avg_similarity": avg_similarity,
            "avg_density": avg_density,
            "size_distribution": dict(size_distribution),
            "largest_cluster_size": max(len(cluster.blocks) for cluster in clusters),
            "smallest_cluster_size": min(len(cluster.blocks) for cluster in clusters)
        }
"""
Graph construction: Her kategori için ayrı graflar oluşturma
"""
from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


class CategoryGraphBuilder:
    """Her kategori için graph oluşturma"""

    def __init__(self):
        self.graphs = {}  # {category: nx.DiGraph}
        self.category_stats = {}  # {category: stats_dict}

    def build_graph(self,
                   sequences: List[List[str]],
                   category: str,
                   weight_type: str = 'frequency') -> nx.DiGraph:
        """
        Bir kategori için directed graph oluştur
        
        Args:
            sequences: Her text için kelime dizileri listesi
            category: Kategori adı
            weight_type: 'frequency' veya 'count'
        
        Returns:
            networkx.DiGraph: Yönlü graph
        """
        G = nx.DiGraph()
        edge_weights = Counter()
        node_counts = Counter()

        # Count edges for each sequence
        for sequence in sequences:
            if len(sequence) < 2:
                continue

            # Count nodes
            for node in sequence:
                node_counts[node] += 1

            # Create edges (sequential transitions)
            for i in range(len(sequence) - 1):
                source = sequence[i]
                target = sequence[i + 1]
                edge_weights[(source, target)] += 1

        # Add nodes to graph
        for node, count in node_counts.items():
            G.add_node(node, count=count, frequency=count / len(sequences))

        # Add edges to graph
        total_edges = sum(edge_weights.values())
        for (source, target), count in edge_weights.items():
            if weight_type == 'frequency':
                weight = count / total_edges if total_edges > 0 else 0
            else:
                weight = count

            G.add_edge(source, target,
                      weight=weight,
                      count=count,
                      frequency=count / total_edges if total_edges > 0 else 0)

        return G

    def build_category_graphs(self,
                             df: pd.DataFrame,
                             text_column: str,
                             category_column: str,
                             sequence_column: str = 'tokens') -> Dict[str, nx.DiGraph]:
        """
        Her kategori için ayrı graph oluştur
        
        Args:
            df: Processed DataFrame
            text_column: Text column adı
            category_column: Kategori column adı
            sequence_column: Sequence column adı ('tokens', 'bigrams', vb.)
        
        Returns:
            Dict: {category: graph}
        """
        self.graphs = {}
        self.category_stats = {}

        categories = df[category_column].unique()

        for category in categories:
            category_df = df[df[category_column] == category]
            sequences = category_df[sequence_column].tolist()

            # Create graph
            graph = self.build_graph(sequences, category)
            self.graphs[category] = graph

            # Save statistics
            self.category_stats[category] = {
                'num_documents': len(category_df),
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
            }

        return self.graphs

    def get_top_nodes(self, category: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """Bir kategorideki en önemli node'ları getir (degree'e göre)"""
        if category not in self.graphs:
            return []

        graph = self.graphs[category]
        degrees = dict(graph.degree(weight='weight'))
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        return sorted_nodes[:top_n]

    def get_top_edges(self, category: str, top_n: int = 20) -> List[Tuple[Tuple[str, str], float]]:
        """Bir kategorideki en önemli edge'leri getir"""
        if category not in self.graphs:
            return []

        graph = self.graphs[category]
        edges_with_weights = [(u, v, data['weight'])
                             for u, v, data in graph.edges(data=True)]
        sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)

        return [(e[:2], e[2]) for e in sorted_edges[:top_n]]

    def find_paths(self,
                  category: str,
                  source: str = None,
                  target: str = None,
                  max_length: int = 5,
                  min_weight: float = 0.0) -> List[List[str]]:
        """
        Graph'ta path'leri bul (nPath benzeri)
        
        Args:
            category: Kategori adı
            source: Başlangıç node (None ise tüm node'lar)
            target: Hedef node (None ise tüm node'lar)
            max_length: Maksimum path uzunluğu
            min_weight: Minimum edge weight
        
        Returns:
            List of paths: [[node1, node2, ...], ...]
        """
        if category not in self.graphs:
            return []

        graph = self.graphs[category]

        # Weight filtresi uygula
        filtered_graph = graph.copy()
        edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True)
                          if d.get('weight', 0) < min_weight]
        filtered_graph.remove_edges_from(edges_to_remove)

        paths = []

        if source and target:
            # Belirli bir path bul
            try:
                all_paths = list(nx.all_simple_paths(
                    filtered_graph, source, target, cutoff=max_length
                ))
                paths.extend(all_paths)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                pass
        elif source:
            # All paths starting from a specific node
            for target_node in filtered_graph.nodes():
                if target_node != source:
                    try:
                        all_paths = list(nx.all_simple_paths(
                            filtered_graph, source, target_node, cutoff=max_length
                        ))
                        paths.extend(all_paths)
                    except (nx.NodeNotFound, nx.NetworkXNoPath):
                        pass
        else:
            # Find all important paths (from top edges)
            top_edges = self.get_top_edges(category, top_n=50)
            for (u, v), weight in top_edges:
                if weight >= min_weight:
                    paths.append([u, v])

        return paths


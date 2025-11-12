"""
nPath-like pattern finding: Kategoriye özgü pattern'leri bulma
"""
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import pandas as pd
import networkx as nx
from graph_builder import CategoryGraphBuilder


class PatternFinder:
    """Kategoriye özgü pattern'leri bulma (nPath benzeri)"""
    
    def __init__(self, graph_builder: CategoryGraphBuilder):
        self.graph_builder = graph_builder
        self.patterns = {}  # {category: {pattern: score}}
    
    def calculate_path_score(self, 
                            path: List[str], 
                            graph: nx.DiGraph) -> float:
        """
        Bir path'in skorunu hesapla (edge weight'lerinin çarpımı veya toplamı)
        
        Args:
            path: Node dizisi
            graph: Graph
        
        Returns:
            float: Path score
        """
        if len(path) < 2:
            return 0.0
        
        # Edge weight'lerini topla
        total_weight = 0.0
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if graph.has_edge(source, target):
                weight = graph[source][target].get('weight', 0.0)
                total_weight += weight
            else:
                return 0.0  # Path geçersiz
        
        return total_weight
    
    def find_category_specific_patterns(self,
                                        category: str,
                                        min_support: float = 0.01,
                                        max_path_length: int = 5,
                                        top_n: int = 50) -> List[Tuple[List[str], float]]:
        """
        Bir kategoriye özgü pattern'leri bul
        
        Args:
            category: Kategori adı
            min_support: Minimum support threshold
            max_path_length: Maksimum path uzunluğu
            top_n: En iyi N pattern
        
        Returns:
            List of (pattern, score) tuples
        """
        if category not in self.graph_builder.graphs:
            return []
        
        graph = self.graph_builder.graphs[category]
        patterns = []
        
        # Tüm node'lardan başlayarak path'leri bul
        nodes = list(graph.nodes())
        
        # Kısa path'ler (2-3 node)
        for length in range(2, min(max_path_length + 1, 4)):
            for i, start_node in enumerate(nodes):
                # DFS ile path'leri bul
                paths = self._dfs_paths(graph, start_node, length, min_support)
                
                for path in paths:
                    score = self.calculate_path_score(path, graph)
                    if score >= min_support:
                        patterns.append((path, score))
        
        # Uzun path'ler için (top edges'den başla)
        top_edges = self.graph_builder.get_top_edges(category, top_n=100)
        visited_paths = set()
        
        for (u, v), weight in top_edges:
            if weight < min_support:
                continue
            
            # Bu edge'den başlayarak uzat
            extended_paths = self._extend_path(graph, [u, v], max_path_length, min_support)
            
            for path in extended_paths:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    score = self.calculate_path_score(path, graph)
                    if score >= min_support:
                        patterns.append((path, score))
        
        # Skora göre sırala
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns[:top_n]
    
    def _dfs_paths(self, 
                   graph: nx.DiGraph, 
                   start: str, 
                   max_length: int,
                   min_weight: float) -> List[List[str]]:
        """DFS ile path'leri bul"""
        paths = []
        
        def dfs(node: str, current_path: List[str], visited: Set[str]):
            if len(current_path) >= max_length:
                if len(current_path) >= 2:
                    paths.append(current_path.copy())
                return
            
            for neighbor in graph.successors(node):
                if neighbor not in visited and graph.has_edge(node, neighbor):
                    edge_weight = graph[node][neighbor].get('weight', 0.0)
                    if edge_weight >= min_weight:
                        current_path.append(neighbor)
                        visited.add(neighbor)
                        dfs(neighbor, current_path, visited)
                        current_path.pop()
                        visited.remove(neighbor)
        
        dfs(start, [start], {start})
        return paths
    
    def _extend_path(self,
                    graph: nx.DiGraph,
                    path: List[str],
                    max_length: int,
                    min_weight: float) -> List[List[str]]:
        """Mevcut path'i uzat"""
        extended = [path.copy()]
        
        if len(path) >= max_length:
            return extended
        
        last_node = path[-1]
        
        for neighbor in graph.successors(last_node):
            if neighbor not in path:
                edge_weight = graph[last_node][neighbor].get('weight', 0.0)
                if edge_weight >= min_weight:
                    new_path = path + [neighbor]
                    extended.append(new_path)
                    # Recursive olarak daha fazla uzat
                    if len(new_path) < max_length:
                        extended.extend(self._extend_path(graph, new_path, max_length, min_weight))
        
        return extended
    
    def compare_patterns_across_categories(self,
                                           categories: List[str],
                                           min_support: float = 0.01,
                                           max_path_length: int = 5) -> Dict:
        """
        Kategoriler arası pattern karşılaştırması
        
        Returns:
            Dict with:
            - category_specific: {category: [patterns]}
            - common_patterns: Patterns that appear in multiple categories
            - unique_patterns: {category: [unique patterns]}
        """
        all_patterns = {}
        pattern_to_categories = defaultdict(list)
        
        # Her kategori için pattern'leri bul
        for category in categories:
            patterns = self.find_category_specific_patterns(
                category, min_support, max_path_length
            )
            all_patterns[category] = patterns
            
            # Pattern'leri kaydet
            for pattern, score in patterns:
                pattern_key = tuple(pattern)
                pattern_to_categories[pattern_key].append((category, score))
        
        # Ortak pattern'leri bul
        common_patterns = {
            pattern: cats for pattern, cats in pattern_to_categories.items()
            if len(cats) > 1
        }
        
        # Benzersiz pattern'leri bul
        unique_patterns = {}
        for category in categories:
            unique = [
                (list(pattern), score) for pattern, score in all_patterns[category]
                if tuple(pattern) not in common_patterns
            ]
            unique_patterns[category] = unique
        
        return {
            'category_specific': all_patterns,
            'common_patterns': common_patterns,
            'unique_patterns': unique_patterns
        }
    
    def find_discriminative_patterns(self,
                                    categories: List[str],
                                    min_support: float = 0.01,
                                    max_path_length: int = 5,
                                    top_n: int = 20) -> Dict[str, List[Tuple[List[str], float]]]:
        """
        Kategorileri ayırt eden pattern'leri bul
        (Bir kategoride çok yüksek, diğerlerinde düşük)
        
        Returns:
            {category: [(pattern, discriminative_score), ...]}
        """
        discriminative = {}
        
        for category in categories:
            if category not in self.graph_builder.graphs:
                continue
            
            category_patterns = self.find_category_specific_patterns(
                category, min_support, max_path_length, top_n=100
            )
            
            category_graph = self.graph_builder.graphs[category]
            other_graphs = {
                c: g for c, g in self.graph_builder.graphs.items() 
                if c != category
            }
            
            discriminative_patterns = []
            
            for pattern, score in category_patterns:
                # Bu pattern'in diğer kategorilerdeki skorunu hesapla
                other_scores = []
                for other_cat, other_graph in other_graphs.items():
                    other_score = self.calculate_path_score(pattern, other_graph)
                    other_scores.append(other_score)
                
                # Discriminative score: category_score / (avg_other_score + epsilon)
                avg_other = sum(other_scores) / len(other_scores) if other_scores else 0.0
                epsilon = 0.0001
                discriminative_score = score / (avg_other + epsilon)
                
                if discriminative_score > 1.5:  # En az 1.5x daha yüksek
                    discriminative_patterns.append((pattern, discriminative_score))
            
            # En discriminative pattern'leri seç
            discriminative_patterns.sort(key=lambda x: x[1], reverse=True)
            discriminative[category] = discriminative_patterns[:top_n]
        
        return discriminative


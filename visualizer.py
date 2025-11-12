"""
Visualization tools: Graph ve pattern görselleştirme
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


class PatternVisualizer:
    """Pattern ve graph görselleştirme"""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize

    def visualize_graph(self,
                       graph: nx.DiGraph,
                       category: str,
                       top_n_nodes: int = 30,
                       top_n_edges: int = 50,
                       layout: str = 'spring',
                       node_size_factor: float = 1000,
                       edge_width_factor: float = 2.0):
        """
        Bir kategori için graph görselleştir
        
        Args:
            graph: NetworkX graph
            category: Kategori adı
            top_n_nodes: Görüntülenecek en önemli N node
            top_n_edges: Görüntülenecek en önemli N edge
            layout: Graph layout ('spring', 'circular', 'kamada_kawai')
        """
        # Select most important nodes (by degree)
        degrees = dict(graph.degree(weight='weight'))
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n_nodes]
        top_node_set = set([node for node, _ in top_nodes])

        # Create subgraph
        subgraph = graph.subgraph(top_node_set).copy()

        # Select most important edges
        edges_with_weights = [(u, v, data['weight'])
                             for u, v, data in subgraph.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        top_edges = [(u, v) for u, v, w in edges_with_weights[:top_n_edges]]

        # Graph containing only top edges
        final_graph = nx.DiGraph()
        final_graph.add_nodes_from(subgraph.nodes(data=True))
        for u, v in top_edges:
            if subgraph.has_edge(u, v):
                final_graph.add_edge(u, v, **subgraph[u][v])

        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(final_graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(final_graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(final_graph)
        else:
            pos = nx.spring_layout(final_graph)

        # Drawing
        plt.figure(figsize=self.figsize)

        # Draw nodes
        node_sizes = [degrees.get(node, 0) * node_size_factor for node in final_graph.nodes()]
        nx.draw_networkx_nodes(final_graph, pos,
                              node_size=node_sizes,
                              node_color='lightblue',
                              alpha=0.7)

        # Draw edges
        edge_widths = [final_graph[u][v].get('weight', 0) * edge_width_factor
                       for u, v in final_graph.edges()]
        nx.draw_networkx_edges(final_graph, pos,
                              width=edge_widths,
                              alpha=0.5,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->')

        # Draw labels (only for important nodes)
        labels = {node: node if degrees.get(node, 0) > np.percentile(list(degrees.values()), 75)
                 else '' for node in final_graph.nodes()}
        nx.draw_networkx_labels(final_graph, pos, labels, font_size=8)

        plt.title(f'Graph Visualization: {category}\n'
                 f'Nodes: {final_graph.number_of_nodes()}, '
                 f'Edges: {final_graph.number_of_edges()}',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()

    def visualize_patterns_comparison(self,
                                     patterns_by_category: Dict[str, List[Tuple[List[str], float]]],
                                     top_n: int = 10):
        """
        Kategoriler arası pattern karşılaştırması görselleştir
        """
        fig, axes = plt.subplots(len(patterns_by_category), 1,
                                figsize=(15, 5 * len(patterns_by_category)))

        if len(patterns_by_category) == 1:
            axes = [axes]

        for idx, (category, patterns) in enumerate(patterns_by_category.items()):
            top_patterns = patterns[:top_n]

            if not top_patterns:
                continue

            # Convert patterns to strings
            pattern_strings = [' -> '.join(p) for p, _ in top_patterns]
            scores = [s for _, s in top_patterns]

            # Bar plot
            axes[idx].barh(range(len(pattern_strings)), scores, color='steelblue')
            axes[idx].set_yticks(range(len(pattern_strings)))
            axes[idx].set_yticklabels(pattern_strings, fontsize=9)
            axes[idx].set_xlabel('Pattern Score', fontsize=11)
            axes[idx].set_title(f'Top {top_n} Patterns: {category}',
                               fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_discriminative_patterns(self,
                                         discriminative_patterns: Dict[str, List[Tuple[List[str], float]]],
                                         top_n: int = 15):
        """
        Discriminative pattern'leri görselleştir
        """
        fig, axes = plt.subplots(len(discriminative_patterns), 1,
                                figsize=(15, 5 * len(discriminative_patterns)))

        if len(discriminative_patterns) == 1:
            axes = [axes]

        for idx, (category, patterns) in enumerate(discriminative_patterns.items()):
            top_patterns = patterns[:top_n]

            if not top_patterns:
                continue

            pattern_strings = [' -> '.join(p) for p, _ in top_patterns]
            scores = [s for _, s in top_patterns]

            axes[idx].barh(range(len(pattern_strings)), scores, color='coral')
            axes[idx].set_yticks(range(len(pattern_strings)))
            axes[idx].set_yticklabels(pattern_strings, fontsize=9)
            axes[idx].set_xlabel('Discriminative Score', fontsize=11)
            axes[idx].set_title(f'Top {top_n} Discriminative Patterns: {category}',
                               fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
            axes[idx].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
            axes[idx].legend()

        plt.tight_layout()
        return fig

    def visualize_category_statistics(self,
                                     category_stats: Dict[str, Dict]):
        """
        Kategori istatistiklerini görselleştir
        """
        categories = list(category_stats.keys())
        stats_names = ['num_documents', 'num_nodes', 'num_edges', 'avg_degree']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, stat_name in enumerate(stats_names):
            values = [category_stats[cat].get(stat_name, 0) for cat in categories]

            axes[idx].bar(categories, values, color='steelblue', alpha=0.7)
            axes[idx].set_title(stat_name.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_pattern_heatmap(self,
                                  patterns_by_category: Dict[str, List[Tuple[List[str], float]]],
                                  top_n_patterns: int = 20):
        """
        Pattern'lerin kategoriler arası dağılımını heatmap olarak göster
        """
        # Collect all patterns
        all_patterns = set()
        pattern_scores = defaultdict(dict)

        for category, patterns in patterns_by_category.items():
            for pattern, score in patterns[:top_n_patterns]:
                pattern_key = ' -> '.join(pattern)
                all_patterns.add(pattern_key)
                pattern_scores[pattern_key][category] = score

        # Create matrix
        categories = list(patterns_by_category.keys())
        pattern_list = sorted(list(all_patterns))[:top_n_patterns]

        matrix = np.zeros((len(pattern_list), len(categories)))

        for i, pattern in enumerate(pattern_list):
            for j, category in enumerate(categories):
                matrix[i, j] = pattern_scores[pattern].get(category, 0.0)

        # Heatmap
        plt.figure(figsize=(12, max(8, len(pattern_list) * 0.3)))
        sns.heatmap(matrix,
                   xticklabels=categories,
                   yticklabels=pattern_list,
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Pattern Score'})
        plt.title('Pattern Distribution Across Categories', fontsize=14, fontweight='bold')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Pattern', fontsize=12)
        plt.tight_layout()
        return plt.gcf()


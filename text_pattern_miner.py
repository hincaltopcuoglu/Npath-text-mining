"""
Ana Text Pattern Miner: nPath-like pattern mining pipeline
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from utils import TextPreprocessor
from graph_builder import CategoryGraphBuilder
from pattern_finder import PatternFinder
from visualizer import PatternVisualizer
from ngram_analyzer import NGramAnalyzer


class TextPatternMiner:
    """
    Ana pattern mining sınıfı - Teradata Aster nPath benzeri yaklaşım
    """
    
    def __init__(self,
                 data: pd.DataFrame = None,
                 data_path: str = None,
                 text_column: str = 'text',
                 category_column: str = 'category',
                 remove_stopwords: bool = False,
                 lowercase: bool = True,
                 language: str = 'turkish'):
        """
        Args:
            data: DataFrame (veya data_path kullan)
            data_path: CSV dosya yolu
            text_column: Text column adı
            category_column: Kategori column adı
            remove_stopwords: Stopword'leri kaldır
            lowercase: Küçük harfe çevir
            language: Dil ('turkish' veya 'english')
        """
        # Veri yükleme
        if data is not None:
            self.df = data.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either 'data' or 'data_path' must be provided")
        
        self.text_column = text_column
        self.category_column = category_column
        
        # Preprocessor
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            lowercase=lowercase,
            language=language
        )
        
        # Graph builder
        self.graph_builder = CategoryGraphBuilder()
        
        # Pattern finder (graph builder ile initialize edilecek)
        self.pattern_finder = None
        
        # Visualizer
        self.visualizer = PatternVisualizer()
        
        # N-gram analyzer
        self.ngram_analyzer = NGramAnalyzer(
            preprocessor=self.preprocessor,
            remove_stopwords=remove_stopwords,
            lowercase=lowercase,
            language=language
        )
        
        # Processed data
        self.processed_df = None
    
    def preprocess(self):
        """Veriyi işle ve sequence'leri çıkar"""
        print("Preprocessing text data...")
        self.processed_df = self.preprocessor.process_dataframe(
            self.df, self.text_column, self.category_column
        )
        print(f"Processed {len(self.processed_df)} documents")
        return self.processed_df
    
    def build_graphs(self, sequence_column: str = 'tokens'):
        """Her kategori için graph'ları oluştur"""
        if self.processed_df is None:
            self.preprocess()
        
        print("Building category graphs...")
        graphs = self.graph_builder.build_category_graphs(
            self.processed_df,
            self.text_column,
            self.category_column,
            sequence_column=sequence_column
        )
        
        print(f"Built graphs for {len(graphs)} categories")
        for category, stats in self.graph_builder.category_stats.items():
            print(f"  {category}: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Pattern finder'ı initialize et
        self.pattern_finder = PatternFinder(self.graph_builder)
        
        return graphs
    
    def mine_patterns(self,
                     min_support: float = 0.01,
                     max_path_length: int = 5,
                     top_n: int = 50,
                     sequence_column: str = 'tokens') -> Dict[str, List[Tuple[List[str], float]]]:
        """
        Pattern mining - nPath benzeri
        
        Args:
            min_support: Minimum support threshold
            max_path_length: Maksimum path uzunluğu
            top_n: Her kategori için en iyi N pattern
            sequence_column: Hangi sequence column kullanılacak ('tokens', 'bigrams', 'trigrams')
        
        Returns:
            {category: [(pattern, score), ...]}
        """
        if self.graph_builder.graphs == {}:
            self.build_graphs(sequence_column=sequence_column)
        
        print(f"Mining patterns (min_support={min_support}, max_length={max_path_length})...")
        
        categories = self.processed_df[self.category_column].unique()
        patterns_by_category = {}
        
        for category in categories:
            print(f"  Finding patterns for category: {category}")
            patterns = self.pattern_finder.find_category_specific_patterns(
                category,
                min_support=min_support,
                max_path_length=max_path_length,
                top_n=top_n
            )
            patterns_by_category[category] = patterns
            print(f"    Found {len(patterns)} patterns")
        
        return patterns_by_category
    
    def find_discriminative_patterns(self,
                                    min_support: float = 0.01,
                                    max_path_length: int = 5,
                                    top_n: int = 20,
                                    sequence_column: str = 'tokens') -> Dict[str, List[Tuple[List[str], float]]]:
        """
        Kategorileri ayırt eden pattern'leri bul
        
        Returns:
            {category: [(pattern, discriminative_score), ...]}
        """
        if self.graph_builder.graphs == {}:
            self.build_graphs(sequence_column=sequence_column)
        
        print("Finding discriminative patterns...")
        
        categories = list(self.processed_df[self.category_column].unique())
        discriminative = self.pattern_finder.find_discriminative_patterns(
            categories,
            min_support=min_support,
            max_path_length=max_path_length,
            top_n=top_n
        )
        
        return discriminative
    
    def compare_categories(self,
                          min_support: float = 0.01,
                          max_path_length: int = 5) -> Dict:
        """
        Kategoriler arası karşılaştırma
        
        Returns:
            Dict with category_specific, common_patterns, unique_patterns
        """
        if self.graph_builder.graphs == {}:
            self.build_graphs()
        
        categories = list(self.processed_df[self.category_column].unique())
        comparison = self.pattern_finder.compare_patterns_across_categories(
            categories,
            min_support=min_support,
            max_path_length=max_path_length
        )
        
        return comparison
    
    def visualize_graphs(self, 
                        categories: Optional[List[str]] = None,
                        top_n_nodes: int = 30,
                        top_n_edges: int = 50):
        """Kategori graph'larını görselleştir"""
        if self.graph_builder.graphs == {}:
            self.build_graphs()
        
        if categories is None:
            categories = list(self.graph_builder.graphs.keys())
        
        figures = []
        for category in categories:
            if category in self.graph_builder.graphs:
                fig = self.visualizer.visualize_graph(
                    self.graph_builder.graphs[category],
                    category,
                    top_n_nodes=top_n_nodes,
                    top_n_edges=top_n_edges
                )
                figures.append((category, fig))
        
        return figures
    
    def visualize_patterns(self,
                          patterns: Optional[Dict[str, List[Tuple[List[str], float]]]] = None,
                          top_n: int = 10):
        """Pattern'leri görselleştir"""
        if patterns is None:
            patterns = self.mine_patterns()
        
        return self.visualizer.visualize_patterns_comparison(patterns, top_n=top_n)
    
    def visualize_discriminative(self,
                                discriminative_patterns: Optional[Dict] = None,
                                top_n: int = 15):
        """Discriminative pattern'leri görselleştir"""
        if discriminative_patterns is None:
            discriminative_patterns = self.find_discriminative_patterns()
        
        return self.visualizer.visualize_discriminative_patterns(
            discriminative_patterns, top_n=top_n
        )
    
    def visualize_statistics(self):
        """Kategori istatistiklerini görselleştir"""
        if self.graph_builder.category_stats == {}:
            self.build_graphs()
        
        return self.visualizer.visualize_category_statistics(
            self.graph_builder.category_stats
        )
    
    def visualize_heatmap(self,
                         patterns: Optional[Dict[str, List[Tuple[List[str], float]]]] = None,
                         top_n_patterns: int = 20):
        """Pattern heatmap görselleştir"""
        if patterns is None:
            patterns = self.mine_patterns()
        
        return self.visualizer.visualize_pattern_heatmap(patterns, top_n_patterns)
    
    def get_summary(self) -> Dict:
        """Özet istatistikler"""
        if self.processed_df is None:
            self.preprocess()
        
        if self.graph_builder.category_stats == {}:
            self.build_graphs()
        
        summary = {
            'total_documents': len(self.processed_df),
            'categories': list(self.processed_df[self.category_column].unique()),
            'category_stats': self.graph_builder.category_stats
        }
        
        return summary
    
    def export_patterns(self,
                       patterns: Dict[str, List[Tuple[List[str], float]]],
                       output_path: str = 'patterns.csv'):
        """Pattern'leri CSV olarak export et"""
        rows = []
        for category, pattern_list in patterns.items():
            for pattern, score in pattern_list:
                rows.append({
                    'category': category,
                    'pattern': ' -> '.join(pattern),
                    'pattern_length': len(pattern),
                    'score': score
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} patterns to {output_path}")
        return df
    
    def analyze_ngrams(self,
                      n_values: List[int] = [2, 3, 4],
                      min_support: float = 0.0,
                      top_n: int = 50):
        """
        N-gram analizi yap - İlk aşama
        
        Args:
            n_values: Hangi n değerleri için analiz [2, 3, 4] = bigram, trigram, 4-gram
            min_support: Minimum support threshold
            top_n: Her type için en iyi N n-gram
        
        Returns:
            N-gram counts dictionary
        """
        print("=" * 80)
        print("N-GRAM ANALİZİ - İLK AŞAMA")
        print("=" * 80)
        
        # N-gram'ları say
        ngram_counts = self.ngram_analyzer.count_ngrams_by_type(
            self.df,
            self.text_column,
            self.category_column,
            n_values=n_values
        )
        
        # Her n değeri için sonuçları yazdır
        for n in n_values:
            self.ngram_analyzer.print_ngram_summary(
                n=n,
                min_support=min_support,
                top_n=top_n
            )
        
        return ngram_counts
    
    def export_ngrams(self,
                     n_values: List[int] = [2, 3, 4],
                     min_support: float = 0.0,
                     top_n: int = 100):
        """N-gram'ları CSV olarak export et"""
        # Önce analiz yap
        self.analyze_ngrams(n_values, min_support, top_n)
        
        # Her n değeri için export
        exported_files = []
        for n in n_values:
            output_path = f'ngrams_{n}gram.csv'
            df = self.ngram_analyzer.export_ngrams_to_csv(
                n=n,
                output_path=output_path,
                min_support=min_support,
                top_n=top_n
            )
            exported_files.append(output_path)
        
        return exported_files

    def analyze_discriminative_ngrams(self,
                                     n: int,
                                     min_support: float = 0.01,
                                     discriminative_threshold: float = 1.5,
                                     top_n: int = 50):
        """
        İkinci aşama: Discriminative n-gram analizi

        Args:
            n: n-gram boyutu (2, 3, 4)
            min_support: Minimum support threshold
            discriminative_threshold: Discriminative score threshold
            top_n: Her type için en iyi N discriminative n-gram

        Returns:
            Discriminative n-gram dictionary
        """
        if not hasattr(self, 'ngram_analyzer') or not self.ngram_analyzer.ngram_counts:
            print("Önce n-gram analizi yapın: miner.analyze_ngrams()")
            return {}

        print("=" * 80)
        print("İKİNCİ AŞAMA - DISCRIMINATIVE N-GRAM ANALİZİ")
        print("=" * 80)

        self.ngram_analyzer.print_discriminative_analysis(
            n=n,
            min_support=min_support,
            discriminative_threshold=discriminative_threshold,
            top_n=top_n
        )

        discriminative = self.ngram_analyzer.find_discriminative_ngrams(
            n=n,
            min_support=min_support,
            discriminative_threshold=discriminative_threshold,
            top_n=top_n
        )

        return discriminative

    def export_discriminative_ngrams(self,
                                    n: int,
                                    output_path: str = None,
                                    min_support: float = 0.01,
                                    discriminative_threshold: float = 1.5,
                                    top_n: int = 100):
        """Discriminative n-gram'ları export et"""
        if output_path is None:
            output_path = f'discriminative_{n}grams.csv'

        df = self.ngram_analyzer.export_discriminative_ngrams(
            n=n,
            output_path=output_path,
            min_support=min_support,
            discriminative_threshold=discriminative_threshold,
            top_n=top_n
        )

        return output_path


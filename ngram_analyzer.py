"""
N-gram Analizi: Her type için ağır baskın n-gram'ları bulma
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from utils import TextPreprocessor


class NGramAnalyzer:
    """Her type için n-gram frequency analizi"""
    
    def __init__(self,
                 preprocessor: TextPreprocessor = None,
                 remove_stopwords: bool = False,
                 lowercase: bool = True,
                 language: str = 'turkish'):
        """
        Args:
            preprocessor: TextPreprocessor instance (None ise yeni oluşturulur)
            remove_stopwords: Stopword'leri kaldır
            lowercase: Küçük harfe çevir
            language: Dil
        """
        if preprocessor is None:
            self.preprocessor = TextPreprocessor(
                remove_stopwords=remove_stopwords,
                lowercase=lowercase,
                language=language
            )
        else:
            self.preprocessor = preprocessor
        
        self.ngram_counts = {}  # {type: {n: Counter}}
        self.type_doc_counts = {}  # {type: document_count}
    
    def extract_ngrams_from_text(self, text: str, n: int) -> List[Tuple[str, ...]]:
        """Bir text'ten n-gram'ları çıkar"""
        return self.preprocessor.extract_sequences(text, n=n)
    
    def count_ngrams_by_type(self,
                            df: pd.DataFrame,
                            text_column: str,
                            type_column: str,
                            n_values: List[int] = [2, 3, 4]) -> Dict[str, Dict[int, Counter]]:
        """
        Her type için n-gram'ları say
        
        Args:
            df: DataFrame
            text_column: Text column adı
            type_column: Type/category column adı
            n_values: Hangi n değerleri için analiz yapılacak [2, 3, 4] = bigram, trigram, 4-gram
        
        Returns:
            {type: {n: Counter(ngram: count)}}
        """
        self.ngram_counts = {}
        self.type_doc_counts = {}
        
        types = df[type_column].unique()
        
        for type_name in types:
            type_df = df[df[type_column] == type_name]
            self.type_doc_counts[type_name] = len(type_df)
            
            # Her n değeri için
            type_ngrams = {}
            for n in n_values:
                ngram_counter = Counter()
                
                for text in type_df[text_column]:
                    ngrams = self.extract_ngrams_from_text(text, n)
                    for ngram in ngrams:
                        ngram_counter[ngram] += 1
                
                type_ngrams[n] = ngram_counter
            
            self.ngram_counts[type_name] = type_ngrams
        
        return self.ngram_counts
    
    def calculate_support(self, ngram_count: int, total_docs: int) -> float:
        """
        Support hesapla: Bu n-gram'ın kaç dokümanda göründüğü / toplam doküman
        
        Not: Burada ngram_count aslında kaç kez göründüğü, 
        ama support genelde "kaç dokümanda göründüğü" olarak hesaplanır.
        Bu yüzden document-level support için ayrı bir metod gerekebilir.
        """
        return ngram_count / total_docs if total_docs > 0 else 0.0
    
    def get_top_ngrams(self,
                      type_name: str,
                      n: int,
                      min_support: float = 0.0,
                      top_n: int = 50) -> List[Tuple[Tuple[str, ...], int, float]]:
        """
        Bir type için en sık görülen n-gram'ları getir
        
        Args:
            type_name: Type adı
            n: n-gram boyutu (2, 3, 4, vb.)
            min_support: Minimum support threshold
            top_n: En iyi N n-gram
        
        Returns:
            [(ngram_tuple, count, support), ...]
        """
        if type_name not in self.ngram_counts:
            return []
        
        if n not in self.ngram_counts[type_name]:
            return []
        
        ngram_counter = self.ngram_counts[type_name][n]
        total_docs = self.type_doc_counts.get(type_name, 1)
        
        results = []
        for ngram, count in ngram_counter.most_common():
            support = self.calculate_support(count, total_docs)
            if support >= min_support:
                results.append((ngram, count, support))
        
        return results[:top_n]
    
    def get_all_top_ngrams(self,
                          n: int,
                          min_support: float = 0.0,
                          top_n: int = 50) -> Dict[str, List[Tuple[Tuple[str, ...], int, float]]]:
        """
        Tüm type'lar için en sık görülen n-gram'ları getir
        
        Returns:
            {type: [(ngram, count, support), ...]}
        """
        results = {}
        for type_name in self.ngram_counts.keys():
            results[type_name] = self.get_top_ngrams(
                type_name, n, min_support, top_n
            )
        return results
    
    def print_ngram_summary(self,
                           n: int,
                           min_support: float = 0.0,
                           top_n: int = 20):
        """N-gram özetini yazdır"""
        print(f"\n{'='*80}")
        print(f"{n}-GRAM ANALİZİ (Min Support: {min_support:.2%}, Top {top_n})")
        print(f"{'='*80}")
        
        all_top = self.get_all_top_ngrams(n, min_support, top_n)
        
        for type_name, ngrams in all_top.items():
            print(f"\n📊 TYPE: {type_name}")
            print(f"   Toplam Doküman: {self.type_doc_counts.get(type_name, 0)}")
            print(f"   Bulunan {n}-gram sayısı: {len(ngrams)}")
            print(f"\n   Top {min(top_n, len(ngrams))} {n}-gram:")
            print(f"   {'-'*76}")
            
            for i, (ngram, count, support) in enumerate(ngrams[:top_n], 1):
                ngram_str = ' '.join(ngram)
                print(f"   {i:2d}. [{support:6.2%}] (count: {count:4d})  {ngram_str}")
    
    def export_ngrams_to_csv(self,
                           n: int,
                           output_path: str = None,
                           min_support: float = 0.0,
                           top_n: int = 100):
        """N-gram'ları CSV olarak export et"""
        if output_path is None:
            output_path = f'ngrams_{n}gram.csv'
        
        rows = []
        all_top = self.get_all_top_ngrams(n, min_support, top_n)
        
        for type_name, ngrams in all_top.items():
            for ngram, count, support in ngrams:
                rows.append({
                    'type': type_name,
                    'ngram': ' '.join(ngram),
                    'ngram_length': n,
                    'count': count,
                    'support': support,
                    'total_docs': self.type_doc_counts.get(type_name, 0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ {len(df)} n-gram exported to {output_path}")
        return df
    
    def calculate_discriminative_score(self, ngram: Tuple[str, ...], type_name: str) -> float:
        """
        Bir n-gram'ın discriminative score'unu hesapla

        Score = Support(type_i) / Average_Support(other_types)
        Yüksek score = Bu type'a çok özgü
        """
        if type_name not in self.ngram_counts:
            return 0.0

        # Bu type'taki support
        type_support = 0.0
        for n, counter in self.ngram_counts[type_name].items():
            if ngram in counter:
                type_support = self.calculate_support(counter[ngram], self.type_doc_counts[type_name])
                break

        # Diğer type'lardaki ortalama support
        other_supports = []
        for other_type in self.ngram_counts.keys():
            if other_type != type_name:
                support = 0.0
                for n, counter in self.ngram_counts[other_type].items():
                    if ngram in counter:
                        support = self.calculate_support(counter[ngram], self.type_doc_counts[other_type])
                        break
                other_supports.append(support)

        if not other_supports:
            return 0.0

        avg_other_support = sum(other_supports) / len(other_supports)
        epsilon = 0.0001  # Sıfır bölme önleme

        return type_support / (avg_other_support + epsilon)

    def find_discriminative_ngrams(self,
                                  n: int,
                                  min_support: float = 0.01,
                                  discriminative_threshold: float = 1.5,
                                  top_n: int = 50) -> Dict[str, List[Tuple[Tuple[str, ...], float, float]]]:
        """
        Her type için discriminative n-gram'ları bul

        Returns:
            {type: [(ngram, discriminative_score, support), ...]}
        """
        discriminative_ngrams = {}

        for type_name in self.ngram_counts.keys():
            type_discriminatives = []

            # Bu type'taki tüm n-gram'ları kontrol et
            if n in self.ngram_counts[type_name]:
                counter = self.ngram_counts[type_name][n]

                for ngram, count in counter.items():
                    support = self.calculate_support(count, self.type_doc_counts[type_name])

                    if support >= min_support:
                        disc_score = self.calculate_discriminative_score(ngram, type_name)

                        if disc_score >= discriminative_threshold:
                            type_discriminatives.append((ngram, disc_score, support))

            # Score'a göre sırala
            type_discriminatives.sort(key=lambda x: x[1], reverse=True)
            discriminative_ngrams[type_name] = type_discriminatives[:top_n]

        return discriminative_ngrams

    def print_discriminative_analysis(self,
                                     n: int,
                                     min_support: float = 0.01,
                                     discriminative_threshold: float = 1.5,
                                     top_n: int = 20):
        """Discriminative n-gram analizi sonuçlarını yazdır"""
        print(f"\n{'='*80}")
        print(f"DISCRIMINATIVE {n}-GRAM ANALİZİ")
        print(f"{'='*80}")
        print(f"Min Support: {min_support:.2%} | Discriminative Threshold: {discriminative_threshold}")
        print(f"Score = Support(type) / Avg_Support(other_types)")
        print(f"Score ≥ {discriminative_threshold} = Bu type'a çok özgü")

        discriminative = self.find_discriminative_ngrams(n, min_support, discriminative_threshold, top_n)

        for type_name, ngrams in discriminative.items():
            if not ngrams:
                continue

            print(f"\n📊 TYPE: {type_name} ({len(ngrams)} discriminative {n}-gram)")
            print(f"   {'-'*76}")

            for i, (ngram, disc_score, support) in enumerate(ngrams[:top_n], 1):
                ngram_str = ' '.join(ngram)
                print(f"   {i:2d}. [{disc_score:.1f}x] (support: {support:.2%})  {ngram_str}")

    def export_discriminative_ngrams(self,
                                    n: int,
                                    output_path: str = None,
                                    min_support: float = 0.01,
                                    discriminative_threshold: float = 1.5,
                                    top_n: int = 100):
        """Discriminative n-gram'ları CSV olarak export et"""
        if output_path is None:
            output_path = f'discriminative_{n}grams.csv'

        discriminative = self.find_discriminative_ngrams(n, min_support, discriminative_threshold, top_n)

        rows = []
        for type_name, ngrams in discriminative.items():
            for ngram, disc_score, support in ngrams:
                rows.append({
                    'type': type_name,
                    'ngram': ' '.join(ngram),
                    'ngram_length': n,
                    'discriminative_score': disc_score,
                    'support': support,
                    'total_docs': self.type_doc_counts.get(type_name, 0)
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ {len(df)} discriminative n-gram exported to {output_path}")
        return df

    def get_ngram_statistics(self) -> Dict:
        """Genel istatistikler"""
        stats = {
            'types': list(self.ngram_counts.keys()),
            'type_doc_counts': self.type_doc_counts.copy(),
            'ngram_counts_by_type': {}
        }

        for type_name, ngrams_by_n in self.ngram_counts.items():
            stats['ngram_counts_by_type'][type_name] = {
                n: len(counter) for n, counter in ngrams_by_n.items()
            }

        return stats


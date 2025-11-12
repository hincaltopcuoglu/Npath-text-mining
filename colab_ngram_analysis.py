"""
N-Gram Analysis for Opinions Dataset - Optimized for Google Colab
Creates bigrams, trigrams, 4-grams and performs discriminative analysis
for text classification (text -> type prediction)
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (with error handling for Colab)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")
    print("This is normal in some environments. Make sure NLTK data is available.")

class ColabNgramAnalyzer:
    """
    Efficient N-Gram Analysis for Google Colab
    Optimized for large datasets and memory efficiency
    """

    def __init__(self, data_path='data/raw/opinions.csv', text_col='text', target_col='type'):
        self.data_path = data_path
        self.text_col = text_col
        self.target_col = target_col
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Results storage
        self.ngram_counts = {}
        self.discriminative_scores = {}
        self.class_distributions = {}

    def load_and_clean_data(self):
        """Load and clean the opinions dataset"""
        print("🔄 Loading and cleaning data...")

        # Try different parsing approaches
        try:
            # First attempt: Standard pandas read
            self.df = pd.read_csv(self.data_path,
                                 sep=',',
                                 quotechar='"',
                                 escapechar='\\',
                                 on_bad_lines='skip',
                                 engine='python')
        except Exception as e1:
            print(f"Standard parsing failed: {e1}")
            try:
                # Fallback: Manual parsing
                rows = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line_num > 100000:  # Limit rows for memory
                            break

                        line = line.strip()
                        while line.endswith(';'):
                            line = line[:-1]

                        # Split carefully
                        parts = line.split(',')
                        if len(parts) >= 4:  # At least id, topic_id, text, type
                            # Reconstruct text field if it was split
                            if len(parts) > 4:
                                text_parts = parts[2:-2]  # Everything between topic_id and last fields
                                text = ','.join(text_parts)
                                parts = parts[:2] + [text] + parts[-2:]

                            if len(parts) >= 4:
                                rows.append(parts[:4])  # Take first 4 columns

                self.df = pd.DataFrame(rows, columns=['id', 'topic_id', 'text', 'type'])
                print("Used manual parsing fallback")

            except Exception as e2:
                raise Exception(f"Both parsing methods failed: {e1}, {e2}")

        # Clean column names
        self.df.columns = self.df.columns.str.replace(';;;;;;', '').str.strip()

        # Clean data
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[self.text_col, self.target_col])
        if self.text_col in self.df.columns:
            self.df = self.df[self.df[self.text_col].str.len() > 10]  # Remove very short texts
        if self.target_col in self.df.columns:
            self.df = self.df[self.df[self.target_col].str.len() > 0]  # Remove empty types

        print(f"✅ Loaded {len(self.df)} rows from {initial_rows} total")
        print(f"📊 Columns: {list(self.df.columns)}")

        if self.target_col in self.df.columns:
            print(f"🎯 Target classes: {self.df[self.target_col].nunique()}")

            # Show class distribution
            print("\n📈 Class Distribution:")
            class_counts = self.df[self.target_col].value_counts()
            for cls, count in class_counts.head(10).items():
                print(f"  {str(cls)[:30]}...: {count}")
        else:
            print(f"⚠️  Warning: {self.target_col} column not found")

        return self.df

    def preprocess_text(self, text):
        """Clean and preprocess text for n-gram analysis"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and len(token) > 1]

        return tokens

    def generate_ngrams_batch(self, tokens_list, n):
        """Generate n-grams from a batch of tokenized texts"""
        ngram_freq = Counter()

        for tokens in tokens_list:
            if len(tokens) >= n:
                ngram_list = list(ngrams(tokens, n))
                ngram_freq.update(ngram_list)

        return ngram_freq

    def analyze_ngrams_by_class(self, n_values=[2, 3, 4], batch_size=1000, min_freq=5):
        """
        Analyze n-grams for each class efficiently
        Uses batching to handle large datasets
        """
        print(f"\n🔄 Analyzing {n_values} n-grams by class...")

        # Group by class
        class_groups = self.df.groupby(self.target_col)

        for class_name in tqdm(class_groups.groups.keys(), desc="Classes"):
            class_data = class_groups.get_group(class_name)
            self.class_distributions[class_name] = len(class_data)

            # Process in batches
            all_tokens = []
            for i in range(0, len(class_data), batch_size):
                batch = class_data.iloc[i:i+batch_size]
                batch_tokens = [self.preprocess_text(text) for text in batch[self.text_col]]
                all_tokens.extend(batch_tokens)

            # Generate n-grams for this class
            for n in n_values:
                if n not in self.ngram_counts:
                    self.ngram_counts[n] = {}

                if class_name not in self.ngram_counts[n]:
                    self.ngram_counts[n][class_name] = Counter()

                # Generate n-grams
                ngram_freq = self.generate_ngrams_batch(all_tokens, n)

                # Filter by minimum frequency
                filtered_ngrams = {ngram: freq for ngram, freq in ngram_freq.items()
                                 if freq >= min_freq}

                self.ngram_counts[n][class_name] = filtered_ngrams

                print(f"  Class '{class_name[:20]}...': {len(filtered_ngrams)} {n}-grams (≥{min_freq} freq)")

    def calculate_discriminative_scores(self, n, min_support=10, top_k=1000):
        """
        Calculate discriminative scores for n-grams
        Higher score = more discriminative for that class
        """
        print(f"\n🔄 Calculating discriminative scores for {n}-grams...")

        if n not in self.ngram_counts:
            print(f"❌ No {n}-gram data found. Run analyze_ngrams_by_class first.")
            return {}

        class_ngrams = self.ngram_counts[n]
        discriminative_scores = defaultdict(dict)

        # Get all unique n-grams across all classes
        all_ngrams = set()
        for class_name, ngram_counter in class_ngrams.items():
            all_ngrams.update(ngram_counter.keys())

        print(f"📊 Analyzing {len(all_ngrams)} unique {n}-grams...")

        for ngram in tqdm(all_ngrams, desc="N-grams"):
            ngram_scores = {}

            # Calculate frequency in each class
            total_freq_all_classes = 0
            for class_name, ngram_counter in class_ngrams.items():
                freq_in_class = ngram_counter.get(ngram, 0)
                total_freq_all_classes += freq_in_class

                # Only consider if minimum support met
                if freq_in_class >= min_support:
                    # Calculate class-specific score
                    class_total_docs = self.class_distributions[class_name]
                    class_freq_ratio = freq_in_class / class_total_docs

                    # Calculate how much more frequent in this class vs others
                    other_classes_freq = total_freq_all_classes - freq_in_class
                    other_classes_total_docs = sum(self.class_distributions.values()) - class_total_docs
                    other_freq_ratio = other_classes_freq / other_classes_total_docs if other_classes_total_docs > 0 else 0

                    # Discriminative score (ratio of ratios)
                    if other_freq_ratio > 0:
                        score = class_freq_ratio / other_freq_ratio
                        ngram_scores[class_name] = score

            # Store scores for this ngram
            for class_name, score in ngram_scores.items():
                discriminative_scores[class_name][ngram] = score

        # Sort and keep top-k per class
        self.discriminative_scores[n] = {}
        for class_name in discriminative_scores:
            sorted_ngrams = sorted(discriminative_scores[class_name].items(),
                                 key=lambda x: x[1], reverse=True)
            self.discriminative_scores[n][class_name] = dict(sorted_ngrams[:top_k])

        return self.discriminative_scores[n]

    def export_results(self, output_dir='colab_results'):
        """Export n-gram and discriminative analysis results"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 Exporting results to {output_dir}/...")

        # Export n-gram counts
        for n in self.ngram_counts:
            ngram_df = []
            for class_name, ngram_counter in self.ngram_counts[n].items():
                for ngram, freq in ngram_counter.items():
                    ngram_text = ' '.join(ngram)
                    ngram_df.append({
                        'class': class_name,
                        'ngram': ngram_text,
                        'frequency': freq
                    })

            if ngram_df:
                df = pd.DataFrame(ngram_df)
                df.to_csv(f'{output_dir}/{n}gram_counts.csv', index=False)
                print(f"  ✅ {n}-gram counts: {len(df)} entries")

        # Export discriminative scores
        for n in self.discriminative_scores:
            disc_df = []
            for class_name, ngram_scores in self.discriminative_scores[n].items():
                for ngram, score in ngram_scores.items():
                    ngram_text = ' '.join(ngram)
                    disc_df.append({
                        'class': class_name,
                        'ngram': ngram_text,
                        'discriminative_score': score
                    })

            if disc_df:
                df = pd.DataFrame(disc_df)
                df.to_csv(f'{output_dir}/{n}gram_discriminative.csv', index=False)
                print(f"  ✅ {n}-gram discriminative: {len(df)} entries")

    def visualize_top_discriminative(self, n=2, top_k=20, output_dir='colab_results'):
        """Create visualizations for top discriminative n-grams"""
        if n not in self.discriminative_scores:
            print(f"❌ No discriminative scores for {n}-grams")
            return

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Create subplots for each class
        classes = list(self.discriminative_scores[n].keys())
        n_classes = len(classes)

        if n_classes == 0:
            print("❌ No classes found")
            return

        fig, axes = plt.subplots(n_classes, 1, figsize=(15, 6*n_classes))
        if n_classes == 1:
            axes = [axes]

        for i, class_name in enumerate(classes):
            top_ngrams = list(self.discriminative_scores[n][class_name].items())[:top_k]

            if top_ngrams:
                ngrams_text = [' '.join(ngram) for ngram, score in top_ngrams]
                scores = [score for ngram, score in top_ngrams]

                axes[i].barh(range(len(ngrams_text)), scores)
                axes[i].set_yticks(range(len(ngrams_text)))
                axes[i].set_yticklabels(ngrams_text, fontsize=8)
                axes[i].set_xlabel('Discriminative Score')
                axes[i].set_title(f'Top {top_k} Discriminative {n}-grams for "{class_name}"', fontsize=12)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_{n}gram_discriminative.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Visualization saved: top_{n}gram_discriminative.png")

    def run_complete_analysis(self,
                            n_values=[2, 3, 4],
                            min_freq=5,
                            min_support=10,
                            top_k=500,
                            batch_size=1000):
        """
        Run complete n-gram analysis pipeline
        Optimized for Google Colab execution
        """
        print("🚀 Starting Complete N-Gram Analysis for Opinions Dataset")
        print("=" * 60)

        # Step 1: Load and clean data
        self.load_and_clean_data()

        # Step 2: Analyze n-grams by class
        self.analyze_ngrams_by_class(n_values=n_values,
                                   batch_size=batch_size,
                                   min_freq=min_freq)

        # Step 3: Calculate discriminative scores
        for n in n_values:
            self.calculate_discriminative_scores(n=n,
                                               min_support=min_support,
                                               top_k=top_k)

        # Step 4: Export results
        self.export_results()

        # Step 5: Create visualizations
        for n in n_values:
            self.visualize_top_discriminative(n=n, top_k=20)

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)

        # Summary
        total_ngrams = sum(len(class_data) for n_data in self.ngram_counts.values()
                          for class_data in n_data.values())
        total_discriminative = sum(len(class_data) for n_data in self.discriminative_scores.values()
                                  for class_data in n_data.values())

        print("📊 Summary:")
        print(f"  Classes analyzed: {len(self.class_distributions)}")
        print(f"  Total documents: {sum(self.class_distributions.values())}")
        print(f"  Total n-grams generated: {total_ngrams}")
        print(f"  Total discriminative n-grams: {total_discriminative}")
        print(f"  Results saved to: colab_results/")


if __name__ == '__main__':
    # Initialize analyzer
    analyzer = ColabNgramAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis(
        n_values=[2, 3, 4],      # bigrams, trigrams, 4-grams
        min_freq=5,              # minimum frequency for n-grams
        min_support=10,          # minimum support for discriminative analysis
        top_k=500,               # top k discriminative n-grams per class
        batch_size=1000          # processing batch size for memory efficiency
    )

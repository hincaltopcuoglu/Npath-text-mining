"""
Pattern Ranking System for NPath-like Text Analysis
Implements multiple scoring metrics for discriminative n-gram ranking
"""
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PatternRanker:
    """
    Advanced pattern ranking system with multiple scoring metrics
    Similar to nPath pattern discovery but for text classification
    """

    def __init__(self, results_dir='colab_results'):
        self.results_dir = results_dir
        self.pattern_data = {}
        self.class_stats = {}
        self.global_stats = {}
        self.ranked_patterns = {}

        # Scoring parameters
        self.min_support = 5
        self.min_confidence = 0.1
        self.lift_threshold = 1.5

    def load_pattern_data(self):
        """Load pattern data from colab_results directory"""
        print("📥 Loading pattern data from results...")

        n_gram_types = [2, 3, 4]  # bigrams, trigrams, 4-grams

        for n in n_gram_types:
            # Load counts data
            counts_file = f'{self.results_dir}/{n}gram_counts.csv'
            disc_file = f'{self.results_dir}/{n}gram_discriminative.csv'

            try:
                if pd.io.common.file_exists(counts_file):
                    counts_df = pd.read_csv(counts_file)
                    print(f"  ✅ Loaded {n}-gram counts: {len(counts_df)} entries")

                if pd.io.common.file_exists(disc_file):
                    disc_df = pd.read_csv(disc_file)
                    print(f"  ✅ Loaded {n}-gram discriminative: {len(disc_df)} entries")

                    # Store in pattern_data
                    self.pattern_data[n] = {
                        'counts': counts_df if 'counts_df' in locals() else None,
                        'discriminative': disc_df
                    }

            except Exception as e:
                print(f"  ❌ Error loading {n}-gram data: {e}")

        print(f"📊 Loaded pattern data for {len(self.pattern_data)} n-gram types")

    def calculate_class_statistics(self):
        """Calculate statistics for each class"""
        print("📊 Calculating class statistics...")

        if not self.pattern_data:
            self.load_pattern_data()

        # Get all unique classes
        all_classes = set()
        for n_data in self.pattern_data.values():
            if 'discriminative' in n_data and n_data['discriminative'] is not None:
                all_classes.update(n_data['discriminative']['class'].unique())

        print(f"🎯 Found {len(all_classes)} classes")

        # Calculate total documents per class (from pattern frequencies)
        class_doc_counts = defaultdict(int)

        for n_data in self.pattern_data.values():
            if 'counts' in n_data and n_data['counts'] is not None:
                for _, row in n_data['counts'].iterrows():
                    class_name = row['class']
                    frequency = row['frequency']
                    # Estimate documents: assume each pattern appears once per document
                    # This is a rough approximation - in reality you'd need the original data
                    class_doc_counts[class_name] = max(class_doc_counts[class_name], frequency)

        self.class_stats = dict(class_doc_counts)
        total_docs = sum(self.class_stats.values())

        print("📈 Class document counts:")
        for class_name, count in sorted(self.class_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_docs * 100) if total_docs > 0 else 0
            print(".1f")

        self.global_stats = {
            'total_classes': len(all_classes),
            'total_docs': total_docs,
            'avg_docs_per_class': total_docs / len(all_classes) if all_classes else 0
        }

    def calculate_confidence_scores(self, n: int) -> pd.DataFrame:
        """
        Calculate confidence scores for n-grams
        Confidence = (Pattern frequency in class) / (Total patterns in class)
        """
        if n not in self.pattern_data or 'counts' not in self.pattern_data[n]:
            print(f"❌ No {n}-gram count data available")
            return pd.DataFrame()

        df = self.pattern_data[n]['counts'].copy()

        # Calculate total patterns per class
        class_totals = df.groupby('class')['frequency'].sum().to_dict()

        # Calculate confidence for each pattern
        confidences = []
        for _, row in df.iterrows():
            class_name = row['class']
            pattern_freq = row['frequency']
            class_total = class_totals.get(class_name, 1)  # Avoid division by zero

            confidence = pattern_freq / class_total if class_total > 0 else 0
            confidences.append(confidence)

        df['confidence'] = confidences
        df['class_total_patterns'] = df['class'].map(class_totals)

        return df

    def calculate_lift_scores(self, n: int) -> pd.DataFrame:
        """
        Calculate lift scores for n-grams
        Lift = (Pattern freq in class / Class size) / (Pattern freq overall / Total size)
        """
        if n not in self.pattern_data or 'counts' not in self.pattern_data[n]:
            print(f"❌ No {n}-gram count data available")
            return pd.DataFrame()

        df = self.pattern_data[n]['counts'].copy()

        # Calculate overall pattern frequencies
        overall_pattern_freq = df.groupby('ngram')['frequency'].sum().to_dict()

        # Calculate lift for each pattern
        lifts = []
        for _, row in df.iterrows():
            class_name = row['class']
            ngram = row['ngram']
            pattern_in_class = row['frequency']

            class_size = self.class_stats.get(class_name, 1)
            pattern_overall = overall_pattern_freq.get(ngram, 0)
            total_size = self.global_stats.get('total_docs', 1)

            # Lift calculation
            expected_prob = pattern_overall / total_size if total_size > 0 else 0
            actual_prob = pattern_in_class / class_size if class_size > 0 else 0

            lift = actual_prob / expected_prob if expected_prob > 0 else 0
            lifts.append(lift)

        df['lift'] = lifts
        df['overall_frequency'] = df['ngram'].map(overall_pattern_freq)

        return df

    def calculate_rarity_penalty(self, frequency: int, total_patterns: int) -> float:
        """
        Calculate rarity penalty - penalize overly common patterns
        Higher penalty for more common patterns
        """
        if total_patterns == 0:
            return 1.0

        frequency_ratio = frequency / total_patterns

        # Penalize patterns that appear in >10% of all patterns
        if frequency_ratio > 0.1:
            penalty = 0.5 * (1 - frequency_ratio) + 0.5
        else:
            penalty = 1.0

        return penalty

    def calculate_combined_score(self, n: int) -> pd.DataFrame:
        """
        Calculate combined score using multiple metrics
        Combined = Confidence * Lift * Rarity_Penalty * Discriminative_Score
        """
        if n not in self.pattern_data:
            print(f"❌ No {n}-gram data available")
            return pd.DataFrame()

        # Get confidence scores
        conf_df = self.calculate_confidence_scores(n)

        # Get lift scores
        lift_df = self.calculate_lift_scores(n)

        # Get discriminative scores
        disc_df = None
        if 'discriminative' in self.pattern_data[n]:
            disc_df = self.pattern_data[n]['discriminative']

        if conf_df.empty or lift_df.empty:
            print(f"❌ Cannot calculate combined scores for {n}-grams")
            return pd.DataFrame()

        # Merge dataframes
        merged_df = conf_df.merge(lift_df[['ngram', 'class', 'lift', 'overall_frequency']],
                                on=['ngram', 'class'], how='left')

        # Add discriminative scores if available
        if disc_df is not None:
            merged_df = merged_df.merge(disc_df[['ngram', 'class', 'discriminative_score']],
                                      on=['ngram', 'class'], how='left')
        else:
            merged_df['discriminative_score'] = 1.0

        # Fill NaN values
        merged_df = merged_df.fillna({
            'lift': 1.0,
            'overall_frequency': 0,
            'discriminative_score': 1.0
        })

        # Calculate rarity penalty
        rarity_penalties = []
        for _, row in merged_df.iterrows():
            penalty = self.calculate_rarity_penalty(
                row['overall_frequency'],
                merged_df['overall_frequency'].sum()
            )
            rarity_penalties.append(penalty)

        merged_df['rarity_penalty'] = rarity_penalties

        # Calculate combined score
        merged_df['combined_score'] = (
            merged_df['confidence'] *
            merged_df['lift'] *
            merged_df['rarity_penalty'] *
            merged_df['discriminative_score']
        )

        # Apply thresholds
        merged_df = merged_df[
            (merged_df['confidence'] >= self.min_confidence) &
            (merged_df['lift'] >= self.lift_threshold) &
            (merged_df['frequency'] >= self.min_support)
        ]

        # Sort by combined score
        merged_df = merged_df.sort_values('combined_score', ascending=False)

        return merged_df

    def rank_patterns_by_class(self, n: int, top_k: int = 20) -> Dict[str, List[Dict]]:
        """
        Rank top patterns for each class
        Returns: {class_name: [pattern_dicts]}
        """
        combined_df = self.calculate_combined_score(n)

        if combined_df.empty:
            print(f"❌ No valid {n}-gram patterns found")
            return {}

        # Group by class and get top patterns
        class_rankings = {}

        for class_name in combined_df['class'].unique():
            class_patterns = combined_df[combined_df['class'] == class_name]

            top_patterns = []
            for _, row in class_patterns.head(top_k).iterrows():
                pattern_info = {
                    'ngram': row['ngram'],
                    'frequency': row['frequency'],
                    'confidence': row['confidence'],
                    'lift': row['lift'],
                    'rarity_penalty': row['rarity_penalty'],
                    'discriminative_score': row['discriminative_score'],
                    'combined_score': row['combined_score'],
                    'overall_frequency': row['overall_frequency'],
                    'class_total_patterns': row['class_total_patterns']
                }
                top_patterns.append(pattern_info)

            class_rankings[class_name] = top_patterns

        return class_rankings

    def create_pattern_report(self, n: int, top_k: int = 10):
        """
        Create a comprehensive pattern ranking report
        """
        print(f"\n{'='*60}")
        print(f"🎯 PATTERN RANKING REPORT - {n}-GRAMS")
        print(f"{'='*60}")

        if not self.class_stats:
            self.calculate_class_statistics()

        # Get rankings
        rankings = self.rank_patterns_by_class(n, top_k=top_k)

        if not rankings:
            print(f"❌ No patterns found for {n}-grams")
            return

        print(f"📊 Classes analyzed: {len(rankings)}")
        print(f"🎯 Top {top_k} patterns per class")
        print(f"📈 Scoring metrics: Confidence, Lift, Rarity, Discriminative, Combined")
        print()

        for class_name, patterns in rankings.items():
            if not patterns:
                continue

            print(f"🏆 Class: {class_name}")
            print("-" * 50)

            for i, pattern in enumerate(patterns[:top_k], 1):
                ngram = pattern['ngram']
                score = pattern['combined_score']
                confidence = pattern['confidence']
                lift = pattern['lift']

                print(f"{i:2d}. Pattern: '{ngram}' | Score: {score:.3f} | Confidence: {confidence:.3f} | Lift: {lift:.2f}")
            print()

    def export_rankings(self, n_values=[2, 3, 4], top_k=50):
        """
        Export pattern rankings to CSV files
        """
        print("💾 Exporting pattern rankings...")

        for n in n_values:
            rankings = self.rank_patterns_by_class(n, top_k=top_k)

            if not rankings:
                continue

            # Flatten rankings for CSV export
            rows = []
            for class_name, patterns in rankings.items():
                for pattern in patterns:
                    row = {
                        'class': class_name,
                        'ngram': pattern['ngram'],
                        'frequency': pattern['frequency'],
                        'confidence': pattern['confidence'],
                        'lift': pattern['lift'],
                        'rarity_penalty': pattern['rarity_penalty'],
                        'discriminative_score': pattern['discriminative_score'],
                        'combined_score': pattern['combined_score'],
                        'overall_frequency': pattern['overall_frequency'],
                        'class_total_patterns': pattern['class_total_patterns'],
                        'rank': patterns.index(pattern) + 1
                    }
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                output_file = f'{self.results_dir}/{n}gram_rankings.csv'
                df.to_csv(output_file, index=False)
                print(f"  ✅ Exported {len(df)} {n}-gram rankings to {output_file}")

    def visualize_top_patterns(self, n: int, top_classes: int = 5, top_patterns: int = 10):
        """
        Create visualization of top patterns across classes
        """
        rankings = self.rank_patterns_by_class(n, top_k=top_patterns)

        if not rankings:
            return

        # Get top classes by number of patterns
        class_sizes = {cls: len(patterns) for cls, patterns in rankings.items()}
        top_class_names = sorted(class_sizes.keys(), key=lambda x: class_sizes[x], reverse=True)[:top_classes]

        plt.figure(figsize=(15, 10))

        for i, class_name in enumerate(top_class_names):
            patterns = rankings[class_name][:top_patterns]

            if patterns:
                # Plot combined scores
                scores = [p['combined_score'] for p in patterns]
                ngrams = [p['ngram'][:20] + '...' if len(p['ngram']) > 20 else p['ngram'] for p in patterns]

                plt.subplot(len(top_class_names), 1, i+1)
                bars = plt.barh(range(len(scores)), scores)
                plt.yticks(range(len(scores)), ngrams, fontsize=8)
                plt.xlabel('Combined Score')
                plt.title(f'Top {top_patterns} {n}-gram Patterns - Class: {class_name}', fontsize=12)
                plt.grid(True, alpha=0.3)

                # Color bars by lift score
                for j, (bar, pattern) in enumerate(zip(bars, patterns)):
                    lift = pattern['lift']
                    if lift > 3:
                        bar.set_color('red')
                    elif lift > 2:
                        bar.set_color('orange')
                    elif lift > 1.5:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('blue')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/top_{n}gram_patterns_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Visualization saved: top_{n}gram_patterns_comparison.png")

    def run_complete_ranking_analysis(self, n_values=[2, 3, 4], top_k=20):
        """
        Run complete pattern ranking analysis
        """
        print("🚀 Starting Complete Pattern Ranking Analysis")
        print("=" * 60)

        # Load data
        self.load_pattern_data()

        # Calculate statistics
        self.calculate_class_statistics()

        # Analyze each n-gram type
        for n in n_values:
            print(f"\n🔍 Analyzing {n}-gram patterns...")

            # Create report
            self.create_pattern_report(n, top_k=top_k)

            # Create visualizations
            self.visualize_top_patterns(n, top_classes=5, top_patterns=top_k)

        # Export all rankings
        self.export_rankings(n_values, top_k=top_k)

        print(f"\n{'='*60}")
        print("✅ Pattern Ranking Analysis Complete!")
        print(f"{'='*60}")

        # Summary
        total_patterns = sum(len(rankings) for rankings in self.ranked_patterns.values())
        print("📊 Summary:")
        print(f"  N-gram types analyzed: {len(n_values)}")
        print(f"  Classes analyzed: {len(self.class_stats)}")
        print(f"  Total documents: {self.global_stats.get('total_docs', 0)}")
        print(f"  Rankings exported to: {self.results_dir}/")
        print(f"  Visualizations created: top_*gram_patterns_comparison.png")


if __name__ == '__main__':
    # Initialize ranker
    ranker = PatternRanker()

    # Run complete analysis
    ranker.run_complete_ranking_analysis(
        n_values=[2, 3, 4],  # Analyze bigrams, trigrams, 4-grams
        top_k=20             # Top 20 patterns per class
    )

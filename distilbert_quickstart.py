#!/usr/bin/env python3
"""
Quick-start script for DistilBERT semantic pattern analysis

Usage:
    python distilbert_quickstart.py --sample-size 500 --n-themes 5 --n-sub-themes 3

This script provides a simple CLI interface to the pattern analyzer.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from distilbert_pattern_analyzer import DistilBertPatternAnalyzer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DistilBERT Semantic Pattern Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze first 500 texts with 5 themes
  python distilbert_quickstart.py --sample-size 500

  # Full analysis with 8 themes
  python distilbert_quickstart.py --sample-size all --n-themes 8

  # Custom output directory
  python distilbert_quickstart.py --output-dir my_analysis
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/opinions.csv',
        help='Path to CSV file with text and type columns (default: data/raw/opinions.csv)'
    )

    parser.add_argument(
        '--sample-size',
        type=str,
        default='500',
        help='Number of texts to analyze. Use "all" for entire dataset (default: 500)'
    )

    parser.add_argument(
        '--n-themes',
        type=int,
        default=5,
        help='Number of semantic themes to extract (default: 5)'
    )

    parser.add_argument(
        '--n-sub-themes',
        type=int,
        default=3,
        help='Number of sub-themes per theme (default: 3)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32, reduce if OOM)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='distilbert_results',
        help='Output directory for results (default: distilbert_results)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='distilbert-base-uncased',
        help='HuggingFace model ID (default: distilbert-base-uncased)'
    )

    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()

    # Load data
    print(f"📥 Loading data from {args.data_path}...")
    try:
        df = pd.read_csv(
            args.data_path,
            sep=',',
            quotechar='"',
            escapechar='\\',
            on_bad_lines='skip',
            engine='python'
        )
    except FileNotFoundError:
        print(f"❌ File not found: {args.data_path}")
        sys.exit(1)

    # Clean columns
    df.columns = df.columns.str.replace(';;;;;;', '')

    # Validate required columns
    if 'text' not in df.columns or 'type' not in df.columns:
        print("❌ CSV must contain 'text' and 'type' columns")
        sys.exit(1)

    # Filter valid data
    df = df.dropna(subset=['text', 'type'])
    df = df[df['text'].str.len() > 10]
    df = df[df['type'].str.len() > 0]

    print(f"✅ Loaded {len(df)} valid texts")
    print(f"\n📊 Category distribution:")
    print(df['type'].value_counts().to_string())

    # Determine sample size
    if args.sample_size.lower() == 'all':
        sample_size = len(df)
    else:
        sample_size = min(int(args.sample_size), len(df))

    texts = df['text'].head(sample_size).tolist()
    categories = df['type'].head(sample_size).tolist()

    print(f"\n🎯 Analyzing {sample_size} texts with {args.n_themes} themes "
          f"({args.n_sub_themes} sub-themes each)")

    # Initialize analyzer
    print(f"\n🤖 Initializing {args.model}...")
    analyzer = DistilBertPatternAnalyzer(
        model_name=args.model,
        results_dir=args.output_dir
    )

    # Extract embeddings
    print(f"\n🔍 Extracting embeddings and attention patterns...")
    analyzer.extract_embeddings_and_attention(texts, batch_size=args.batch_size)

    # Extract themes
    print(f"\n🎨 Extracting {args.n_themes} semantic themes...")
    analyzer.extract_semantic_themes(
        n_themes=args.n_themes,
        n_sub_themes=args.n_sub_themes
    )

    # Map to categories
    print(f"\n🗺️  Mapping texts to themes and categories...")
    analyzer.map_texts_to_themes(texts, categories)

    # Create Sankey
    print(f"\n🌊 Creating Sankey diagram...")
    analyzer.create_sankey_diagram('distilbert_patterns_sankey.html')

    # Export patterns
    print(f"\n💾 Exporting patterns...")
    df_patterns = analyzer.export_patterns_to_csv('distilbert_patterns.csv')

    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"=" * 70)
    print(f"\n📂 Results saved to: {args.output_dir}/")
    print(f"\n📊 Output files:")
    print(f"   • distilbert_patterns_sankey.html - Interactive Sankey diagram")
    print(f"   • distilbert_patterns.csv - Pattern mappings")
    print(f"\n🎯 Key Statistics:")
    print(f"   • Total texts analyzed: {len(texts)}")
    print(f"   • Semantic themes: {args.n_themes}")
    print(f"   • Sub-themes per theme: {args.n_sub_themes}")
    print(f"   • Total patterns: {args.n_themes * args.n_sub_themes}")
    print(f"   • Categories: {len(set(categories))}")
    print(f"\n📈 Theme distribution:")
    for theme_id, theme_info in analyzer.themes.items():
        print(f"   • {theme_info['name']}: {theme_info['instances']} texts")
    print(f"\n📈 Category distribution in patterns:")
    print(df_patterns['category'].value_counts().to_string())
    print(f"\n💡 Next steps:")
    print(f"   1. Open distilbert_results/distilbert_patterns_sankey.html in browser")
    print(f"   2. Explore the interactive Sankey diagram")
    print(f"   3. Analyze distilbert_results/distilbert_patterns.csv for details")
    print(f"   4. Adjust n_themes and n_sub_themes to refine patterns")
    print(f"\n")


if __name__ == '__main__':
    main()

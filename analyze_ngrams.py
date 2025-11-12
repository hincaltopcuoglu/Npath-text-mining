"""
N-gram Analizi Scripti - İlk Aşama
Her type için ağır baskın n-gram'ları bulur
"""
import csv
import sys

import pandas as pd

from text_pattern_miner import TextPatternMiner


def main():
    """Run n-gram analysis"""

    # Data loading
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/raw/opinions.csv'

    print("=" * 80)
    print("N-GRAM ANALYSIS - PHASE 1")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print()

    # Load data - solve CSV parsing issues
    try:
        # Parse file manually - smarter parsing
        rows = []
        with open(data_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Clean extra ; at end of line
                while line.endswith(';'):
                    line = line[:-1]

                # Split by comma count
                parts = line.split(',')
                if len(parts) == 5:
                    # Normal row: id,topic_id,text,type,effectiveness
                    rows.append(parts)
                elif len(parts) == 1:
                    # Single part row - probably multiple fields in quotes
                    # Parse again
                    try:
                        reader = csv.reader([line], quotechar='"', delimiter=',')
                        parsed = next(reader)
                        if len(parsed) >= 5:
                            rows.append(parsed[:5])
                    except Exception:
                        # Parse edilemezse atla
                        continue
                else:
                    # Take first 5 parts for other cases
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=['id', 'topic_id', 'text', 'type', 'effectiveness'])
        # Clean empty rows
        df = df.dropna(subset=['text', 'type'])
        print(f"✓ {len(df)} documents loaded")

        # Check column names
        print(f"Columns: {list(df.columns)}")

        # Find type column (type, category, label, etc.)
        type_column = None
        for col in ['type', 'category', 'label', 'class']:
            if col in df.columns:
                type_column = col
                break

        if type_column is None:
            print("ERROR: 'type', 'category', 'label' or 'class' column not found!")
            print(f"Available columns: {list(df.columns)}")
            return

        # Find text column
        text_column = None
        for col in ['text', 'sentence', 'content', 'opinion']:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            print("ERROR: 'text', 'sentence', 'content' or 'opinion' column not found!")
            print(f"Available columns: {list(df.columns)}")
            return

        print(f"✓ Text column: {text_column}")
        print(f"✓ Type column: {type_column}")
        print(f"✓ Types: {df[type_column].unique()}")
        print()

    except FileNotFoundError:
        print(f"ERROR: File not found: {data_path}")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Create TextPatternMiner
    miner = TextPatternMiner(
        data=df,
        text_column=text_column,
        category_column=type_column,
        remove_stopwords=False,  # Recommended False for Turkish
        lowercase=True,
        language='turkish'
    )

    # Run n-gram analysis
    # Bigram (2), Trigram (3), 4-gram analysis
    miner.analyze_ngrams(
        n_values=[2, 3, 4],
        min_support=0.01,  # At least 1% support
        top_n=30  # Top 30 for each type
    )

    # Export to CSV
    print("\n" + "=" * 80)
    print("EXPORTING N-GRAMS...")
    print("=" * 80)
    exported_files = miner.export_ngrams(
        n_values=[2, 3, 4],
        min_support=0.01,
        top_n=100
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)
    print("Exported files:")
    for file in exported_files:
        print(f"  - {file}")

    # Statistics
    stats = miner.ngram_analyzer.get_ngram_statistics()
    print("\nStatistics:")
    print(f"  Number of types: {len(stats['types'])}")
    for type_name in stats['types']:
        print(f"  {type_name}: {stats['type_doc_counts'][type_name]} documents")


if __name__ == '__main__':
    main()


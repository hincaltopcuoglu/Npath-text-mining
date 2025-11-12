"""
Discriminative N-gram Analizi Scripti - İkinci Aşama
Hangi n-gram'ların hangi type'a özgü olduğunu bulur
"""
import pandas as pd
from text_pattern_miner import TextPatternMiner

def main():
    """Run discriminative n-gram analysis"""

    # Data loading
    data_path = 'data/raw/opinions.csv'

    print("=" * 80)
    print("PHASE 2 - DISCRIMINATIVE N-GRAM ANALYSIS")
    print("=" * 80)

    # Load data - solve CSV parsing issues
    try:
        # Parse file manually first
        import csv
        rows = []
        with open(data_path, 'r', encoding='utf-8') as f:
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
                    try:
                        reader = csv.reader([line], quotechar='"', delimiter=',')
                        parsed = next(reader)
                        if len(parsed) >= 5:
                            rows.append(parsed[:5])
                    except:
                        continue
                else:
                    # Take first 5 parts for other cases
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=['id', 'topic_id', 'text', 'type', 'effectiveness'])
        # Clean empty rows
        df = df.dropna(subset=['text', 'type'])
        print(f"✓ {len(df)} documents loaded")

    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Create TextPatternMiner
    miner = TextPatternMiner(
        data=df,
        text_column='text',
        category_column='type',
        remove_stopwords=False,
        lowercase=True,
        language='turkish'
    )

    # Phase 1: N-gram analysis (if not done yet)
    print("\nChecking Phase 1...")
    if not hasattr(miner, 'ngram_analyzer') or not miner.ngram_analyzer.ngram_counts:
        print("Running n-gram analysis...")
        miner.analyze_ngrams(n_values=[2, 3, 4], min_support=0.01, top_n=50)
    else:
        print("✓ N-gram analysis already completed")

    # Phase 2: Discriminative analysis
    print("\n" + "=" * 80)
    print("STARTING PHASE 2")
    print("=" * 80)

    # Discriminative analysis for each n-gram size
    for n in [2, 3, 4]:
        print(f"\n--- {n}-GRAM DISCRIMINATIVE ANALYSIS ---")
        discriminative = miner.analyze_discriminative_ngrams(
            n=n,
            min_support=0.01,
            discriminative_threshold=2.0,  # Stricter threshold
            top_n=30
        )

        print(f"  {n}-gram discriminative analysis completed")

    # Export discriminative n-grams
    print("\n" + "=" * 80)
    print("EXPORTING DISCRIMINATIVE N-GRAMS...")
    print("=" * 80)

    exported_files = []
    for n in [2, 3, 4]:
        output_path = f'discriminative_{n}grams.csv'
        miner.export_discriminative_ngrams(
            n=n,
            output_path=output_path,
            min_support=0.01,
            discriminative_threshold=2.0,
            top_n=100
        )
        exported_files.append(output_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)
    print("Exported files:")
    for file in exported_files:
        print(f"  - {file}")

    print(f"\nAnalysis Summary:")
    print(f"  Total documents: {len(df)}")
    print(f"  Number of types: {len(df['type'].unique())}")
    print(f"  Types: {', '.join(df['type'].unique())}")


if __name__ == '__main__':
    main()


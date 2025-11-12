"""
Discriminative N-gram Analizi Scripti - İkinci Aşama
Hangi n-gram'ların hangi type'a özgü olduğunu bulur
"""
import pandas as pd
from text_pattern_miner import TextPatternMiner

def main():
    """Discriminative n-gram analizi çalıştır"""

    # Veri yükleme
    data_path = 'data/raw/opinions.csv'

    print("=" * 80)
    print("İKİNCİ AŞAMA - DISCRIMINATIVE N-GRAM ANALİZİ")
    print("=" * 80)

    # Veriyi yükle - CSV parsing sorununu çöz
    try:
        # Önce dosyayı manuel parse et
        import csv
        rows = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Satırın sonunda fazla ; varsa temizle
                while line.endswith(';'):
                    line = line[:-1]

                # Virgül sayısına göre ayır
                parts = line.split(',')
                if len(parts) == 5:
                    # Normal satır: id,topic_id,text,type,effectiveness
                    rows.append(parts)
                elif len(parts) == 1:
                    # Tek parçalı satır - muhtemelen quotes içinde çoklu alan
                    try:
                        reader = csv.reader([line], quotechar='"', delimiter=',')
                        parsed = next(reader)
                        if len(parsed) >= 5:
                            rows.append(parsed[:5])
                    except:
                        continue
                else:
                    # Diğer durumlar için ilk 5 parçayı al
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=['id', 'topic_id', 'text', 'type', 'effectiveness'])
        # Boş satırları temizle
        df = df.dropna(subset=['text', 'type'])
        print(f"✓ {len(df)} doküman yüklendi")

    except Exception as e:
        print(f"HATA: {e}")
        return

    # TextPatternMiner oluştur
    miner = TextPatternMiner(
        data=df,
        text_column='text',
        category_column='type',
        remove_stopwords=False,
        lowercase=True,
        language='turkish'
    )

    # İlk aşama: N-gram analizi (eğer yapılmamışsa)
    print("\nBirinci aşama kontrol ediliyor...")
    if not hasattr(miner, 'ngram_analyzer') or not miner.ngram_analyzer.ngram_counts:
        print("N-gram analizi yapılıyor...")
        miner.analyze_ngrams(n_values=[2, 3, 4], min_support=0.01, top_n=50)
    else:
        print("✓ N-gram analizi zaten yapılmış")

    # İkinci aşama: Discriminative analizi
    print("\n" + "=" * 80)
    print("İKİNCİ AŞAMA BAŞLATILIYOR")
    print("=" * 80)

    # Her n-gram boyutu için discriminative analizi
    for n in [2, 3, 4]:
        print(f"\n--- {n}-GRAM DISCRIMINATIVE ANALİZİ ---")
        discriminative = miner.analyze_discriminative_ngrams(
            n=n,
            min_support=0.01,
            discriminative_threshold=2.0,  # Daha katı threshold
            top_n=30
        )

        print(f"  {n}-gram discriminative analizi tamamlandı")

    # Export discriminative n-grams
    print("\n" + "=" * 80)
    print("DISCRIMINATIVE N-GRAM'LARI EXPORT EDİYOR...")
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
    print("ANALİZ TAMAMLANDI!")
    print("=" * 80)
    print("Export edilen dosyalar:")
    for file in exported_files:
        print(f"  - {file}")

    print(f"\nAnaliz Özeti:")
    print(f"  Toplam doküman: {len(df)}")
    print(f"  Type sayısı: {len(df['type'].unique())}")
    print(f"  Type'lar: {', '.join(df['type'].unique())}")


if __name__ == '__main__':
    main()


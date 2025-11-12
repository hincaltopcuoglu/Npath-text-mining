"""
N-gram Analizi Scripti - İlk Aşama
Her type için ağır baskın n-gram'ları bulur
"""
import pandas as pd
import sys
from text_pattern_miner import TextPatternMiner

def main():
    """N-gram analizi çalıştır"""
    
    # Veri yükleme
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/raw/opinions.csv'
    
    print("=" * 80)
    print("N-GRAM ANALİZİ - İLK AŞAMA")
    print("=" * 80)
    print(f"Veri dosyası: {data_path}")
    print()
    
    # Veriyi yükle - CSV parsing sorununu çöz
    try:
        # Önce dosyayı manuel parse et - daha akıllı parsing
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
                    # Tekrar parse et
                    try:
                        reader = csv.reader([line], quotechar='"', delimiter=',')
                        parsed = next(reader)
                        if len(parsed) >= 5:
                            rows.append(parsed[:5])
                    except:
                        # Parse edilemezse atla
                        continue
                else:
                    # Diğer durumlar için ilk 5 parçayı al
                    rows.append(parts[:5])

        df = pd.DataFrame(rows, columns=['id', 'topic_id', 'text', 'type', 'effectiveness'])
        # Boş satırları temizle
        df = df.dropna(subset=['text', 'type'])
        print(f"✓ {len(df)} doküman yüklendi")
        
        # Kolon isimlerini kontrol et
        print(f"Kolonlar: {list(df.columns)}")
        
        # Type kolonunu bul (type, category, label, vb.)
        type_column = None
        for col in ['type', 'category', 'label', 'class']:
            if col in df.columns:
                type_column = col
                break
        
        if type_column is None:
            print("HATA: 'type', 'category', 'label' veya 'class' kolonu bulunamadı!")
            print(f"Mevcut kolonlar: {list(df.columns)}")
            return
        
        # Text kolonunu bul
        text_column = None
        for col in ['text', 'sentence', 'content', 'opinion']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print("HATA: 'text', 'sentence', 'content' veya 'opinion' kolonu bulunamadı!")
            print(f"Mevcut kolonlar: {list(df.columns)}")
            return
        
        print(f"✓ Text kolonu: {text_column}")
        print(f"✓ Type kolonu: {type_column}")
        print(f"✓ Type'lar: {df[type_column].unique()}")
        print()
        
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı: {data_path}")
        return
    except Exception as e:
        print(f"HATA: {e}")
        return
    
    # TextPatternMiner oluştur
    miner = TextPatternMiner(
        data=df,
        text_column=text_column,
        category_column=type_column,
        remove_stopwords=False,  # Türkçe için False önerilir
        lowercase=True,
        language='turkish'
    )
    
    # N-gram analizi yap
    # Bigram (2), Trigram (3), 4-gram analizi
    ngram_counts = miner.analyze_ngrams(
        n_values=[2, 3, 4],
        min_support=0.01,  # En az %1 support
        top_n=30  # Her type için top 30
    )
    
    # CSV olarak export et
    print("\n" + "=" * 80)
    print("N-GRAM'LARI EXPORT EDİYOR...")
    print("=" * 80)
    exported_files = miner.export_ngrams(
        n_values=[2, 3, 4],
        min_support=0.01,
        top_n=100
    )
    
    print("\n" + "=" * 80)
    print("ANALİZ TAMAMLANDI!")
    print("=" * 80)
    print(f"Export edilen dosyalar:")
    for file in exported_files:
        print(f"  - {file}")
    
    # İstatistikler
    stats = miner.ngram_analyzer.get_ngram_statistics()
    print(f"\nİstatistikler:")
    print(f"  Type sayısı: {len(stats['types'])}")
    for type_name in stats['types']:
        print(f"  {type_name}: {stats['type_doc_counts'][type_name]} doküman")


if __name__ == '__main__':
    main()


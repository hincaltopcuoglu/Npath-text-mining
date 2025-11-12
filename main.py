"""
Örnek kullanım: Text Pattern Mining
"""
import pandas as pd
from text_pattern_miner import TextPatternMiner
import matplotlib.pyplot as plt

# Örnek veri oluştur (gerçek verinizi buraya yükleyebilirsiniz)
def create_sample_data():
    """Örnek 4 kategorili text verisi"""
    data = {
        'text': [
            # Kategori 1: Teknoloji
            'Yapay zeka ve makine öğrenmesi gelecekte çok önemli olacak',
            'Yazılım geliştirme ve programlama dilleri hızla değişiyor',
            'Bulut bilişim ve veri analizi şirketler için kritik',
            'Yapay zeka algoritmaları veri bilimi ile gelişiyor',
            'Yazılım mühendisliği ve kod kalitesi önemli',
            
            # Kategori 2: Sağlık
            'Sağlıklı beslenme ve düzenli egzersiz çok önemli',
            'Doktor kontrolü ve ilaç kullanımı dikkatli olmalı',
            'Sağlık taraması ve erken teşhis hayat kurtarır',
            'Beslenme alışkanlıkları ve yaşam tarzı sağlığı etkiler',
            'Hastane ziyareti ve tedavi süreci dikkat gerektirir',
            
            # Kategori 3: Eğitim
            'Öğrenme süreci ve eğitim metotları gelişiyor',
            'Öğretmen ve öğrenci ilişkisi başarıyı etkiler',
            'Eğitim sistemi ve müfredat güncellenmeli',
            'Öğrenme teknikleri ve çalışma yöntemleri önemli',
            'Okul ve üniversite eğitimi hayatı şekillendirir',
            
            # Kategori 4: İş Dünyası
            'İş stratejisi ve pazarlama planı başarı getirir',
            'Müşteri memnuniyeti ve satış performansı artırılmalı',
            'Yönetim ekibi ve çalışan motivasyonu önemli',
            'İş planı ve finansal yönetim şirket için kritik',
            'Pazarlama kampanyası ve müşteri ilişkileri geliştirilmeli'
        ],
        'category': [
            'Teknoloji', 'Teknoloji', 'Teknoloji', 'Teknoloji', 'Teknoloji',
            'Sağlık', 'Sağlık', 'Sağlık', 'Sağlık', 'Sağlık',
            'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim',
            'İş Dünyası', 'İş Dünyası', 'İş Dünyası', 'İş Dünyası', 'İş Dünyası'
        ]
    }
    return pd.DataFrame(data)


def main():
    """Ana çalıştırma fonksiyonu"""
    
    # Veri yükleme (örnek veri veya kendi veriniz)
    print("=" * 60)
    print("Text Pattern Mining - nPath-like Approach")
    print("=" * 60)
    
    # Örnek veri oluştur (veya kendi verinizi yükleyin)
    # df = pd.read_csv('your_data.csv')  # Kendi verinizi yükleyin
    df = create_sample_data()
    
    print(f"\nLoaded {len(df)} documents")
    print(f"Categories: {df['category'].unique()}")
    
    # Pattern Miner oluştur
    miner = TextPatternMiner(
        data=df,
        text_column='text',
        category_column='category',
        remove_stopwords=False,  # Türkçe için False önerilir
        lowercase=True,
        language='turkish'
    )
    
    # Preprocessing
    miner.preprocess()
    
    # Graph'ları oluştur
    miner.build_graphs()
    
    # Pattern mining
    print("\n" + "=" * 60)
    print("PATTERN MINING")
    print("=" * 60)
    patterns = miner.mine_patterns(
        min_support=0.05,
        max_path_length=4,
        top_n=20
    )
    
    # Pattern'leri yazdır
    print("\nTop Patterns by Category:")
    print("-" * 60)
    for category, pattern_list in patterns.items():
        print(f"\n{category}:")
        for i, (pattern, score) in enumerate(pattern_list[:10], 1):
            pattern_str = ' -> '.join(pattern)
            print(f"  {i}. [{score:.4f}] {pattern_str}")
    
    # Discriminative pattern'leri bul
    print("\n" + "=" * 60)
    print("DISCRIMINATIVE PATTERNS")
    print("=" * 60)
    discriminative = miner.find_discriminative_patterns(
        min_support=0.05,
        max_path_length=4,
        top_n=10
    )
    
    for category, pattern_list in discriminative.items():
        print(f"\n{category}:")
        for i, (pattern, score) in enumerate(pattern_list[:5], 1):
            pattern_str = ' -> '.join(pattern)
            print(f"  {i}. [{score:.2f}x] {pattern_str}")
    
    # Görselleştirme
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    # Kategori istatistikleri
    miner.visualize_statistics()
    plt.savefig('category_statistics.png', dpi=150, bbox_inches='tight')
    print("Saved: category_statistics.png")
    
    # Pattern karşılaştırması
    miner.visualize_patterns(patterns, top_n=10)
    plt.savefig('patterns_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: patterns_comparison.png")
    
    # Discriminative patterns
    miner.visualize_discriminative(discriminative, top_n=10)
    plt.savefig('discriminative_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved: discriminative_patterns.png")
    
    # Heatmap
    miner.visualize_heatmap(patterns, top_n_patterns=15)
    plt.savefig('pattern_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: pattern_heatmap.png")
    
    # Graph görselleştirme (her kategori için)
    graph_figs = miner.visualize_graphs(top_n_nodes=20, top_n_edges=30)
    for category, fig in graph_figs:
        fig.savefig(f'graph_{category}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: graph_{category}.png")
        plt.close(fig)
    
    # Export patterns
    miner.export_patterns(patterns, 'patterns.csv')
    
    # Özet
    summary = miner.get_summary()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total documents: {summary['total_documents']}")
    print(f"Categories: {summary['categories']}")
    for cat, stats in summary['category_stats'].items():
        print(f"\n{cat}:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()


# Text Pattern Mining with Graph-Based Approach (nPath-like)

Bu proje, Teradata Aster'in nPath özelliğine benzer bir yaklaşımla, kategorili text verilerinde hangi cümle/kelime yapılarının hangi sınıflarda daha çok yoğunlaştığını bulmayı amaçlar.

## Özellikler

- **Text Preprocessing**: Tokenization, cleaning, sequence extraction
- **Graph Construction**: Her kategori için ayrı graflar oluşturma
- **Pattern Mining**: Kategoriye özgü path'leri bulma (nPath benzeri)
- **Visualization**: Graph ve pattern görselleştirme
- **Comparison**: Kategoriler arası farklılık analizi

## Kurulum

### Makefile ile (Önerilen)

```bash
# Tüm bağımlılıkları yükle ve ortamı hazırla
make setup

# Veya sadece bağımlılıkları yükle
make install
```

### Manuel Kurulum

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

## Kullanım

### Makefile ile Hızlı Başlangıç

```bash
# Her şeyi hazırla ve çalıştır (setup + örnek veri + analiz)
make all

# Sadece örnek veri oluştur
make create-sample-data

# Veriyi yükle ve analiz et
make run

# Özel veri dosyası ile çalıştır
make run-custom DATA_FILE=data/raw/your_file.csv

# Yardım menüsü
make help
```

### Python ile Kullanım

```python
from text_pattern_miner import TextPatternMiner

# Veri yükleme
miner = TextPatternMiner(data_path='your_data.csv', 
                        text_column='text', 
                        category_column='category')

# Pattern mining
patterns = miner.mine_patterns(min_support=0.1, max_path_length=5)

# Sonuçları görüntüle
miner.visualize_patterns(patterns)
```

## Makefile Komutları

| Komut | Açıklama |
|-------|----------|
| `make help` | Tüm komutları listeler |
| `make setup` | Proje ortamını hazırlar (klasörler + bağımlılıklar) |
| `make install` | Sadece bağımlılıkları yükler |
| `make create-sample-data` | Örnek 4 kategorili veri oluşturur |
| `make load-data` | Veri dosyalarını kontrol eder |
| `make preprocess-data` | Veriyi ön işleme yapar |
| `make run` | Ana analizi çalıştırır |
| `make run-custom DATA_FILE=path` | Özel veri dosyası ile çalıştırır |
| `make test` | Modülleri test eder |
| `make clean` | Geçici dosyaları temizler |
| `make info` | Proje durumunu gösterir |
| `make all` | Her şeyi hazırla ve çalıştır |

## Proje Yapısı

```
├── Makefile                 # Veri yükleme ve proje yönetimi
├── requirements.txt         # Python bağımlılıkları
├── text_pattern_miner.py    # Ana pattern mining modülü
├── graph_builder.py         # Graph construction
├── pattern_finder.py        # nPath-like pattern finding
├── visualizer.py            # Visualization tools
├── utils.py                 # Yardımcı fonksiyonlar
├── main.py                  # Örnek kullanım
├── data/                    # Veri klasörü (Makefile ile oluşturulur)
│   ├── raw/                 # Ham veri dosyaları
│   └── processed/           # İşlenmiş veri dosyaları
└── README.md                # Bu dosya
```

## Veri Formatı

CSV dosyanız şu kolonları içermelidir:
- `text`: Analiz edilecek text verisi
- `category`: Kategori etiketi (4 kategori)

Örnek:
```csv
text,category
"Yapay zeka ve makine öğrenmesi gelecekte çok önemli olacak",Teknoloji
"Sağlıklı beslenme ve düzenli egzersiz çok önemli",Sağlık
```


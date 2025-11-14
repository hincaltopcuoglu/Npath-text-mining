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

## DistilBERT Semantic Analysis (Yeni)

### Sankey Visualization

Proje artık DistilBERT kullanarak semantic themes'i çıkarıyor ve Sankey diyagramı ile görselleştiriyor:

**Özellikler:**
- 🤖 DistilBERT model kullanarak semantic embeddings
- 🎯 K-means clustering ile semantic themes extraction
- 📊 Sankey flow: Themes → Sub-themes → Categories
- 📈 Pattern distribution analizi
- 💾 CSV export

### Colab Notebook Kullanımı

Google Colab'da çalıştırmak için:

1. **Notebook'u açın**: [npath_text_analysis.ipynb](https://colab.research.google.com/github/hincaltopcuoglu/Npath-text-mining/blob/master/npath_text_analysis.ipynb)

2. **Cells'i sırayla çalıştırın**:
   - Cell 18: Install dependencies
   - Cell 19: Download analyzer module
   - Cell 20: Upload `opinions.csv`
   - Cell 21: Run Analysis (5-15 min)
   - Cell 22: View Sankey diagram
   - Cell 23: Analyze pattern distribution
   - Cell 24: Download results

3. **Output files**:
   - `distilbert_patterns_sankey.html` - Interactive Sankey visualization
   - `distilbert_patterns.csv` - Pattern mappings

### Sankey Visualization Açıklaması

Sankey diyagramı şu yapıyı gösterir:

```
Semantic Themes (5)
    ↓
Sub-themes (3 per theme)
    ↓
Opinion Categories (type column)
```

![DistilBERT Semantic Patterns Sankey Visualization](./images/sankey_visualization.png)

*Yukarıdaki görsel: Semantic themes'in opinion categories'ler ile nasıl bağlandığını gösteren Sankey diyagramı*

**İnteraktif Özellikler:**
- 🖱️ Hover: Flow değerlerini görmek için üzerine gelin
- 🔗 Click: Categories arasındaki bağlantıları izleyin
- 💾 Save: HTML sağ tıkla → Kaydet

### Örnek Sonuç

Generated files:
- `distilbert_patterns_sankey.html` - Themes'in opinions'larla nasıl bağlandığını gösterir
- `distilbert_patterns.csv` - Tüm pattern mappings

## Veri Formatı

CSV dosyanız şu kolonları içermelidir:
- `text`: Analiz edilecek text verisi
- `category`: Kategori etiketi (4 kategori) veya `type`: Opinion type etiketi

Örnek:
```csv
text,type
"Yapay zeka ve makine öğrenmesi gelecekte çok önemli olacak",Teknoloji
"Sağlıklı beslenme ve düzenli egzersiz çok önemli",Sağlık
```


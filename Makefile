.PHONY: help install setup download-data preprocess-data run clean test

# Varsayılan hedef
.DEFAULT_GOAL := help

# Değişkenler
# Aktif environment'daki Python'u kullan (environment aktif olmalı)
# Önce python'u dene, yoksa python3 kullan
PYTHON := $(shell which python 2>/dev/null || which python3)
PIP := $(shell which pip 2>/dev/null || which pip3)
DATA_DIR := data
RAW_DATA := $(DATA_DIR)/raw
PROCESSED_DATA := $(DATA_DIR)/processed
VENV := venv
VENV_BIN := $(VENV)/bin

# Renkli çıktı için
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Bu yardım mesajını göster
	@echo "$(BLUE)Text Pattern Mining - nPath-like Project$(NC)"
	@echo ""
	@echo "$(GREEN)Kullanılabilir komutlar:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Bağımlılıkları yükle
	@echo "$(BLUE)Bağımlılıklar yükleniyor...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)NLTK verileri indiriliyor...$(NC)"
	$(PYTHON) -m nltk.downloader punkt stopwords --quiet
	@echo "$(GREEN)✓ Kurulum tamamlandı!$(NC)"

setup: ## Proje ortamını hazırla (klasörler, bağımlılıklar)
	@echo "$(BLUE)Proje ortamı hazırlanıyor...$(NC)"
	@mkdir -p $(RAW_DATA) $(PROCESSED_DATA)
	@echo "$(GREEN)✓ Klasörler oluşturuldu$(NC)"
	@$(MAKE) install
	@echo "$(GREEN)✓ Kurulum tamamlandı!$(NC)"

download-data: ## Örnek veri oluştur (veya URL'den indir)
	@echo "$(BLUE)Veri hazırlanıyor...$(NC)"
	@mkdir -p $(RAW_DATA)
	@echo "$(YELLOW)Not: Kendi verinizi $(RAW_DATA)/ klasörüne koyabilirsiniz$(NC)"
	@echo "$(YELLOW)Veya örnek veri oluşturmak için: make create-sample-data$(NC)"

create-sample-data: ## Örnek 4 kategorili veri oluştur
	@echo "$(BLUE)Örnek veri oluşturuluyor...$(NC)"
	@mkdir -p $(RAW_DATA)
	$(PYTHON) -c "import pandas as pd; \
	data = {'text': [ \
		'Yapay zeka ve makine öğrenmesi gelecekte çok önemli olacak', \
		'Yazılım geliştirme ve programlama dilleri hızla değişiyor', \
		'Bulut bilişim ve veri analizi şirketler için kritik', \
		'Yapay zeka algoritmaları veri bilimi ile gelişiyor', \
		'Yazılım mühendisliği ve kod kalitesi önemli', \
		'Sağlıklı beslenme ve düzenli egzersiz çok önemli', \
		'Doktor kontrolü ve ilaç kullanımı dikkatli olmalı', \
		'Sağlık taraması ve erken teşhis hayat kurtarır', \
		'Beslenme alışkanlıkları ve yaşam tarzı sağlığı etkiler', \
		'Hastane ziyareti ve tedavi süreci dikkat gerektirir', \
		'Öğrenme süreci ve eğitim metotları gelişiyor', \
		'Öğretmen ve öğrenci ilişkisi başarıyı etkiler', \
		'Eğitim sistemi ve müfredat güncellenmeli', \
		'Öğrenme teknikleri ve çalışma yöntemleri önemli', \
		'Okul ve üniversite eğitimi hayatı şekillendirir', \
		'İş stratejisi ve pazarlama planı başarı getirir', \
		'Müşteri memnuniyeti ve satış performansı artırılmalı', \
		'Yönetim ekibi ve çalışan motivasyonu önemli', \
		'İş planı ve finansal yönetim şirket için kritik', \
		'Pazarlama kampanyası ve müşteri ilişkileri geliştirilmeli' \
	], 'category': [ \
		'Teknoloji', 'Teknoloji', 'Teknoloji', 'Teknoloji', 'Teknoloji', \
		'Sağlık', 'Sağlık', 'Sağlık', 'Sağlık', 'Sağlık', \
		'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', \
		'İş Dünyası', 'İş Dünyası', 'İş Dünyası', 'İş Dünyası', 'İş Dünyası' \
	]}; \
	df = pd.DataFrame(data); \
	df.to_csv('$(RAW_DATA)/sample_data.csv', index=False, encoding='utf-8'); \
	print('✓ Örnek veri oluşturuldu: $(RAW_DATA)/sample_data.csv')"

load-data: ## Veri yükleme scriptini çalıştır (data/raw/ klasöründeki CSV'leri kullanır)
	@echo "$(BLUE)Veri yükleniyor...$(NC)"
	@if [ ! -d "$(RAW_DATA)" ] || [ -z "$$(ls -A $(RAW_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "$(YELLOW)Uyarı: $(RAW_DATA)/ klasöründe CSV dosyası bulunamadı$(NC)"; \
		echo "$(YELLOW)Örnek veri oluşturmak için: make create-sample-data$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Veri dosyaları bulundu$(NC)"
	@ls -lh $(RAW_DATA)/*.csv

preprocess-data: ## Veriyi ön işleme (preprocessing)
	@echo "$(BLUE)Veri ön işleme yapılıyor...$(NC)"
	@mkdir -p $(PROCESSED_DATA)
	@if [ ! -f "$(RAW_DATA)/sample_data.csv" ] && [ -z "$$(ls -A $(RAW_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "$(YELLOW)Önce veri yükleyin: make create-sample-data$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) -c "import pandas as pd; \
	import os; \
	from utils import TextPreprocessor; \
	raw_dir = '$(RAW_DATA)'; \
	processed_dir = '$(PROCESSED_DATA)'; \
	for file in os.listdir(raw_dir): \
		if file.endswith('.csv'): \
			df = pd.read_csv(os.path.join(raw_dir, file)); \
			preprocessor = TextPreprocessor(); \
			processed = preprocessor.process_dataframe(df, 'text', 'category'); \
			output_file = os.path.join(processed_dir, 'processed_' + file); \
			processed.to_csv(output_file, index=False, encoding='utf-8'); \
			print(f'✓ İşlendi: {file} -> processed_{file}')"
	@echo "$(GREEN)✓ Ön işleme tamamlandı!$(NC)"

run: ## Ana analizi çalıştır
	@echo "$(BLUE)Pattern mining analizi başlatılıyor...$(NC)"
	@if [ ! -f "$(RAW_DATA)/sample_data.csv" ] && [ -z "$$(ls -A $(RAW_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "$(YELLOW)Önce veri yükleyin: make create-sample-data$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) main.py
	@echo "$(GREEN)✓ Analiz tamamlandı!$(NC)"

run-custom: ## Özel veri dosyası ile çalıştır (DATA_FILE=path/to/file.csv)
	@if [ -z "$(DATA_FILE)" ]; then \
		echo "$(YELLOW)Hata: DATA_FILE parametresi gerekli$(NC)"; \
		echo "$(YELLOW)Kullanım: make run-custom DATA_FILE=data/raw/your_file.csv$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Özel veri ile analiz başlatılıyor: $(DATA_FILE)$(NC)"
	$(PYTHON) -c "import sys; sys.path.insert(0, '.'); \
	from text_pattern_miner import TextPatternMiner; \
	import pandas as pd; \
	df = pd.read_csv('$(DATA_FILE)'); \
	miner = TextPatternMiner(data=df, text_column='text', category_column='category'); \
	miner.preprocess(); \
	miner.build_graphs(); \
	patterns = miner.mine_patterns(min_support=0.01, max_path_length=5, top_n=20); \
	print('\nTop Patterns:'); \
	[print(f'{cat}: {len(p)} patterns') for cat, p in patterns.items()]"

test: ## Testleri çalıştır
	@echo "$(BLUE)Testler çalıştırılıyor...$(NC)"
	$(PYTHON) -c "from utils import TextPreprocessor; \
	from graph_builder import CategoryGraphBuilder; \
	from pattern_finder import PatternFinder; \
	print('✓ Tüm modüller başarıyla import edildi')"
	@echo "$(GREEN)✓ Testler başarılı!$(NC)"

clean: ## Geçici dosyaları ve cache'i temizle
	@echo "$(BLUE)Temizlik yapılıyor...$(NC)"
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf *.pyc */*.pyc */*/*.pyc
	@rm -rf .pytest_cache .mypy_cache
	@find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Temizlik tamamlandı!$(NC)"

clean-all: clean ## Tüm oluşturulan dosyaları temizle (veri hariç)
	@echo "$(BLUE)Tüm oluşturulan dosyalar temizleniyor...$(NC)"
	@rm -rf $(PROCESSED_DATA)
	@rm -f *.png *.csv patterns.csv
	@echo "$(GREEN)✓ Tüm dosyalar temizlendi!$(NC)"

clean-data: ## Veri dosyalarını da temizle (DİKKAT!)
	@echo "$(YELLOW)UYARI: Tüm veri dosyaları silinecek!$(NC)"
	@read -p "Devam etmek istediğinizden emin misiniz? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@rm -rf $(RAW_DATA) $(PROCESSED_DATA)
	@echo "$(GREEN)✓ Veri dosyaları temizlendi!$(NC)"

info: ## Proje bilgilerini göster
	@echo "$(BLUE)Proje Bilgileri$(NC)"
	@echo "=================="
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "Pip: $$($(PIP) --version 2>&1)"
	@echo ""
	@echo "$(GREEN)Veri Durumu:$(NC)"
	@if [ -d "$(RAW_DATA)" ] && [ -n "$$(ls -A $(RAW_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "  Raw data: $$(ls -1 $(RAW_DATA)/*.csv | wc -l) dosya"; \
		ls -lh $(RAW_DATA)/*.csv | awk '{print "    - " $$9 " (" $$5 ")"}'; \
	else \
		echo "  Raw data: Dosya bulunamadı"; \
	fi
	@if [ -d "$(PROCESSED_DATA)" ] && [ -n "$$(ls -A $(PROCESSED_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "  Processed data: $$(ls -1 $(PROCESSED_DATA)/*.csv | wc -l) dosya"; \
	else \
		echo "  Processed data: Henüz işlenmedi"; \
	fi
	@echo ""
	@echo "$(GREEN)Çıktı Dosyaları:$(NC)"
	@if [ -n "$$(ls -1 *.png *.csv 2>/dev/null)" ]; then \
		ls -lh *.png *.csv 2>/dev/null | awk '{print "  - " $$9 " (" $$5 ")"}'; \
	else \
		echo "  Henüz oluşturulmadı"; \
	fi

analyze-ngrams: ## N-gram analizi yap (İlk aşama)
	@echo "$(BLUE)N-gram analizi başlatılıyor...$(NC)"
	@if [ -z "$$(ls -A $(RAW_DATA)/*.csv 2>/dev/null)" ]; then \
		echo "$(YELLOW)Uyarı: $(RAW_DATA)/ klasöründe CSV dosyası bulunamadı$(NC)"; \
		echo "$(YELLOW)Örnek veri oluşturmak için: make create-sample-data$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) analyze_ngrams.py $(RAW_DATA)/*.csv 2>/dev/null || \
	$(PYTHON) analyze_ngrams.py $$(ls -t $(RAW_DATA)/*.csv | head -1)
	@echo "$(GREEN)✓ N-gram analizi tamamlandı!$(NC)"

analyze-discriminative: ## Discriminative n-gram analizi yap (İkinci aşama)
	@echo "$(BLUE)Discriminative n-gram analizi başlatılıyor...$(NC)"
	@if [ ! -f "ngrams_2gram.csv" ] && [ ! -f "ngrams_3gram.csv" ] && [ ! -f "ngrams_4gram.csv" ]; then \
		echo "$(YELLOW)N-gram analizi henüz yapılmamış. Önce n-gram analizi yapılıyor...$(NC)"; \
		$(MAKE) analyze-ngrams; \
	fi
	$(PYTHON) analyze_discriminative.py
	@echo "$(GREEN)✓ Discriminative analizi tamamlandı!$(NC)"

# Hızlı başlangıç: Her şeyi hazırla
all: setup create-sample-data run ## Her şeyi hazırla ve çalıştır (setup + veri + analiz)


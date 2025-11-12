# 🚀 NPath Text Mining - Colab Setup Guide

## Overview
This project implements n-gram analysis and discriminative feature extraction for text classification using the **opinions dataset**.

- **Target**: `type` column (Claim, Evidence, Counterclaim, Rebuttal, etc.)
- **Features**: `text` column (student opinion texts)
- **Goal**: Find discriminative n-grams for classification

## 📁 Project Structure
```
/Users/hincaltopcuoglu/Npath-text-mining/
├── data/
│   └── raw/
│       └── opinions.csv          # Your dataset (34K rows)
├── colab_ngram_analysis.py       # Main analysis script
├── npath_text_analysis.ipynb     # Google Colab notebook
├── npath/                        # Virtual environment
└── colab_results/                # Generated results (after running)
```

## 🔧 Local Setup (Already Done)
```bash
# Virtual environment created
python3 -m venv npath
source npath/bin/activate
pip install -r requirements.txt
```

## 📊 Data Understanding
- **34K+ student opinion texts**
- **Multiple argumentation types**: Claim, Evidence, Counterclaim, Rebuttal
- **Complex CSV format** with quoted text fields

## 🚀 Running on Google Colab

### Step 1: Create GitHub Repository
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit - NPath text mining project"

# Create GitHub repo and push
# Go to https://github.com/new
# Repository name: Npath-text-mining
# Make it public/private as preferred

# Add remote and push
git remote add origin https://github.com/YOUR-USERNAME/Npath-text-mining.git
git branch -M main
git push -u origin main
```

### Step 2: Open in Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"File" → "Open notebook"**
3. Select **"GitHub"** tab
4. Enter: `YOUR-USERNAME/Npath-text-mining`
5. Select: `npath_text_analysis.ipynb`
6. Click **"Open"**

### Step 3: Configure and Run
1. **Update GitHub credentials** in the first cell:
   ```python
   GITHUB_USERNAME = "your-actual-github-username"
   REPO_NAME = "Npath-text-mining"
   BRANCH = "main"
   ```

2. **Run cells sequentially**:
   - **GitHub Sync & Setup** (downloads your repo)
   - **Install Dependencies** (Colab packages)
   - **Quick Data Check** (verify data loading)
   - **Run N-Gram Analysis** (main analysis - takes ~10-30 minutes)
   - **Check Results** (examine outputs)
   - **Download Results** (get files locally)
   - **Sync Back to GitHub** (push results)

## ⚙️ Analysis Parameters
```python
N_VALUES = [2, 3, 4]        # bigrams, trigrams, 4-grams
MIN_FREQ = 5                # minimum frequency for n-grams
MIN_SUPPORT = 10            # minimum support for discriminative analysis
TOP_K = 500                 # top k discriminative n-grams per class
BATCH_SIZE = 1000           # processing batch size
```

## 📤 Output Files
After analysis, you'll get:
- `colab_results/2gram_counts.csv` - Raw bigram frequencies
- `colab_results/2gram_discriminative.csv` - Discriminative bigrams
- `colab_results/3gram_counts.csv` - Trigram frequencies
- `colab_results/3gram_discriminative.csv` - Discriminative trigrams
- `colab_results/4gram_counts.csv` - 4-gram frequencies
- `colab_results/4gram_discriminative.csv` - Discriminative 4-grams
- `colab_results/top_Xgram_discriminative.png` - Visualization plots

## 🔬 What the Analysis Does

### 1. N-Gram Generation
- Tokenizes and cleans text
- Generates n-grams (2, 3, 4 consecutive words)
- Counts frequency by argumentation type

### 2. Discriminative Scoring
For each n-gram, calculates how much more frequent it is in one class vs. others:
```
discriminative_score = (freq_in_class / total_class_docs) / (freq_in_others / total_other_docs)
```

### 3. Top Features
- Keeps top 500 most discriminative n-grams per class
- Exports for use as features in ML models

## 🎯 Next Steps After Analysis
1. **Feature Selection**: Use discriminative n-grams as features
2. **Model Training**: Train classifiers (SVM, Random Forest, BERT)
3. **Evaluation**: Compare performance across n-gram types
4. **Production**: Deploy best model for real-time classification

## 💡 Tips for Colab
- **Runtime**: Analysis takes 10-30 minutes depending on parameters
- **Memory**: Colab has 12GB RAM - sufficient for 34K texts
- **Storage**: Results are ~50-200MB depending on parameters
- **GPU**: Not needed for n-gram analysis (CPU is fine)

## 🔄 Development Workflow
1. **Code locally** → Test changes
2. **Push to GitHub** → Colab pulls latest
3. **Run on Colab** → Large-scale analysis
4. **Download results** → Use for modeling
5. **Sync back** → Store results in repo

---
**Ready to analyze! 🚀** Open the Colab notebook and start with the GitHub sync cell.

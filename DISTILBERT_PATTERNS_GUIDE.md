# DistilBERT Semantic Pattern Analysis Guide

## Overview

This guide explains how to use **DistilBERT attention mechanisms** to identify semantic patterns in your text data and visualize them in a **Sankey diagram** showing the flow: **Semantic Themes → Sub-themes → Categories**.

## Key Differences from N-gram Analysis

| Aspect | N-gram Analysis | DistilBERT Analysis |
|--------|-----------------|-------------------|
| **Pattern Type** | Surface-level word sequences | Semantic/contextual understanding |
| **Context** | Limited to word proximity | Full document attention |
| **Interpretability** | Explicit word sequences | Learned representations |
| **Scalability** | Fast, lightweight | Slower, requires GPU preferred |
| **Flexibility** | Fixed n-gram sizes | Adaptive to all token interactions |

## Architecture: Themes → Sub-themes → Categories

```
┌─────────────────────────────────────────────────────────┐
│                    Text Corpus (N texts)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   DistilBERT Embeddings    │
        │   (N × 768-dimensional)    │
        └────────────┬───────────────┘
                     │
        ┌────────────▼────────────┐
        │  Clustering (K-means)   │
        └────────────┬────────────┘
                     │
    ┌────────────────┴──────────────────┐
    │                                   │
    ▼                                   ▼
5 Semantic Themes              Sub-themes per Theme
(Conceptual,                   (e.g., Basic, Advanced,
 Emotional,                     Integrated)
 Analytical,
 ...)
    │                                   │
    └────────────────┬──────────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  Category Labels (type) │
        │  (Claim, Evidence, ...) │
        └─────────────────────────┘
```

## Installation & Setup

### 1. Install Required Packages

```bash
pip install torch transformers scikit-learn plotly pandas numpy
```

### 2. Files Included

- **`distilbert_pattern_analyzer.py`** - Main analyzer class
- **`distilbert_colab_example.py`** - Step-by-step Colab notebook cells
- **`DISTILBERT_PATTERNS_GUIDE.md`** - This guide

## How It Works

### Step 1: Load Data

```python
import pandas as pd
from distilbert_pattern_analyzer import DistilBertPatternAnalyzer

df = pd.read_csv('data/raw/opinions.csv')
texts = df['text'].tolist()
categories = df['type'].tolist()
```

### Step 2: Initialize Analyzer

```python
analyzer = DistilBertPatternAnalyzer(
    model_name='distilbert-base-uncased',
    results_dir='distilbert_results'
)
```

**Parameters:**
- `model_name`: HuggingFace model ID (distilbert-base-uncased recommended)
- `results_dir`: Where to save outputs

### Step 3: Extract Embeddings & Attention

```python
analyzer.extract_embeddings_and_attention(texts, batch_size=32)
```

This:
- Tokenizes all texts (max 128 tokens)
- Gets DistilBERT CLS token embeddings (768-dim)
- Extracts attention patterns from the model's final layer
- **Output**: `embeddings` (N × 768), `attention_patterns` (N × 12 × 128 × 128)

### Step 4: Extract Semantic Themes

```python
analyzer.extract_semantic_themes(n_themes=5, n_sub_themes=3)
```

This:
- Clusters embeddings into 5 main semantic themes
- For each theme, creates 3 sub-themes via K-means
- Generates theme names automatically
- **Output**: 5 themes with 3 sub-themes each = 15 total semantic patterns

**Example themes generated:**
- Theme 0: **Conceptual Understanding**
  - Basic Understanding
  - Advanced Understanding
  - Integrated Understanding
- Theme 1: **Emotional Expression**
  - Positive Expression
  - Negative Expression
  - Neutral Expression

### Step 5: Map Texts to Themes & Categories

```python
analyzer.map_texts_to_themes(texts, categories)
```

This:
- Assigns each text to a theme and sub-theme
- Tracks relationships: Text → Theme → Sub-theme → Category
- Creates the flow data for Sankey visualization

### Step 6: Create Sankey Diagram

```python
analyzer.create_sankey_diagram('distilbert_patterns_sankey.html')
```

**Sankey Flow:**
```
Themes (left)  →  Sub-themes (middle)  →  Categories (right)
   5 nodes           15 nodes               N unique categories
```

**Features:**
- Interactive hover tooltips (show flow counts)
- Color-coded by semantic theme
- Node sizes proportional to text counts
- Exportable to PNG via Plotly UI

### Step 7: Export Patterns to CSV

```python
df_patterns = analyzer.export_patterns_to_csv('distilbert_patterns.csv')
```

**Output CSV columns:**
- `text_index`: Index in original data
- `text_sample`: First 100 chars of text
- `category`: Original category (type)
- `theme_id`: Theme (0-4)
- `theme_name`: Theme name (e.g., "Conceptual Understanding")
- `sub_theme_id`: Sub-theme (0-2)
- `sub_theme_name`: Sub-theme name

## Google Colab Usage

### Quick Start (11 Cells)

1. **Cell 1**: Install packages
   ```python
   !pip install transformers torch scikit-learn plotly pandas numpy -q
   ```

2. **Cell 2**: Import modules
   ```python
   from distilbert_pattern_analyzer import DistilBertPatternAnalyzer
   ```

3. **Cell 3**: Load data
   ```python
   df = pd.read_csv('opinions.csv')
   texts = df['text'].head(500).tolist()
   categories = df['type'].head(500).tolist()
   ```

4. **Cell 4**: Initialize and extract embeddings (⏱️ ~2-5 min)
   ```python
   analyzer = DistilBertPatternAnalyzer()
   analyzer.extract_embeddings_and_attention(texts, batch_size=32)
   ```

5. **Cell 5**: Extract themes
   ```python
   analyzer.extract_semantic_themes(n_themes=5, n_sub_themes=3)
   ```

6. **Cell 6**: Map to categories
   ```python
   analyzer.map_texts_to_themes(texts, categories)
   ```

7. **Cell 7**: Create Sankey
   ```python
   analyzer.create_sankey_diagram()
   ```

8. **Cell 8-11**: Export & download

See **`distilbert_colab_example.py`** for complete cell content ready to copy-paste.

## Performance Notes

| Aspect | Details |
|--------|---------|
| **Speed** | ~1-2 sec per 32 texts (CPU), ~0.5 sec (GPU) |
| **Memory** | ~2GB for 1000 texts + model |
| **GPU** | Strongly recommended for >500 texts |
| **Sample Size** | Start with 500, increase if needed |

**Colab Tips:**
- Use GPU runtime: Runtime → Change runtime type → GPU
- Colab provides free K80/T4 GPUs (~8-10GB VRAM)
- For 34K texts: Use batching across multiple runs

## Customization

### Adjust Number of Themes

```python
# More detailed semantic breakdown
analyzer.extract_semantic_themes(n_themes=8, n_sub_themes=4)

# Coarser patterns
analyzer.extract_semantic_themes(n_themes=3, n_sub_themes=2)
```

### Use Different DistilBERT Variant

```python
analyzer = DistilBertPatternAnalyzer(
    model_name='distilbert-base-multilingual-cased'  # For non-English
)
```

### Extract Specific Layer's Attention

Modify `extract_embeddings_and_attention()` to use different layers:
```python
# Currently uses: outputs.attentions[-1][:, 0, :, :]  (last layer, first head)
# Can modify to use other layers or attention heads
```

## Understanding the Outputs

### Sankey Diagram Analysis

**What each flow represents:**
- **Width** = Number of texts flowing through that path
- **Source (Themes)** = Semantic conceptual grouping
- **Middle (Sub-themes)** = Sub-category within theme
- **Target (Categories)** = Original annotation category (Claim, Evidence, etc.)

**Example interpretation:**
```
"Conceptual Understanding" theme contains mostly "Claim" category texts,
while "Emotional Expression" theme contains more "Evidence" category texts.
This suggests Claims tend to be conceptually focused,
while Evidence tends to be more emotional in tone.
```

### CSV Pattern Report

```
text_index | category | theme_name                | sub_theme_name
-----------|----------|---------------------------|---------------------
0          | Claim    | Conceptual Understanding  | Advanced Understanding
1          | Evidence | Emotional Expression      | Positive Expression
2          | Claim    | Analytical Reasoning      | Logical Reasoning
```

This shows:
- Which category each text belongs to
- Its semantic theme classification
- Its sub-theme within that classification

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```python
# Reduce batch size
analyzer.extract_embeddings_and_attention(texts, batch_size=16)

# Or use CPU (slower but works)
# analyzer.device = torch.device('cpu')
```

### Issue: Sankey Not Showing Links
**Solution:**
- Ensure `map_texts_to_themes()` was called before `create_sankey_diagram()`
- Check that theme assignments exist: `analyzer.themes` should not be empty

### Issue: Empty or Trivial Themes
**Solution:**
- Increase sample size (more diverse texts help clustering)
- Adjust `n_themes` and `n_sub_themes` parameters
- Use different theme extraction method (e.g., attention-based clustering)

## Next Steps

1. **Experiment with theme counts** - Try n_themes=3-10 to find optimal granularity
2. **Analyze pattern distribution** - Which categories cluster in which themes?
3. **Manual theme interpretation** - Read sample texts from each theme to understand meaning
4. **Cross-validate** - Compare against your manual categorization
5. **Visualize in dashboard** - Use Plotly for interactive exploration

## Advanced: Manual Theme Assignment

You can customize theme names:

```python
analyzer.themes[0]['name'] = 'Academic Discourse'
analyzer.themes[1]['name'] = 'Personal Narrative'
# ... customize as needed
```

Then regenerate Sankey:
```python
analyzer.create_sankey_diagram('custom_themes_sankey.html')
```

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Understanding BERT Attention](https://github.com/jessevig/bertviz)
- [Attention Mechanisms Explained](https://arxiv.org/abs/1706.03762)

## Citation

If you use this analyzer in research:

```
@software{distilbert_pattern_analyzer,
  title={DistilBERT Semantic Pattern Analyzer},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Npath-text-mining}
}
```

---

**Questions?** Check `distilbert_pattern_analyzer.py` docstrings or analyze sample notebooks.

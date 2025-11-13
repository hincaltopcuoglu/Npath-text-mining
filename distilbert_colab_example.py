"""
Google Colab setup for DistilBERT Pattern Analysis

Copy and paste the cells below into your Google Colab notebook
"""

# ============================================================================
# CELL 1: Install dependencies (run this once)
# ============================================================================
"""
!pip install transformers torch scikit-learn plotly pandas numpy -q
!pip install -U transformers -q
"""

# ============================================================================
# CELL 2: Import and setup
# ============================================================================
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set up path to find local modules
if '/content' in os.getcwd():  # Running in Colab
    from google.colab import files
    # Upload your CSV and Python files
    # files.upload()
    
from distilbert_pattern_analyzer import DistilBertPatternAnalyzer
import pandas as pd
import numpy as np
"""

# ============================================================================
# CELL 3: Load data
# ============================================================================
"""
# Load your opinions data
df = pd.read_csv('/content/opinions.csv',
                  sep=',',
                  quotechar='"',
                  escapechar='\\\\',
                  on_bad_lines='skip',
                  engine='python')

# Clean column names
df.columns = df.columns.str.replace(';;;;;;', '')

# Filter valid data
df = df.dropna(subset=['text', 'type'])
df = df[df['text'].str.len() > 10]
df = df[df['type'].str.len() > 0]

print(f"✅ Loaded {len(df)} texts")
print(f"\\nCategory distribution:")
print(df['type'].value_counts())

# For faster testing, use a subset (adjust as needed)
# For full analysis, use: sample_size = len(df)
sample_size = 500  # Change to len(df) for full analysis
texts = df['text'].head(sample_size).tolist()
categories = df['type'].head(sample_size).tolist()

print(f"\\n📊 Using {sample_size} texts for analysis")
"""

# ============================================================================
# CELL 4: Initialize analyzer and extract embeddings
# ============================================================================
"""
# Initialize analyzer
analyzer = DistilBertPatternAnalyzer(
    model_name='distilbert-base-uncased',
    results_dir='/content/distilbert_results'
)

# Extract embeddings and attention patterns
# This may take a few minutes depending on sample size
analyzer.extract_embeddings_and_attention(texts, batch_size=32)
"""

# ============================================================================
# CELL 5: Extract semantic themes and sub-themes
# ============================================================================
"""
# Extract themes (adjust n_themes and n_sub_themes as needed)
analyzer.extract_semantic_themes(n_themes=5, n_sub_themes=3)
"""

# ============================================================================
# CELL 6: Map texts to themes and categories
# ============================================================================
"""
# Map all texts to their themes and categories
analyzer.map_texts_to_themes(texts, categories)
"""

# ============================================================================
# CELL 7: Create Sankey visualization
# ============================================================================
"""
# Create Sankey diagram: Themes → Sub-themes → Categories
analyzer.create_sankey_diagram('distilbert_patterns_sankey.html')

# Display the diagram
from IPython.display import IFrame
IFrame(src='/content/distilbert_results/distilbert_patterns_sankey.html',
       width=1600, height=1000)
"""

# ============================================================================
# CELL 8: Export patterns to CSV
# ============================================================================
"""
# Export the pattern mappings to CSV for further analysis
df_patterns = analyzer.export_patterns_to_csv('distilbert_patterns.csv')

# Download the CSV
files.download('/content/distilbert_results/distilbert_patterns.csv')
"""

# ============================================================================
# CELL 9: Download Sankey HTML
# ============================================================================
"""
# Download the Sankey diagram
files.download('/content/distilbert_results/distilbert_patterns_sankey.html')
"""

# ============================================================================
# CELL 10: Analyze patterns by category
# ============================================================================
"""
# Get summary of patterns per category
print("\\n=== Pattern Analysis by Category ===\\n")

for category in sorted(df_patterns['category'].unique()):
    cat_data = df_patterns[df_patterns['category'] == category]
    print(f"\\n📊 {category} (n={len(cat_data)})")
    print(f"   Themes:")
    for theme in cat_data['theme_name'].unique():
        count = len(cat_data[cat_data['theme_name'] == theme])
        print(f"      • {theme}: {count} texts")
    print(f"   Sub-themes:")
    for subtheme in cat_data['sub_theme_name'].unique():
        count = len(cat_data[cat_data['sub_theme_name'] == subtheme])
        print(f"      • {subtheme}: {count} texts")
"""

# ============================================================================
# CELL 11: Get sample texts from each theme
# ============================================================================
"""
# Show sample texts from each semantic theme
print("\\n=== Sample Texts from Each Theme ===\\n")

for theme_id, theme_info in analyzer.themes.items():
    print(f"\\n🎯 Theme {theme_id}: {theme_info['name']}")
    print(f"   (Total instances: {theme_info['instances']})\\n")
    
    # Get first 2 samples
    samples = df_patterns[df_patterns['theme_id'] == theme_id].head(2)
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"   Sample {idx}:")
        print(f"   Text: {row['text_sample']}...")
        print(f"   Category: {row['category']}")
        print(f"   Sub-theme: {row['sub_theme_name']}\\n")
"""

if __name__ == '__main__':
    print("""
    DistilBERT Pattern Analyzer for Google Colab
    =============================================
    
    This script provides a step-by-step guide for analyzing semantic patterns
    in your text data using DistilBERT attention mechanisms.
    
    Usage:
    1. Copy each cell into your Colab notebook
    2. Run cells sequentially (Cell 1 → Cell 11)
    3. Modify sample_size and theme parameters as needed
    4. Download results (Sankey diagram and CSV)
    
    Key outputs:
    - distilbert_patterns_sankey.html: Interactive Sankey diagram
    - distilbert_patterns.csv: Pattern mappings
    
    For more info, see: distilbert_pattern_analyzer.py
    """)

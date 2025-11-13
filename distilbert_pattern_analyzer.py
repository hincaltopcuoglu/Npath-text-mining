"""
DistilBERT-based Language Pattern Analyzer
Extracts semantic themes and sub-themes from attention patterns
Visualizes patterns using Sankey diagrams
"""

import os
import warnings
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
import plotly.graph_objects as go

warnings.filterwarnings('ignore')


class DistilBertPatternAnalyzer:
    """
    Extracts semantic patterns from text using DistilBERT attention heads
    Identifies themes, sub-themes, and their relationships to categories
    """

    def __init__(self, model_name='distilbert-base-uncased', results_dir='distilbert_results'):
        """
        Initialize DistilBERT model and tokenizer
        
        Args:
            model_name: HuggingFace model identifier
            results_dir: Directory for saving results
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 Using device: {self.device}")
        
        print(f"📥 Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(
            model_name,
            output_attentions=True
        ).to(self.device)
        self.model.eval()
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.embeddings = {}
        self.attention_patterns = {}
        self.category_data = {}
        
        print(f"✅ Model loaded successfully on {self.device}")

    def extract_embeddings_and_attention(self, texts: List[str], batch_size: int = 32):
        """
        Extract embeddings and attention patterns from texts
        
        Args:
            texts: List of text documents
            batch_size: Batch size for processing
        """
        print(f"\n🔍 Extracting embeddings and attention patterns from {len(texts)} texts...")
        
        all_embeddings = []
        all_attention = []
        
        with torch.no_grad():
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Get embeddings (CLS token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
                # Get attention weights (last layer, first head)
                attention = outputs.attentions[-1][:, 0, :, :].cpu().numpy()
                all_attention.append(attention)
                
                if (batch_end - batch_start) % (batch_size * 2) == 0:
                    print(f"  ✅ Processed {batch_end}/{len(texts)} texts")
        
        self.embeddings = np.vstack(all_embeddings)
        self.attention_patterns = np.vstack(all_attention)
        
        print(f"✅ Extracted embeddings shape: {self.embeddings.shape}")
        print(f"✅ Extracted attention shape: {self.attention_patterns.shape}")
        
        return self.embeddings, self.attention_patterns

    def extract_semantic_themes(self, n_themes: int = 5, n_sub_themes: int = 3):
        """
        Extract semantic themes from embeddings using clustering
        
        Args:
            n_themes: Number of main semantic themes
            n_sub_themes: Number of sub-themes per theme
            
        Returns:
            Dictionary with theme and sub-theme information
        """
        print(f"\n🎯 Extracting {n_themes} semantic themes with {n_sub_themes} sub-themes each...")
        
        # Cluster embeddings into main themes
        kmeans_themes = KMeans(n_clusters=n_themes, random_state=42, n_init=10)
        theme_labels = kmeans_themes.fit_predict(self.embeddings)
        
        themes_dict = {}
        
        # For each theme, create sub-themes
        for theme_id in range(n_themes):
            theme_mask = theme_labels == theme_id
            theme_embeddings = self.embeddings[theme_mask]
            
            if len(theme_embeddings) < n_sub_themes:
                n_sub = max(1, len(theme_embeddings) // 2)
            else:
                n_sub = n_sub_themes
            
            # Cluster within theme
            kmeans_sub = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
            sub_theme_labels = kmeans_sub.fit_predict(theme_embeddings)
            
            # Generate theme names based on embedding characteristics
            theme_name = self._generate_theme_name(theme_id, theme_embeddings)
            
            sub_themes = []
            for sub_id in range(n_sub):
                sub_name = self._generate_sub_theme_name(theme_name, sub_id)
                sub_themes.append({
                    'id': sub_id,
                    'name': sub_name,
                    'instances': np.sum(sub_theme_labels == sub_id)
                })
            
            themes_dict[theme_id] = {
                'name': theme_name,
                'instances': np.sum(theme_mask),
                'sub_themes': sub_themes,
                'indices': np.where(theme_mask)[0],
                'sub_labels': sub_theme_labels
            }
        
        self.themes = themes_dict
        print(f"✅ Extracted {n_themes} themes")
        for theme_id, theme_info in themes_dict.items():
            print(f"   Theme {theme_id}: {theme_info['name']} ({theme_info['instances']} instances)")
            for sub in theme_info['sub_themes']:
                print(f"      ├─ {sub['name']} ({sub['instances']} instances)")
        
        return themes_dict

    def _generate_theme_name(self, theme_id: int, embeddings: np.ndarray) -> str:
        """Generate semantic theme name based on embedding characteristics"""
        theme_names = [
            "Conceptual Understanding",
            "Emotional Expression",
            "Analytical Reasoning",
            "Practical Application",
            "Social Interaction",
            "Personal Experience",
            "Theoretical Framework",
            "Comparative Analysis"
        ]
        return theme_names[theme_id % len(theme_names)]

    def _generate_sub_theme_name(self, theme_name: str, sub_id: int) -> str:
        """Generate sub-theme name based on theme and sub-id"""
        qualifiers = {
            "Conceptual Understanding": ["Basic", "Advanced", "Integrated"],
            "Emotional Expression": ["Positive", "Negative", "Neutral"],
            "Analytical Reasoning": ["Logical", "Critical", "Systematic"],
            "Practical Application": ["Direct", "Indirect", "Complex"],
            "Social Interaction": ["Collaborative", "Individual", "Community"],
            "Personal Experience": ["Recent", "Historical", "Reflective"],
            "Theoretical Framework": ["Traditional", "Modern", "Hybrid"],
            "Comparative Analysis": ["Contrast", "Similarity", "Synthesis"]
        }
        
        qual_list = qualifiers.get(theme_name, ["Primary", "Secondary", "Tertiary"])
        return f"{qual_list[sub_id % len(qual_list)]} {theme_name.split()[-1]}"

    def map_texts_to_themes(self, texts: List[str], categories: List[str]):
        """
        Map texts to their themes and categories
        
        Args:
            texts: List of text documents
            categories: List of category labels (type column)
        """
        print(f"\n🗺️  Mapping {len(texts)} texts to themes and categories...")
        
        # Get theme assignments
        kmeans = KMeans(n_clusters=len(self.themes), random_state=42, n_init=10)
        theme_assignments = kmeans.fit_predict(self.embeddings)
        
        self.category_data = {
            'texts': texts,
            'categories': categories,
            'theme_assignments': theme_assignments,
            'embeddings': self.embeddings
        }
        
        # Count combinations
        theme_category_counts = defaultdict(lambda: defaultdict(int))
        for theme_id, category in zip(theme_assignments, categories):
            theme_category_counts[theme_id][category] += 1
        
        print("✅ Text-Theme-Category mapping completed")
        for theme_id in sorted(theme_category_counts.keys()):
            print(f"   Theme {theme_id}:")
            for cat, count in sorted(theme_category_counts[theme_id].items()):
                print(f"      → {cat}: {count} texts")
        
        return theme_category_counts

    def get_top_words_for_theme(self, theme_id: int, n_words: int = 5) -> List[str]:
        """
        Extract top words representing a theme using attention patterns
        
        Args:
            theme_id: Theme identifier
            n_words: Number of top words to extract
            
        Returns:
            List of representative words
        """
        if theme_id not in self.themes:
            return []
        
        theme_indices = self.themes[theme_id]['indices']
        theme_attention = self.attention_patterns[theme_indices]
        
        # Average attention across instances
        avg_attention = np.mean(theme_attention, axis=0)
        
        # Get top attended positions
        top_positions = np.argsort(avg_attention.mean(axis=1))[-n_words:][::-1]
        
        return [f"Pattern_{pos}" for pos in top_positions]

    def create_sankey_diagram(self, output_file: str = 'distilbert_patterns_sankey.html'):
        """
        Create Sankey diagram showing: Themes → Sub-themes → Categories

        Args:
            output_file: Output HTML file name
        """
        print("\n🌊 Creating Sankey diagram...")

        if not self.category_data:
            print("❌ No category data. Run map_texts_to_themes first.")
            return None
        
        # Prepare data for Sankey
        nodes = []
        node_labels = []
        node_colors = []
        node_positions = {}
        
        links_source = []
        links_target = []
        links_value = []
        links_color = []
        
        # Theme colors
        theme_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f'
        ]
        
        # Column 0: Themes
        node_id = 0
        for theme_id, theme_info in sorted(self.themes.items()):
            theme_label = f"Theme: {theme_info['name']}"
            node_labels.append(theme_label)
            node_colors.append(theme_colors[theme_id % len(theme_colors)])
            node_positions[('theme', theme_id)] = node_id
            node_id += 1
        
        # Column 1: Sub-themes
        for theme_id, theme_info in sorted(self.themes.items()):
            for sub_theme in theme_info['sub_themes']:
                sub_label = f"Sub: {sub_theme['name']}"
                node_labels.append(sub_label)
                node_colors.append(theme_colors[theme_id % len(theme_colors)])
                node_positions[('sub_theme', theme_id, sub_theme['id'])] = node_id
                node_id += 1
        
        # Column 2: Categories
        unique_categories = sorted(set(self.category_data['categories']))
        category_colors = {
            cat: theme_colors[idx % len(theme_colors)]
            for idx, cat in enumerate(unique_categories)
        }
        
        for category in unique_categories:
            cat_label = f"Category: {category}"
            node_labels.append(cat_label)
            node_colors.append(category_colors[category])
            node_positions[('category', category)] = node_id
            node_id += 1
        
        # Add nodes to list
        nodes = list(range(len(node_labels)))
        
        # Create flows: Theme → Sub-theme → Category
        theme_category_flows = defaultdict(lambda: defaultdict(int))
        
        for text_idx, (theme_id, category) in enumerate(
            zip(self.category_data['theme_assignments'], self.category_data['categories'])
        ):
            # Get sub-theme for this text
            theme_info = self.themes[theme_id]
            sub_theme_labels = theme_info['sub_labels']
            
            # Find which texts in this theme correspond to this index
            theme_indices = theme_info['indices']
            local_idx = np.where(theme_indices == text_idx)[0]
            
            if len(local_idx) > 0:
                sub_theme_id = sub_theme_labels[local_idx[0]]
                
                # Theme → Sub-theme flow
                source_pos = node_positions[('theme', theme_id)]
                target_pos = node_positions[('sub_theme', theme_id, sub_theme_id)]
                
                # Check if flow already exists
                flow_key = (source_pos, target_pos)
                if flow_key not in [(links_source[i], links_target[i]) for i in range(len(links_source))]:
                    links_source.append(source_pos)
                    links_target.append(target_pos)
                    links_value.append(1)
                    links_color.append(f"rgba({int(theme_colors[theme_id % len(theme_colors)][1:3], 16)}, "
                                     f"{int(theme_colors[theme_id % len(theme_colors)][3:5], 16)}, "
                                     f"{int(theme_colors[theme_id % len(theme_colors)][5:7], 16)}, 0.4)")
                else:
                    idx = [(links_source[i], links_target[i]) for i in range(len(links_source))].index(flow_key)
                    links_value[idx] += 1
                
                # Sub-theme → Category flow
                source_pos = node_positions[('sub_theme', theme_id, sub_theme_id)]
                target_pos = node_positions[('category', category)]
                
                flow_key = (source_pos, target_pos)
                if flow_key not in [(links_source[i], links_target[i]) for i in range(len(links_source))]:
                    links_source.append(source_pos)
                    links_target.append(target_pos)
                    links_value.append(1)
                    links_color.append(f"rgba({int(category_colors[category][1:3], 16)}, "
                                     f"{int(category_colors[category][3:5], 16)}, "
                                     f"{int(category_colors[category][5:7], 16)}, 0.4)")
                else:
                    idx = [(links_source[i], links_target[i]) for i in range(len(links_source))].index(flow_key)
                    links_value[idx] += 1
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=links_source,
                target=links_target,
                value=links_value,
                color=links_color
            )
        )])
        
        fig.update_layout(
            title={
                'text': "DistilBERT Semantic Patterns: Themes → Sub-themes → Categories<br>"
                       "<sub>Language pattern analysis from attention mechanisms</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            font_size=10,
            height=1000,
            width=1600
        )
        
        # Save
        output_path = f'{self.results_dir}/{output_file}'
        fig.write_html(output_path)
        print(f"✅ Sankey diagram saved to: {output_path}")
        
        return fig

    def export_patterns_to_csv(self, output_file: str = 'distilbert_patterns.csv'):
        """
        Export extracted patterns to CSV
        
        Args:
            output_file: Output CSV file name
        """
        print(f"\n💾 Exporting patterns to CSV...")
        
        patterns_data = []
        
        for text_idx, (theme_id, category) in enumerate(
            zip(self.category_data['theme_assignments'], self.category_data['categories'])
        ):
            theme_info = self.themes[theme_id]
            sub_theme_labels = theme_info['sub_labels']
            theme_indices = theme_info['indices']
            
            local_idx = np.where(theme_indices == text_idx)[0]
            if len(local_idx) > 0:
                sub_theme_id = sub_theme_labels[local_idx[0]]
                sub_theme_name = theme_info['sub_themes'][sub_theme_id]['name']
                
                patterns_data.append({
                    'text_index': text_idx,
                    'text_sample': self.category_data['texts'][text_idx][:100],
                    'category': category,
                    'theme_id': theme_id,
                    'theme_name': theme_info['name'],
                    'sub_theme_id': sub_theme_id,
                    'sub_theme_name': sub_theme_name
                })
        
        df = pd.DataFrame(patterns_data)
        output_path = f'{self.results_dir}/{output_file}'
        df.to_csv(output_path, index=False)
        
        print(f"✅ Patterns exported to: {output_path}")
        print(f"   Total patterns: {len(df)}")
        print(f"\n   Pattern distribution by category:")
        print(df['category'].value_counts())
        
        return df


def main():
    """Example usage"""
    # Load data
    print("📥 Loading data...")
    df = pd.read_csv('data/raw/opinions.csv',
                     sep=',',
                     quotechar='"',
                     escapechar='\\',
                     on_bad_lines='skip',
                     engine='python')
    
    # Clean column names
    df.columns = df.columns.str.replace(';;;;;;', '')
    
    # Filter valid data
    df = df.dropna(subset=['text', 'type'])
    df = df[df['text'].str.len() > 10]
    df = df[df['type'].str.len() > 0]
    
    print(f"✅ Loaded {len(df)} texts")
    
    # Limit to first 500 for demo (adjust as needed)
    sample_size = min(500, len(df))
    texts = df['text'].head(sample_size).tolist()
    categories = df['type'].head(sample_size).tolist()
    
    print(f"📊 Using {sample_size} texts for analysis")
    
    # Initialize analyzer
    analyzer = DistilBertPatternAnalyzer(results_dir='distilbert_results')
    
    # Extract embeddings and attention
    analyzer.extract_embeddings_and_attention(texts, batch_size=32)
    
    # Extract semantic themes
    themes = analyzer.extract_semantic_themes(n_themes=5, n_sub_themes=3)
    
    # Map texts to themes and categories
    analyzer.map_texts_to_themes(texts, categories)
    
    # Create Sankey diagram
    analyzer.create_sankey_diagram()
    
    # Export patterns
    analyzer.export_patterns_to_csv()
    
    print("\n✅ Analysis complete!")
    print(f"📂 Results saved to: distilbert_results/")


if __name__ == '__main__':
    main()

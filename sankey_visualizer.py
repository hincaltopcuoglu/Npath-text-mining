"""
Sankey Diagram Visualizer for nPath-like Sequential Pattern Analysis
Shows flow of n-grams from 1-gram to 5-gram and their impact on classes
"""
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings('ignore')


class SankeyVisualizer:
    """
    Creates Sankey diagrams showing sequential n-gram patterns
    Similar to Teradata Aster nPath visualization
    """

    def __init__(self, results_dir='colab_results'):
        self.results_dir = results_dir
        self.ngram_data = {}
        self.class_colors = {}

    def load_ngram_data(self, n_values=[1, 2, 3, 4, 5]):
        """Load n-gram data from results directory"""
        print("📥 Loading n-gram data for Sankey visualization...")

        for n in n_values:
            counts_file = f'{self.results_dir}/{n}gram_counts.csv'
            disc_file = f'{self.results_dir}/{n}gram_discriminative.csv'

            try:
                counts_df = None
                disc_df = None

                if os.path.exists(counts_file):
                    counts_df = pd.read_csv(counts_file)
                    print(f"  ✅ Loaded {n}-gram counts: {len(counts_df)} entries")

                if os.path.exists(disc_file):
                    disc_df = pd.read_csv(disc_file)
                    print(f"  ✅ Loaded {n}-gram discriminative: {len(disc_df)} entries")

                if counts_df is not None or disc_df is not None:
                    self.ngram_data[n] = {
                        'counts': counts_df,
                        'discriminative': disc_df
                    }

            except Exception as e:
                print(f"  ⚠️  Could not load {n}-gram data: {e}")

        print(f"📊 Loaded data for {len(self.ngram_data)} n-gram types")

    def extract_sequential_patterns(self, top_k_per_class=20):
        """
        Extract sequential patterns showing how n-grams evolve
        Returns: Dict with flow information for Sankey diagram
        """
        print("\n🔄 Extracting sequential n-gram patterns...")

        # Get all classes from discriminative data
        all_classes = set()
        for n, data in self.ngram_data.items():
            if data.get('discriminative') is not None:
                if 'class' in data['discriminative'].columns:
                    all_classes.update(data['discriminative']['class'].unique())
                elif 'type' in data['discriminative'].columns:
                    all_classes.update(data['discriminative']['type'].unique())

        all_classes = sorted(list(all_classes))
        print(f"  Found {len(all_classes)} classes: {all_classes}")

        # Assign colors to classes
        import matplotlib.colors as mcolors
        # Get colors and convert to hex format
        tableau_colors = list(mcolors.TABLEAU_COLORS.values())
        # Convert matplotlib color names to hex
        hex_colors = []
        for color in tableau_colors:
            if isinstance(color, str) and color.startswith('#'):
                # Ensure it's a clean 6-character hex (remove alpha if present)
                hex_clean = color.lstrip('#')
                if len(hex_clean) == 8:
                    # Remove alpha channel, keep only RGB
                    hex_clean = hex_clean[:6]
                elif len(hex_clean) == 3:
                    # Expand 3-char hex to 6-char
                    hex_clean = ''.join([c*2 for c in hex_clean])
                hex_colors.append(f'#{hex_clean}')
            else:
                # Convert matplotlib color name to hex
                try:
                    rgb = mcolors.to_rgb(color)
                    hex_colors.append(mcolors.rgb2hex(rgb))
                except:
                    hex_colors.append('#888888')  # Default gray
        
        self.class_colors = {cls: hex_colors[i % len(hex_colors)] for i, cls in enumerate(all_classes)}

        # Track which classes have patterns
        classes_with_patterns = set()

        # Build sequential flows
        flows = []
        nodes = []
        node_labels = []
        node_positions = {}  # Track which column each node belongs to

        # Column 0: 1-grams (if available)
        # Column 1: 2-grams
        # Column 2: 3-grams
        # Column 3: 4-grams
        # Column 4: 5-grams

        n_sequence = sorted([n for n in self.ngram_data.keys() if n >= 1])
        print(f"  N-gram sequence: {n_sequence}")

        # Extract top patterns for each n-gram size and class
        top_patterns = defaultdict(lambda: defaultdict(list))

        for n in n_sequence:
            if n not in self.ngram_data:
                continue

            disc_df = self.ngram_data[n].get('discriminative')
            if disc_df is None:
                continue

            # Get class column name
            class_col = 'class' if 'class' in disc_df.columns else 'type'
            ngram_col = 'ngram' if 'ngram' in disc_df.columns else 'pattern'

            # Get score column
            score_col = None
            for col in ['discriminative_score', 'score', 'combined_score', 'lift']:
                if col in disc_df.columns:
                    score_col = col
                    break

            if score_col is None:
                continue

            # Group by class and get top patterns
            for class_name in all_classes:
                class_data = disc_df[disc_df[class_col] == class_name]
                if len(class_data) == 0:
                    print(f"    ⚠️  Class '{class_name}' has no {n}-gram data")
                    continue

                # Sort by score and take top K
                top = class_data.nlargest(top_k_per_class, score_col)
                if len(top) > 0:
                    top_patterns[n][class_name] = top.to_dict('records')
                    classes_with_patterns.add(class_name)
                    print(f"    ✅ Class '{class_name}': {len(top)} {n}-grams")
                else:
                    print(f"    ⚠️  Class '{class_name}': No {n}-grams after filtering")

        # Build nodes and flows
        node_id = 0
        source_target_flows = []  # (source_idx, target_idx, value, class, color)

        # For each n-gram level
        for col_idx, n in enumerate(n_sequence):
            if n not in top_patterns:
                continue

            # For each class
            for class_name in all_classes:
                if class_name not in top_patterns[n]:
                    # Try to include this class even with fewer patterns
                    # Check if it exists in any n-gram data
                    disc_df = self.ngram_data[n].get('discriminative')
                    if disc_df is not None:
                        class_col = 'class' if 'class' in disc_df.columns else 'type'
                        class_data = disc_df[disc_df[class_col] == class_name]
                        if len(class_data) > 0:
                            # Use all available patterns for this class (even if < top_k)
                            score_col = None
                            for col in ['discriminative_score', 'score', 'combined_score', 'lift']:
                                if col in disc_df.columns:
                                    score_col = col
                                    break
                            if score_col:
                                top = class_data.nlargest(min(len(class_data), top_k_per_class), score_col)
                                top_patterns[n][class_name] = top.to_dict('records')
                                print(f"    📌 Included class '{class_name}' with {len(top)} {n}-grams (below threshold)")
                    if class_name not in top_patterns[n]:
                        continue

                patterns = top_patterns[n][class_name]

                for pattern in patterns:
                    ngram_col = 'ngram' if 'ngram' in pattern else 'pattern'
                    ngram_str = pattern.get(ngram_col, '')

                    if not ngram_str:
                        continue

                    # Convert ngram to string if it's a tuple/list
                    if isinstance(ngram_str, (tuple, list)):
                        ngram_str = ' '.join(str(x) for x in ngram_str)
                    elif isinstance(ngram_str, str) and ngram_str.startswith('('):
                        # Handle string representation of tuple
                        ngram_str = ngram_str.strip('()').replace("'", "").replace(',', ' ')

                    # Create node label with class prefix for uniqueness
                    node_label = f"{class_name[:10]}: {ngram_str[:30]}"
                    if len(ngram_str) > 30:
                        node_label += "..."

                    # Check if node already exists
                    if node_label not in node_positions:
                        nodes.append({
                            'id': node_id,
                            'label': node_label,
                            'column': col_idx,
                            'class': class_name,
                            'ngram': ngram_str,
                            'n': n
                        })
                        node_positions[node_label] = node_id
                        node_id += 1

                    current_node_id = node_positions[node_label]

                    # Create flows to next level (n+1)
                    if col_idx < len(n_sequence) - 1:
                        next_n = n_sequence[col_idx + 1]
                        if next_n in top_patterns and class_name in top_patterns[next_n]:
                            next_patterns = top_patterns[next_n][class_name]

                            # Find patterns that contain current ngram
                            current_words = ngram_str.split()
                            for next_pattern in next_patterns:
                                next_ngram_col = 'ngram' if 'ngram' in next_pattern else 'pattern'
                                next_ngram_str = next_pattern.get(next_ngram_col, '')

                                if isinstance(next_ngram_str, (tuple, list)):
                                    next_ngram_str = ' '.join(str(x) for x in next_ngram_str)
                                elif isinstance(next_ngram_str, str) and next_ngram_str.startswith('('):
                                    next_ngram_str = next_ngram_str.strip('()').replace("'", "").replace(',', ' ')

                                next_words = next_ngram_str.split()

                                # Check if next ngram starts with current ngram
                                if len(next_words) >= len(current_words):
                                    if next_words[:len(current_words)] == current_words:
                                        # Create flow
                                        next_node_label = f"{class_name[:10]}: {next_ngram_str[:30]}"
                                        if len(next_ngram_str) > 30:
                                            next_node_label += "..."

                                        if next_node_label not in node_positions:
                                            nodes.append({
                                                'id': node_id,
                                                'label': next_node_label,
                                                'column': col_idx + 1,
                                                'class': class_name,
                                                'ngram': next_ngram_str,
                                                'n': next_n
                                            })
                                            node_positions[next_node_label] = node_id
                                            node_id += 1

                                        target_node_id = node_positions[next_node_label]

                                        # Get flow value (score)
                                        score_col = None
                                        for col in ['discriminative_score', 'score', 'combined_score', 'lift']:
                                            if col in next_pattern:
                                                score_col = col
                                                break

                                        flow_value = next_pattern.get(score_col, 1.0)
                                        if isinstance(flow_value, str):
                                            try:
                                                flow_value = float(flow_value)
                                            except ValueError:
                                                flow_value = 1.0

                                        source_target_flows.append({
                                            'source': current_node_id,
                                            'target': target_node_id,
                                            'value': max(flow_value, 0.1),  # Minimum visible flow
                                            'class': class_name,
                                            'color': self.class_colors.get(class_name, '#888888')
                                        })

        # Create class nodes at the rightmost column
        class_column_idx = len(n_sequence)  # Rightmost column after all n-gram columns
        class_nodes = {}
        
        for class_name in all_classes:
            class_node_label = f"CLASS: {class_name}"
            if class_node_label not in node_positions:
                nodes.append({
                    'id': node_id,
                    'label': class_node_label,
                    'column': class_column_idx,
                    'class': class_name,
                    'ngram': class_name,
                    'n': 'class',
                    'is_class_node': True
                })
                class_nodes[class_name] = node_id
                node_positions[class_node_label] = node_id
                node_id += 1
        
        # Connect ALL n-gram levels to their target classes
        # This creates flows from n-grams at all levels to their target classes
        for n in n_sequence:
            if n not in top_patterns:
                continue
            
            for class_name in all_classes:
                if class_name not in top_patterns[n]:
                    continue
                
                if class_name not in class_nodes:
                    continue
                
                class_node_id = class_nodes[class_name]
                patterns = top_patterns[n][class_name]
                
                # Connect all n-grams of this class to the class node
                for pattern in patterns:
                    ngram_col = 'ngram' if 'ngram' in pattern else 'pattern'
                    ngram_str = pattern.get(ngram_col, '')
                    
                    if not ngram_str:
                        continue
                    
                    # Convert ngram to string if needed
                    if isinstance(ngram_str, (tuple, list)):
                        ngram_str = ' '.join(str(x) for x in ngram_str)
                    elif isinstance(ngram_str, str) and ngram_str.startswith('('):
                        ngram_str = ngram_str.strip('()').replace("'", "").replace(',', ' ')
                    
                    # Find the corresponding n-gram node
                    ngram_node_label = f"{class_name[:10]}: {ngram_str[:30]}"
                    if len(ngram_str) > 30:
                        ngram_node_label += "..."
                    
                    if ngram_node_label in node_positions:
                        ngram_node_id = node_positions[ngram_node_label]
                        
                        # Get flow value (score)
                        score_col = None
                        for col in ['discriminative_score', 'score', 'combined_score', 'lift']:
                            if col in pattern:
                                score_col = col
                                break
                        
                        flow_value = pattern.get(score_col, 1.0)
                        if isinstance(flow_value, str):
                            try:
                                flow_value = float(flow_value)
                            except ValueError:
                                flow_value = 1.0
                        
                        # Create flow from n-gram to class
                        source_target_flows.append({
                            'source': ngram_node_id,
                            'target': class_node_id,
                            'value': max(flow_value, 0.1),
                            'class': class_name,
                            'color': self.class_colors.get(class_name, '#888888')
                        })

        print(f"  Created {len(nodes)} nodes and {len(source_target_flows)} flows")
        print(f"  Class nodes created: {len(class_nodes)}")
        
        # Report on classes
        classes_in_diagram = set(node['class'] for node in nodes)
        missing_classes = set(all_classes) - classes_in_diagram
        
        if missing_classes:
            print(f"\n  ⚠️  Classes found but not in diagram: {sorted(missing_classes)}")
            print(f"     (May have too few patterns or no sequential matches)")
        else:
            print(f"\n  ✅ All {len(all_classes)} classes are represented in the diagram")
        
        print(f"  📊 Classes in diagram: {sorted(classes_in_diagram)}")
        
        return {
            'nodes': nodes,
            'flows': source_target_flows,
            'classes': all_classes,
            'classes_in_diagram': classes_in_diagram,
            'missing_classes': missing_classes,
            'class_nodes': class_nodes
        }

    def create_sankey_diagram(self, top_k_per_class=15, output_file='sankey_npath_analysis.html'):
        """
        Create Sankey diagram showing sequential n-gram patterns
        """
        print("\n" + "=" * 80)
        print("CREATING SANKEY DIAGRAM - nPath Sequential Pattern Visualization")
        print("=" * 80)

        # Extract patterns
        pattern_data = self.extract_sequential_patterns(top_k_per_class)

        if len(pattern_data['nodes']) == 0:
            print("❌ No patterns found to visualize!")
            return None

        nodes = pattern_data['nodes']
        flows = pattern_data['flows']
        classes = pattern_data['classes']

        # Prepare data for Plotly Sankey
        node_labels = [node['label'] for node in nodes]
        # Validate and convert node colors to rgba format
        def hex_to_rgba_nodes(hex_color, alpha=0.8):
            """Convert hex color to rgba format for nodes"""
            if not hex_color or not isinstance(hex_color, str):
                return f"rgba(136, 136, 136, {alpha})"
            
            hex_color = hex_color.lstrip('#')
            
            if len(hex_color) == 8:
                # 8-character hex with alpha channel
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                alpha_hex = int(hex_color[6:8], 16)
                alpha = alpha_hex / 255.0
            elif len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            elif len(hex_color) == 3:
                r = int(hex_color[0] * 2, 16)
                g = int(hex_color[1] * 2, 16)
                b = int(hex_color[2] * 2, 16)
            else:
                return f"rgba(136, 136, 136, {alpha})"
            
            return f"rgba({r}, {g}, {b}, {alpha})"
        
        node_colors = []
        for node in nodes:
            color = self.class_colors.get(node['class'], '#888888')
            # Ensure color is valid hex and convert to rgba
            if not color or not isinstance(color, str):
                color = '#888888'
            # Convert to rgba format
            node_colors.append(hex_to_rgba_nodes(color, alpha=0.8))

        # Build source, target, value, and color arrays
        source = []
        target = []
        value = []
        link_colors = []

        for flow in flows:
            source.append(flow['source'])
            target.append(flow['target'])
            value.append(flow['value'])
            link_colors.append(flow['color'])

        # Convert hex colors to rgba format for transparency
        def hex_to_rgba(hex_color, alpha=0.5):
            """Convert hex color to rgba format"""
            if not hex_color or not isinstance(hex_color, str):
                return f"rgba(136, 136, 136, {alpha})"  # Default gray
            
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Handle different hex formats
            if len(hex_color) == 8:
                # 8-character hex with alpha channel (e.g., #1f77b480)
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                # Extract alpha from hex (0-255) and convert to 0-1 range
                alpha_hex = int(hex_color[6:8], 16)
                alpha = alpha_hex / 255.0
            elif len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            elif len(hex_color) == 3:
                r = int(hex_color[0] * 2, 16)
                g = int(hex_color[1] * 2, 16)
                b = int(hex_color[2] * 2, 16)
            else:
                # Invalid format, use default
                return f"rgba(136, 136, 136, {alpha})"
            
            return f"rgba({r}, {g}, {b}, {alpha})"

        # Validate and convert link colors
        validated_link_colors = []
        for color in link_colors:
            if not color or not isinstance(color, str):
                validated_link_colors.append("rgba(136, 136, 136, 0.5)")
            else:
                try:
                    validated_link_colors.append(hex_to_rgba(color, alpha=0.5))
                except Exception as e:
                    print(f"  ⚠️  Warning: Invalid color '{color}', using default: {e}")
                    validated_link_colors.append("rgba(136, 136, 136, 0.5)")
        
        # Ensure we have valid data
        if len(source) == 0 or len(target) == 0:
            print("❌ No flows to visualize! Check if sequential patterns exist.")
            return None
        
        if len(validated_link_colors) != len(source):
            print(f"⚠️  Warning: Color count ({len(validated_link_colors)}) doesn't match flow count ({len(source)})")
            # Pad or truncate colors to match
            if len(validated_link_colors) < len(source):
                validated_link_colors.extend(["rgba(136, 136, 136, 0.5)"] * (len(source) - len(validated_link_colors)))
            else:
                validated_link_colors = validated_link_colors[:len(source)]

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
                source=source,
                target=target,
                value=value,
                color=validated_link_colors
            )
        )])

        fig.update_layout(
            title_text="nPath Sequential Pattern Analysis: N-gram Flow → Target Classes",
            font_size=10,
            height=800,
            width=1400
        )

        # Save to HTML
        output_path = f'{self.results_dir}/{output_file}'
        fig.write_html(output_path)
        print(f"\n✅ Sankey diagram saved to: {output_path}")
        print(f"   Open in browser to view interactive visualization")

        return fig

    def create_class_specific_sankey(self, class_name, top_k=20, output_file=None):
        """
        Create Sankey diagram for a specific class only
        """
        if output_file is None:
            output_file = f'sankey_npath_{class_name[:20]}.html'

        print(f"\n🎯 Creating Sankey diagram for class: {class_name}")

        # Filter patterns for this class only
        pattern_data = self.extract_sequential_patterns(top_k_per_class=top_k)

        # Filter nodes and flows
        filtered_nodes = [n for n in pattern_data['nodes'] if n['class'] == class_name]
        filtered_flows = [f for f in pattern_data['flows'] if f['class'] == class_name]

        if len(filtered_nodes) == 0:
            print(f"❌ No patterns found for class: {class_name}")
            return None

        # Remap node IDs
        old_to_new = {node['id']: idx for idx, node in enumerate(filtered_nodes)}
        node_labels = [node['label'] for node in filtered_nodes]
        node_colors = [self.class_colors.get(class_name, '#888888')] * len(filtered_nodes)

        source = []
        target = []
        value = []
        link_colors = []

        for flow in filtered_flows:
            if flow['source'] in old_to_new and flow['target'] in old_to_new:
                source.append(old_to_new[flow['source']])
                target.append(old_to_new[flow['target']])
                value.append(flow['value'])
                link_colors.append(flow['color'])

        # Convert hex color to rgba for transparency
        def hex_to_rgba(hex_color, alpha=0.5):
            """Convert hex color to rgba format"""
            if not hex_color or not isinstance(hex_color, str):
                return f"rgba(136, 136, 136, {alpha})"  # Default gray
            
            hex_color = hex_color.lstrip('#')
            
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            elif len(hex_color) == 3:
                r = int(hex_color[0] * 2, 16)
                g = int(hex_color[1] * 2, 16)
                b = int(hex_color[2] * 2, 16)
            else:
                return f"rgba(136, 136, 136, {alpha})"
            
            return f"rgba({r}, {g}, {b}, {alpha})"

        class_color = self.class_colors.get(class_name, '#888888')
        link_color_rgba = hex_to_rgba(class_color, alpha=0.5)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[link_color_rgba for _ in link_colors]
            )
        )])

        fig.update_layout(
            title_text=f"nPath Sequential Patterns: {class_name}",
            font_size=10,
            height=800,
            width=1400
        )

        output_path = f'{self.results_dir}/{output_file}'
        fig.write_html(output_path)
        print(f"✅ Class-specific Sankey diagram saved to: {output_path}")

        return fig


def main():
    """Example usage"""
    visualizer = SankeyVisualizer(results_dir='colab_results')
    visualizer.load_ngram_data(n_values=[1, 2, 3, 4, 5])
    visualizer.create_sankey_diagram(top_k_per_class=15)


if __name__ == '__main__':
    main()


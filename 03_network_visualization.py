import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from pathlib import Path
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MIN_COOCCURRENCE = 1 
LIFT_THRESHOLD = 1.0
RAW_DATA_FILE = "cognitive_distortions_dataset.csv"
OUTPUT_DIR = "output_visualizations"

# Plotting aesthetics
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

NAME_MAPPING = {
    "Дихотомическое (чёрно-белое) мышление": "All-or-Nothing",
    'Дихотомическое мышление': 'All-or-Nothing',
    'Чёрно-белое мышление': 'All-or-Nothing',
    "Чрезмерное обобщение": "Overgeneralization",
    "Сверхобобщение": "Overgeneralization",
    "Мысленный фильтр": "Mental Filter",
    "Ментальная фильтрация": "Mental Filter",
    "Обесценивание позитивного": "Discounting",
    "Предсказание будущего": "Fortune Tell",
    "Чтение мыслей": "Mind Reading",
    "Катастрофизация": "Catastrophizing",
    "Эмоциональное обоснование": "Emotional",
    "Долженствование": "Should",
    "Навешивание ярлыков": "Labeling",
    "Персонализация": "Personalization",
    "Мышление жертвы": "Ext. Locus",
    "Туннельное мышление": "Tunnel Vision",
    "Сравнение": "Comparison",
    "Рационализация": "Rationalization",
    "Руминация": "Rumination",
    "Выученная беспомощность": "Helplessness",
    "Иллюзия справедливости": "Fairness"
}

# ------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ------------------------------------------------------------------

def compute_lift(pair_count, individual_counts, total_n):
    d1, d2, count = pair_count
    joint_prob = count / total_n
    prob_1 = individual_counts.get(d1, 0) / total_n
    prob_2 = individual_counts.get(d2, 0) / total_n
    if prob_1 == 0 or prob_2 == 0: return 0
    expected_prob = prob_1 * prob_2
    return joint_prob / expected_prob if expected_prob > 0 else 0

def load_and_build_graph():
    """
    Loads raw data, aggregates by English names, calculates Lift, 
    and builds the NetworkX graph.
    """
    print("Processing data and building network...")
    print(f"Filters: Count >= {MIN_COOCCURRENCE}, Lift >= {LIFT_THRESHOLD}")
    
    if not os.path.exists(RAW_DATA_FILE):
        print(f"❌ Critical Error: File '{RAW_DATA_FILE}' not found.")
        exit(1)
        
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', sep=';')
        grouped = df_raw.groupby('input_id')['название_искажения'].apply(list)
        N_TEXTS = len(grouped)
        
        individual_counts_en = Counter()
        pair_counts_en = Counter()
    
        # Aggregation Step
        for distortions_list_ru in grouped:
            unique_dists_en = set(NAME_MAPPING.get(d, d) for d in distortions_list_ru)
            
            for d_en in unique_dists_en:
                individual_counts_en[d_en] += 1
            
            for d1_en, d2_en in combinations(unique_dists_en, 2):
                pair_en = tuple(sorted([d1_en, d2_en]))
                pair_counts_en[pair_en] += 1
        
        G = nx.Graph()
        
        # Edge Creation
        for pair_en, count in pair_counts_en.items():
            if count < MIN_COOCCURRENCE: 
                continue
            
            d1_en, d2_en = pair_en
            lift = compute_lift((d1_en, d2_en, count), individual_counts_en, N_TEXTS)
            
            if lift >= LIFT_THRESHOLD:
                G.add_edge(d1_en, d2_en, lift=lift, count=count)
        
        # Node Attributes
        for node in G.nodes():
            freq = individual_counts_en.get(node, 0)
            G.nodes[node]['frequency'] = freq
        
        print(f"✓ Network ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)

# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------

def draw_network(G):
    print(f"\n🎨 Generating visualization...")
    
    fig = plt.figure(figsize=(16, 16), dpi=300)
    ax = plt.subplot(111)

    # Layout: Circular
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    n = len(sorted_nodes)
    radius = 4.0

    pos = {}
    for i, node in enumerate(sorted_nodes):
        angle = 2 * np.pi * i / n - np.pi / 2
        pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    # --- Draw Edges with Curvature ---
    for (u, v, d) in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lift = d['lift']
        
        # Style logic based on Lift strength
        if lift >= 1.8:
            linestyle = '-'
            color = '#000000'
            width = 3.0
            alpha = 0.85
        elif lift >= 1.5:
            linestyle = '--'
            color = '#202020'
            width = 2.5
            alpha = 0.75
        elif lift >= 1.3:
            linestyle = '-.'
            color = '#505050'
            width = 2.0
            alpha = 0.65
        else:
            linestyle = ':'
            color = '#808080'
            width = 1.5
            alpha = 0.50
        
        # Curvature calculation
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx ** 2 + dy ** 2)
        
        idx_u = sorted_nodes.index(u)
        idx_v = sorted_nodes.index(v)
        node_distance = min(abs(idx_u - idx_v), n - abs(idx_u - idx_v))
        
        if node_distance <= 2: curvature = -0.15
        elif node_distance <= 4: curvature = -0.20
        else: curvature = -0.25
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        if dist > 0:
            perp_x = -dy / dist * curvature * dist
            perp_y = dx / dist * curvature * dist
        else:
            perp_x, perp_y = 0, 0
        
        control_x = mid_x + perp_x
        control_y = mid_y + perp_y
        
        t = np.linspace(0, 1, 100)
        x_curve = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * control_x + t ** 2 * x2
        y_curve = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * control_y + t ** 2 * y2
        
        ax.plot(x_curve, y_curve, color=color, linestyle=linestyle, linewidth=width, alpha=alpha, zorder=1)

    # --- Draw Nodes ---
    node_freqs = nx.get_node_attributes(G, 'frequency').values()
    max_freq = max(node_freqs) if node_freqs else 1 

    for node in G.nodes():
        x, y = pos[node]
        freq = G.nodes[node]['frequency']
        degree = G.degree(node)
        
        # Size based on frequency
        size = 700 + (freq / max_freq) * 2000 
        
        # Shape based on degree (Core vs Peripheral)
        if degree >= 8:
            marker = 'o'
            facecolor = '#000000'
            edgecolor = '#000000'
            edgewidth = 3.5
        elif degree >= 6:
            marker = 'o'
            facecolor = '#606060'
            edgecolor = '#000000'
            edgewidth = 3.0
        elif degree >= 2:
            marker = 's'
            facecolor = 'white'
            edgecolor = '#000000'
            edgewidth = 2.5
        else: # degree = 1
            marker = '^'
            facecolor = 'white'
            edgecolor = '#707070'
            edgewidth = 2.5
        
        ax.scatter(x, y, s=size, marker=marker, facecolor=facecolor, edgecolor=edgecolor, linewidth=edgewidth, zorder=2)

    # --- Draw Labels ---
    label_radius = radius + 1.2
    for node in G.nodes():
        x, y = pos[node]
        angle = np.arctan2(y, x)
        
        text_x = label_radius * np.cos(angle)
        text_y = label_radius * np.sin(angle)
        
        ha = 'left' if x > 0.1 else 'right' if x < -0.1 else 'center'
        va = 'bottom' if y > 0.1 else 'top' if y < -0.1 else 'center'
        
        freq = G.nodes[node]['frequency']
        if freq > 50000:
            fontsize = 15
            fontweight = 'bold'
        elif freq > 30000:
            fontsize = 14
            fontweight = 'bold'
        else:
            fontsize = 13
            fontweight = 'normal'
        
        ax.text(text_x, text_y, node, fontsize=fontsize, ha=ha, va=va, weight=fontweight,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.95), zorder=3)

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # --- Legends ---
    legend_elements = [
        plt.Line2D([0], [0], marker=' ', color='w', label='Size = Frequency', markersize=0),
        plt.Line2D([0], [0], marker='o', color='w', label='Core (degree ≥ 8)', markerfacecolor='black', markersize=13),
        plt.Line2D([0], [0], marker='o', color='w', label='Core (degree 6-7)', markerfacecolor='#606060', markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Peripheral (degree 2-5)', markerfacecolor='white', markeredgecolor='black', markersize=11),
        plt.Line2D([0], [0], marker='^', color='w', label='Peripheral (degree 1)', markerfacecolor='white', markeredgecolor='#707070', markersize=11),
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=11.5, frameon=True, title='Node Attributes')

    line_elements = [
        plt.Line2D([0], [0], color='black', linewidth=3, linestyle='-', label='Strong (Lift ≥ 1.8)'),
        plt.Line2D([0], [0], color='#202020', linewidth=2.5, linestyle='--', label='Moderate (Lift 1.5-1.8)'),
        plt.Line2D([0], [0], color='#505050', linewidth=2, linestyle='-.', label='Weak (Lift 1.3-1.5)'),
        plt.Line2D([0], [0], color='#808080', linewidth=1.5, linestyle=':', label='Very weak (Lift 1.0-1.3)'),
    ]

    ax.legend(handles=line_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11.5, frameon=True, title='Connection Strength')

    # --- Save ---
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    filename = f'network_graph_{G.number_of_nodes()}nodes.png'
    output_path = output_dir / filename
    
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved image to: {output_path}")
    plt.close()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    G = load_and_build_graph()
    draw_network(G)
    print("\n" + "=" * 50)
    print("✓ Visualization Completed Successfully")
    print("=" * 50)
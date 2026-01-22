import pandas as pd
import numpy as np
import networkx as nx
import random
from itertools import combinations
from collections import Counter
from scipy.stats import pearsonr
import warnings
from multiprocessing import Pool, cpu_count
import time
import os

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

MIN_COOCCURRENCE = 1
LIFT_THRESHOLD = 1.0
N_BOOTSTRAPS = 100
RAW_DATA_FILE = "cognitive_distortions_dataset.csv"  # Ensure this file exists

# Mapping of Russian terms to standardized English codes
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

# Global dictionary for multiprocessing access
# Format: {input_id: {distortion_en, ...}}
id_to_dists_map = {}
unique_text_ids = []
N_TEXTS = 0

# ------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------

def load_data(filepath):
    """Loads and preprocesses the dataset."""
    global id_to_dists_map, unique_text_ids, N_TEXTS

    print(f"[!] Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File '{filepath}' not found.")
        exit(1)

    try:
        df_raw = pd.read_csv(filepath, encoding='utf-8-sig', sep=';')
        unique_text_ids = df_raw['input_id'].unique()
        N_TEXTS = len(unique_text_ids)
        print(f"✓ Loaded: {len(df_raw)} rows, {N_TEXTS} unique texts\n")
    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        exit(1)

    # Preprocessing: Map ID -> Set of English Distortion Names
    print("[!] Preprocessing data (ID mapping)...")
    df_grouped_raw = df_raw.groupby('input_id')['название_искажения'].apply(list)
    
    for input_id, distortions_list_ru in df_grouped_raw.items():
        # 1. Translate to English using NAME_MAPPING
        # 2. Keep only unique distortions per text (set)
        unique_dists_en = set(NAME_MAPPING.get(d, d) for d in distortions_list_ru)
        id_to_dists_map[input_id] = unique_dists_en
        
    print(f"✓ Created map for {len(id_to_dists_map)} unique texts.\n")


# ------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ------------------------------------------------------------------

def compute_lift(pair_count, individual_counts, total_n):
    """Calculates the Lift metric for a pair of distortions."""
    d1, d2, count = pair_count
    joint_prob = count / total_n
    
    prob_1 = individual_counts.get(d1, 0) / total_n
    prob_2 = individual_counts.get(d2, 0) / total_n
    
    if prob_1 == 0 or prob_2 == 0:
        return 0
    
    expected_prob = prob_1 * prob_2
    return joint_prob / expected_prob if expected_prob > 0 else 0

def analyze_id_list(id_list):
    """
    Analyzes a list of input_ids (supports duplicates for bootstrapping)
    and returns a list of graph edges.
    """
    total_n = len(id_list)
    if total_n == 0:
        return []
        
    individual_counts_en = Counter()
    pair_counts_en = Counter()
    
    # Iterate through IDs (N_TEXTS times for bootstrap)
    for input_id in id_list:
        unique_dists_en = id_to_dists_map.get(input_id, set())
        
        # Count individual frequencies
        for d_en in unique_dists_en:
            individual_counts_en[d_en] += 1
        
        # Count pair frequencies
        for d1_en, d2_en in combinations(unique_dists_en, 2):
            pair_en = tuple(sorted([d1_en, d2_en]))
            pair_counts_en[pair_en] += 1
            
    # Calculate Lift and Filter
    edges = []
    for pair_en, count in pair_counts_en.items():
        # FILTER 1: Minimum Co-occurrence
        if count < MIN_COOCCURRENCE:
            continue
        
        d1_en, d2_en = pair_en
        lift = compute_lift((d1_en, d2_en, count), individual_counts_en, total_n)
        
        # FILTER 2: Lift Threshold
        if lift >= LIFT_THRESHOLD:
            edges.append((d1_en, d2_en, lift, count))
    
    return edges

def build_graph_from_edges(edges_list):
    """Constructs a NetworkX graph from a list of edges."""
    G = nx.Graph()
    for u, v, lift, count in edges_list:
        G.add_edge(u, v, lift=lift, count=count)
    return G

def bootstrap_iteration(iteration_idx):
    """Single bootstrap iteration (designed for multiprocessing)."""
    
    # 1. Resampling with replacement
    resampled_ids = random.choices(unique_text_ids, k=N_TEXTS)
    
    # 2. Analyze resampled data
    edges = analyze_id_list(resampled_ids)
    G_boot = build_graph_from_edges(edges)
    
    # 3. Return metrics
    return {
        'density': nx.density(G_boot),
        'n_edges': G_boot.number_of_edges(),
        'n_nodes': G_boot.number_of_nodes(),
        'clustering': nx.average_clustering(G_boot) if G_boot.number_of_edges() > 0 else 0,
        'nodes': sorted(G_boot.nodes())
    }

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    
    print("=" * 80)
    print(f"🚀 NETWORK STATISTICAL VALIDATION")
    print(f"   Config: Min Cooc={MIN_COOCCURRENCE}, Bootstraps={N_BOOTSTRAPS}")
    print("=" * 80)
    
    start_time = time.time()
    
    # 0. Load Data
    load_data(RAW_DATA_FILE)
    
    # STEP 1: Analyze Original Data
    print("\n[1] Analyzing Original Data...")
    
    original_edges = analyze_id_list(unique_text_ids) 
    G_original = build_graph_from_edges(original_edges)
    
    original_density = nx.density(G_original)
    original_clustering = nx.average_clustering(G_original)
    original_nodes = sorted(G_original.nodes())
    
    print(f"\n✓ Original Network Stats:")
    print(f"  • Nodes: {G_original.number_of_nodes()}")
    print(f"  • Edges: {G_original.number_of_edges()}")
    print(f"  • Density: {original_density:.3f}")
    print(f"  • Clustering Coeff: {original_clustering:.3f}")
    
    print(f"\n  📊 TOP-10 EDGES (by count):")
    edges_with_counts = [(u, v, d['lift'], d['count']) for u, v, d in G_original.edges(data=True)]
    edges_sorted = sorted(edges_with_counts, key=lambda x: x[3], reverse=True)[:10]
    for i, (u, v, lift, count) in enumerate(edges_sorted, 1):
        print(f"     {i:2d}. {u:20s} — {v:20s}  count={count:4d}, lift={lift:.2f}")
    
    # STEP 2: Bootstrap Analysis
    print(f"\n[2] Running Bootstrap ({N_BOOTSTRAPS} iterations on {cpu_count()} cores)...")
    print("    Progress: ", end="", flush=True)
    
    boot_start = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(bootstrap_iteration, range(N_BOOTSTRAPS)), 1):
            results.append(result)
            if i % 10 == 0:  # Update every 10 iterations
                print(f"{i}...", end="", flush=True)
    
    boot_end = time.time()
    print(f"\n    Done in {boot_end - boot_start:.1f} sec")
    
    # Analyze Bootstrap Results
    bootstrap_densities = [r['density'] for r in results]
    bootstrap_n_edges = [r['n_edges'] for r in results]
    bootstrap_n_nodes = [r['n_nodes'] for r in results]
    bootstrap_clustering = [r['clustering'] for r in results if r['clustering'] > 0]
    
    ci_low = np.percentile(bootstrap_densities, 2.5)
    ci_high = np.percentile(bootstrap_densities, 97.5)
    
    # Node Stability Analysis
    node_frequency = Counter()
    for result in results:
        for node in result['nodes']:
            node_frequency[node] += 1
    
    print("\n" + "!" * 80)
    print("  RESULTS FOR PAPER (Bootstrap)")
    print("!" * 80)
    print(f"  Original density: {original_density:.3f}")
    print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  Mean Density: {np.mean(bootstrap_densities):.3f} (SD={np.std(bootstrap_densities):.3f})")
    
    print(f"\n  Nodes (Mean): {np.mean(bootstrap_n_nodes):.1f} (SD={np.std(bootstrap_n_nodes):.1f})")
    print(f"  Edges (Mean): {np.mean(bootstrap_n_edges):.1f} (SD={np.std(bootstrap_n_edges):.1f})")
    
    if bootstrap_clustering:
        ci_clust_low = np.percentile(bootstrap_clustering, 2.5)
        ci_clust_high = np.percentile(bootstrap_clustering, 97.5)
        print(f"\n  Clustering (Orig): {original_clustering:.3f}")
        print(f"  Bootstrap 95% CI: [{ci_clust_low:.3f}, {ci_clust_high:.3f}]")
    
    print(f"\n  📌 NODE STABILITY (Appearance in {N_BOOTSTRAPS} iterations):")
    for node, freq in sorted(node_frequency.items(), key=lambda x: x[1], reverse=True):
        percent = (freq / N_BOOTSTRAPS) * 100
        marker = "✓" if percent >= 95 else "⚠" if percent >= 80 else "✗"
        print(f"     {marker} {node:25s} {freq:4d}/{N_BOOTSTRAPS} ({percent:5.1f}%)")
    
    print("!" * 80)
    
    # STEP 3: Split-Half Reliability
    print("\n[3] Running Split-Half Reliability Test...")
    
    random.seed(42)  # For reproducibility
    
    shuffled_ids = list(unique_text_ids)
    random.shuffle(shuffled_ids)
    
    split_point = N_TEXTS // 2
    ids_half1 = shuffled_ids[:split_point]
    ids_half2 = shuffled_ids[split_point:]
    
    print(f"  Split: {len(ids_half1)} vs {len(ids_half2)} texts")
    
    edges1 = analyze_id_list(ids_half1)
    edges2 = analyze_id_list(ids_half2)
    
    G1 = build_graph_from_edges(edges1)
    G2 = build_graph_from_edges(edges2)
    
    all_nodes = sorted(set(G1.nodes()) | set(G2.nodes()))
    
    G1_full = G1.copy()
    G2_full = G2.copy()
    
    # Ensure both graphs have same nodes for matrix comparison
    for node in all_nodes:
        if node not in G1_full: G1_full.add_node(node)
        if node not in G2_full: G2_full.add_node(node)
    
    # QAP Correlation (Quadratic Assignment Procedure style correlation)
    M1 = nx.to_numpy_array(G1_full, nodelist=all_nodes, weight='lift')
    M2 = nx.to_numpy_array(G2_full, nodelist=all_nodes, weight='lift')
    
    # Extract upper triangle elements (excluding diagonal)
    vec1 = M1[np.triu_indices_from(M1, k=1)]
    vec2 = M2[np.triu_indices_from(M2, k=1)]
    
    if len(vec1) > 0 and len(vec2) > 0:
        correlation, p_value = pearsonr(vec1, vec2)
        
        print("\n" + "!" * 80)
        print("  RESULTS FOR PAPER (Split-Half)")
        print("!" * 80)
        print(f"  QAP Correlation (r): {correlation:.3f}")
        print(f"  P-value: {p_value:.5f}")
        print("!" * 80)
    else:
        print("\n⚠️ Not enough edges to calculate correlation.")
    
    end_time = time.time()
    print(f"\n{'='*80}")
    print(f"🏁 COMPLETED in {end_time - start_time:.1f} sec")
    print(f"{'='*80}")
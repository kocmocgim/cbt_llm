import pandas as pd
import numpy as np
import networkx as nx
import random
from itertools import combinations
from collections import Counter
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MIN_COOCCURRENCE = 1
LIFT_THRESHOLD = 1.0
RAW_DATA_FILE = "cognitive_distortions_dataset.csv"  # Ensure this file exists
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # 0% to 50%
N_ITERATIONS = 1000  # Number of repetitions per noise level
N_CORES = cpu_count()

# Standardized mapping
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
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def compute_lift(pair_count, individual_counts, total_n):
    """Calculates Lift metric."""
    d1, d2, count = pair_count
    joint_prob = count / total_n
    prob_1 = individual_counts.get(d1, 0) / total_n
    prob_2 = individual_counts.get(d2, 0) / total_n
    
    if prob_1 == 0 or prob_2 == 0:
        return 0
    
    expected_prob = prob_1 * prob_2
    return joint_prob / expected_prob if expected_prob > 0 else 0

def analyze_id_list(id_list, distortions_dict):
    """Builds edge list from a list of IDs and a dictionary of distortions."""
    total_n = len(id_list)
    if total_n == 0:
        return []
    
    individual_counts = Counter()
    pair_counts = Counter()
    
    for input_id in id_list:
        unique_dists = distortions_dict.get(input_id, set())
        for d in unique_dists:
            individual_counts[d] += 1
        for d1, d2 in combinations(unique_dists, 2):
            pair = tuple(sorted([d1, d2]))
            pair_counts[pair] += 1
    
    edges = []
    for pair, count in pair_counts.items():
        if count < MIN_COOCCURRENCE:
            continue
        d1, d2 = pair
        lift = compute_lift((d1, d2, count), individual_counts, total_n)
        if lift >= LIFT_THRESHOLD:
            edges.append((d1, d2, lift, count))
    
    return edges

def build_graph(edges_list):
    """Constructs NetworkX graph."""
    G = nx.Graph()
    for u, v, lift, count in edges_list:
        G.add_edge(u, v, lift=lift, count=count)
    return G

def add_noise(distortions_dict, all_distortions, noise_level, seed=None):
    """
    Simulates annotation errors by randomly modifying labels.
    
    Mechanism:
    1. Removal (False Negatives): Randomly remove N labels based on noise level.
    2. Addition (False Positives): Randomly add N labels based on noise level.
    
    NOTE: Uses round() instead of int() to strictly respect 0% noise 
    and proportional changes for small label sets.
    """
    if seed is not None:
        random.seed(seed)
    
    # If noise is 0, return exact copy without changes
    if noise_level == 0.0:
        return {k: set(v) for k, v in distortions_dict.items()}
    
    noisy_dict = {}
    for input_id, dists in distortions_dict.items():
        noisy_dists = set(dists)
        n_dists = len(noisy_dists)

        # Calculate number of items to remove
        n_to_remove = int(round(n_dists * noise_level))
        
        if n_to_remove > 0 and n_dists > 0:
            # Prevent removing all labels unless noise is 100%
            if n_to_remove >= n_dists and noise_level < 1.0:
                n_to_remove = n_dists - 1
            
            if n_to_remove > 0:
                to_remove = random.sample(list(noisy_dists), n_to_remove)
                noisy_dists -= set(to_remove)
        
        # Calculate number of items to add
        n_to_add = int(round(len(dists) * noise_level))
        
        possible_additions = [d for d in all_distortions if d not in noisy_dists]
        if possible_additions and n_to_add > 0:
            to_add = random.choices(possible_additions, k=min(n_to_add, len(possible_additions)))
            noisy_dists.update(to_add)
        
        noisy_dict[input_id] = noisy_dists
    
    return noisy_dict

def process_single_iteration(args):
    """Worker function for parallel processing."""
    noise_level, iteration_idx, text_ids, dists_map, all_dists, orig_hubs = args
    
    # Unique seed per iteration
    seed = hash((noise_level, iteration_idx)) % (2**32)
    
    noisy_data = add_noise(dists_map, all_dists, noise_level, seed=seed)
    
    edges = analyze_id_list(text_ids, noisy_data)
    G = build_graph(edges)
    
    # Metrics
    if G.number_of_nodes() > 0:
        degree_dict = dict(G.degree())
        top_hubs_noisy = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        hubs_noisy = [node for node, _ in top_hubs_noisy]
    else:
        hubs_noisy = []
    
    # Jaccard similarity for Hub Stability
    set_orig = set(orig_hubs)
    set_noisy = set(hubs_noisy)
    union = len(set_orig | set_noisy)
    jaccard = len(set_orig & set_noisy) / union if union > 0 else 0
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
        'hub_jaccard': jaccard
    }

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == '__main__':
    print(f"[!] Loading data from: {RAW_DATA_FILE}")
    
    if not os.path.exists(RAW_DATA_FILE):
        print(f"❌ ERROR: File '{RAW_DATA_FILE}' not found.")
        exit(1)

    df_raw = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', sep=';')
    unique_text_ids = df_raw['input_id'].unique()
    N_TEXTS = len(unique_text_ids)
    print(f"✓ Loaded: {len(df_raw)} rows, {N_TEXTS} texts")
    print(f"✓ CPU Cores Available: {N_CORES}\n")

    # Preprocessing
    df_grouped_raw = df_raw.groupby('input_id')['название_искажения'].apply(list)
    id_to_dists_map = {}
    for input_id, distortions_list_ru in df_grouped_raw.items():
        unique_dists_en = set(NAME_MAPPING.get(d, d) for d in distortions_list_ru)
        id_to_dists_map[input_id] = unique_dists_en

    all_distortions = list(set(NAME_MAPPING.values()))
    
    # ------------------------------------------------------------------
    # 1. BASELINE ANALYSIS (0% NOISE)
    # ------------------------------------------------------------------
    print("[1] Establishing Baseline (0% Noise)...")
    original_edges = analyze_id_list(unique_text_ids, id_to_dists_map)
    G_original = build_graph(original_edges)

    original_metrics = {
        'nodes': G_original.number_of_nodes(),
        'edges': G_original.number_of_edges(),
        'density': nx.density(G_original),
        'clustering': nx.average_clustering(G_original) if G_original.number_of_edges() > 0 else 0
    }

    degree_dict = dict(G_original.degree())
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    original_hubs = [node for node, _ in top_hubs]

    print(f"✓ Baseline Stats:")
    print(f"  Density: {original_metrics['density']:.3f}")
    print(f"  Top Hubs: {', '.join(original_hubs)}\n")

    # ------------------------------------------------------------------
    # 2. SENSITIVITY TEST
    # ------------------------------------------------------------------
    print(f"[2] Running Sensitivity Analysis ({len(NOISE_LEVELS)} levels x {N_ITERATIONS} iter)...")
    
    results = []

    for noise in NOISE_LEVELS:
        print(f"\n  Processing Noise Level: {noise*100:>3.0f}%")
        
        args_list = [
            (noise, i, unique_text_ids, id_to_dists_map, all_distortions, original_hubs) 
            for i in range(N_ITERATIONS)
        ]
        
        with Pool(processes=N_CORES) as pool:
            iteration_metrics = list(
                tqdm(
                    pool.imap(process_single_iteration, args_list),
                    total=N_ITERATIONS,
                    desc=f"    Progress",
                    leave=False
                )
            )
        
        avg_metrics = {
            'noise_level': noise,
            'nodes_mean': np.mean([m['nodes'] for m in iteration_metrics]),
            'nodes_std': np.std([m['nodes'] for m in iteration_metrics]),
            'edges_mean': np.mean([m['edges'] for m in iteration_metrics]),
            'edges_std': np.std([m['edges'] for m in iteration_metrics]),
            'density_mean': np.mean([m['density'] for m in iteration_metrics]),
            'density_std': np.std([m['density'] for m in iteration_metrics]),
            'clustering_mean': np.mean([m['clustering'] for m in iteration_metrics]),
            'clustering_std': np.std([m['clustering'] for m in iteration_metrics]),
            'hub_jaccard_mean': np.mean([m['hub_jaccard'] for m in iteration_metrics]),
            'hub_jaccard_std': np.std([m['hub_jaccard'] for m in iteration_metrics])
        }
        
        results.append(avg_metrics)
        print(f"    ✓ Density: {avg_metrics['density_mean']:.3f} (±{avg_metrics['density_std']:.3f}) | Hub Jaccard: {avg_metrics['hub_jaccard_mean']:.2f}")

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    print("\n" + "="*100)
    print("SENSITIVITY ANALYSIS REPORT")
    print("="*100)
    print(f"{'Noise':<8} {'Density':<18} {'Clustering':<18} {'Hub Stability (Jaccard)':<18}")
    print("-"*100)

    for r in results:
        print(f"{r['noise_level']*100:>5.0f}%  "
              f"{r['density_mean']:>6.3f}±{r['density_std']:<8.3f}  "
              f"{r['clustering_mean']:>6.3f}±{r['clustering_std']:<8.3f}  "
              f"{r['hub_jaccard_mean']:>6.3f}±{r['hub_jaccard_std']:<8.3f}")
    print("="*100)
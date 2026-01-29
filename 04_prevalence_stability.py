import pandas as pd
import numpy as np
import random
from itertools import combinations
from collections import Counter
from multiprocessing import Pool, cpu_count
import os
import time
from tqdm import tqdm  # Requires: pip install tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
RAW_DATA_FILE = "cognitive_distortions_dataset.csv"
N_BOOTSTRAPS = 10000        # Number of iterations
MIN_COOCCURRENCE = 1        # Minimum pair frequency
LIFT_THRESHOLD = 1.0        # Association threshold (Lift > 1.0)
RANDOM_SEED = 42            # Fixed seed for reproducibility

# Classification Thresholds
THRESH_STABLE = 60.0        # >= 60%: Stable core
THRESH_BORDERLINE = 50.0    # 50-60%: Borderline (exploratory)

# Mapping: Dataset Terms (RU) -> Standardized Codes (EN)
NAME_MAPPING = {
    "Дихотомическое (чёрно-белое) мышление": "All-or-Nothing",
    'Дихотомическое мышление': 'All-or-Nothing',
    'Чёрно-белое мышление': 'All-or-Nothing',
    "Чрезмерное обобщение": "Overgeneralization",
    "Сверхобобщение": "Overgeneralization",
    "Мысленный фильтр": "Mental Filter",
    "Ментальная фильтрация": "Mental Filter",
    "Обесценивание позитивного": "Discounting Positive",
    "Предсказание будущего": "Fortune Telling",
    "Чтение мыслей": "Mind Reading",
    "Катастрофизация": "Catastrophizing",
    "Эмоциональное обоснование": "Emotional Reasoning",
    "Долженствование": "Should Statements",
    "Навешивание ярлыков": "Labeling",
    "Персонализация": "Personalization",
    "Мышление жертвы": "External Locus",
    "Туннельное мышление": "Tunnel Vision",
    "Сравнение": "Social Comparison",
    "Рационализация": "Rationalization",
    "Руминация": "Rumination",
    "Выученная беспомощность": "Learned Helplessness",
    "Иллюзия справедливости": "Fairness Fallacy"
}

# --- GLOBAL VARIABLES (Multiprocessing) ---
worker_data_map = None
worker_text_ids = None
worker_n_texts = None

def init_worker(data_map, text_ids, n_texts):
    """Initialize worker process memory (Windows-compatible)."""
    global worker_data_map, worker_text_ids, worker_n_texts
    worker_data_map = data_map
    worker_text_ids = text_ids
    worker_n_texts = n_texts

def load_data():
    """Load and validate dataset."""
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: File {RAW_DATA_FILE} not found.")
        exit(1)
    
    print(f"Loading {RAW_DATA_FILE}...")
    try:
        df = pd.read_csv(RAW_DATA_FILE, encoding='utf-8-sig', sep=';')
        required = {'input_id', 'название_искажения'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns. Required: {required}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit(1)
    
    # Grouping: 1 document = set of unique distortions
    grouped = df.groupby('input_id')['название_искажения'].apply(list)
    unique_ids = list(grouped.index)
    n_texts = len(unique_ids)
    
    data_map = {}
    for input_id, dists in grouped.items():
        data_map[input_id] = set(NAME_MAPPING.get(d, d) for d in dists)
    
    print(f"Loaded {n_texts} unique texts.")
    return data_map, unique_ids, n_texts

def calculate_lift(n_cooc, n_doc1, n_doc2, total_docs):
    """
    Calculate Lift = P(AB) / (P(A) * P(B)).
    Based on document counts.
    """
    if n_doc1 == 0 or n_doc2 == 0:
        return 0.0
    
    p_joint = n_cooc / total_docs
    p_d1 = n_doc1 / total_docs
    p_d2 = n_doc2 / total_docs
    
    return p_joint / (p_d1 * p_d2)

def get_network_nodes(id_list):
    """
    Construct network on the sample and return nodes passing the Lift filter.
    """
    total_n = len(id_list)
    doc_freqs = Counter()
    cooc_freqs = Counter()
    
    # 1. Counts (including duplicates from bootstrap)
    for input_id in id_list:
        dists = worker_data_map.get(input_id, set())
        
        for d in dists:
            doc_freqs[d] += 1
            
        for d1, d2 in combinations(dists, 2):
            pair = tuple(sorted([d1, d2]))
            cooc_freqs[pair] += 1
            
    # 2. Filtering
    active_nodes = set()
    for pair, count in cooc_freqs.items():
        if count < MIN_COOCCURRENCE:
            continue
        
        d1, d2 = pair
        lift = calculate_lift(
            n_cooc=count,
            n_doc1=doc_freqs[d1],
            n_doc2=doc_freqs[d2],
            total_docs=total_n
        )
        
        if lift >= LIFT_THRESHOLD:
            active_nodes.add(d1)
            active_nodes.add(d2)
            
    return list(active_nodes)

def bootstrap_worker(iteration_idx):
    """Single bootstrap iteration."""
    # Deterministic seeding per process/iteration
    seed = RANDOM_SEED + iteration_idx
    random.seed(seed)
    np.random.seed(seed)
    
    # Resampling with replacement
    resampled_ids = random.choices(worker_text_ids, k=worker_n_texts)
    return get_network_nodes(resampled_ids)

if __name__ == "__main__":
    # 1. Load Data
    data_map, unique_ids, n_texts = load_data()
    
    # 2. Calculate Prevalence (Original Data)
    print("\nCalculating Prevalence (Base Rates)...")
    total_doc_counts = Counter()
    for dists in data_map.values():
        total_doc_counts.update(dists)
        
    stats = []
    for dist, count in total_doc_counts.items():
        stats.append({
            "Distortion": dist,
            "Count": count,
            "Prevalence": (count / n_texts) * 100
        })
    
    # 3. Calculate Stability (Bootstrap)
    print(f"Starting Bootstrap ({N_BOOTSTRAPS} iterations, Seed={RANDOM_SEED})...")
    start_time = time.time()
    
    node_stability_counts = Counter()
    
    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(data_map, unique_ids, n_texts)
    ) as pool:
        
        # Using tqdm for progress visualization
        iterator = pool.imap_unordered(bootstrap_worker, range(N_BOOTSTRAPS), chunksize=50)
        for nodes in tqdm(iterator, total=N_BOOTSTRAPS, desc="Bootstrap Progress"):
            node_stability_counts.update(nodes)
            
    print(f"Bootstrap finished in {time.time() - start_time:.1f} sec.")
    
    # 4. Output Results
    print("\n" + "="*85)
    print(f"{'Distortion':<30} | {'Count':<8} | {'Prev %':<8} | {'Stability %':<12} | {'Status'}")
    print("-" * 85)
    
    stats.sort(key=lambda x: x["Count"], reverse=True)
    
    for row in stats:
        dist = row["Distortion"]
        count = row["Count"]
        prev = row["Prevalence"]
        
        stab_count = node_stability_counts.get(dist, 0)
        stability_pct = (stab_count / N_BOOTSTRAPS) * 100
        
        # Classification
        if stability_pct >= THRESH_STABLE:
            status = "✅ Stable"
        elif stability_pct >= THRESH_BORDERLINE:
            status = "⚠️ Borderline"
        else:
            status = "❌ Excluded"
        
        print(f"{dist:<30} | {count:<8} | {prev:<8.1f} | {stability_pct:<12.1f} | {status}")

    print("="*85)
    print("Prevalence: % of texts containing the distortion.")
    print("Stability: Frequency of inclusion (Lift > 1.0) across bootstrap iterations.")
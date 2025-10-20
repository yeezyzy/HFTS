#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This script merges, refactors, and enhances the functionality for building various graph structures 
from Temporal Knowledge Graph (TKG) datasets. It is designed to be the single source of truth for
data preprocessing and graph construction.

Key Features:
- **Data Loading & Vocabulary Creation**: Loads raw TKG data (train, valid, test splits) and builds
  comprehensive vocabularies for entities, relations, and timestamps.
- **Multiple Graph Types**: Constructs several types of graphs required for different TKG models:
  1.  **Global Graph**: A static graph containing all facts, ignoring timestamps.
  2.  **Temporal Graph**: A time-aware graph where each entity at each timestamp is a distinct node.
      Edges connect facts within a specified time window, capturing temporal dynamics.
  3.  **Window Graph**: Connects nodes representing the same entity across adjacent timestamps, often
      used for self-supervised learning tasks.
  4.  **Relation Graph**: A graph where nodes are relations. An edge between two relations indicates
      they appear sequentially in time.
- **File Caching**: Checks for the existence of output files before generation and skips the process
  if a file is already present, saving computation time.
- **Encapsulation**: Logic is organized into modular functions for clarity, maintainability, and reusability.
- **Parallel Processing**: Supports parallel graph construction to accelerate the preprocessing pipeline
  on multi-core systems.
"""

import os
import copy
import argparse
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

import dgl
import torch
import numpy as np
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build various graphs for TKGC models")
    parser.add_argument("--dataset", default="ICEWS14",
                        choices=["ICEWS14", "ICEWS0515", "YAGO11k", "wikidata", "GDELT", "ICEWS14_6", "ICEWS14_12", "ICEWS14_18"],
                        help="Dataset folder name, which must contain train.txt, valid.txt, and test.txt.")
    parser.add_argument("--window_size", default=12, type=int, 
                        help="Half window size for temporal and window graph construction (w).")
    parser.add_argument("--parallel", action="store_true", 
                        help="Enable parallel graph building using multiple processes.")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of worker processes for parallel mode. Defaults to the number of CPU cores.")
    return parser.parse_args()


def load_raw_data(dataset: str):
    """Loads raw quadruples from train, valid, and test files."""
    print(f"[+] Loading raw data for dataset: {dataset}")
    train_path = os.path.join(dataset, "train.txt")
    valid_path = os.path.join(dataset, "valid.txt")
    test_path = os.path.join(dataset, "test.txt")

    def read_quadruples(file_path):
        with open(file_path, "r", encoding="UTF-8") as f:
            return [line.strip().lower().split("\t") for line in f]

    train_q = read_quadruples(train_path)
    valid_q = read_quadruples(valid_path)
    test_q = read_quadruples(test_path)
    
    return train_q, valid_q, test_q


def build_vocabs_and_ids(train_q, valid_q, test_q):
    """Builds vocabularies and maps quadruples to integer IDs."""
    print("[+] Building vocabularies and mapping to IDs.")
    # Training set vocabs
    train_heads, train_rels, train_tails, train_times = zip(*train_q)
    train_entities = sorted(list(set(train_heads) | set(train_tails)))
    
    # Combined dataset vocabs (total)
    all_quads = train_q + valid_q + test_q
    total_heads, total_rels, total_tails, total_times = zip(*all_quads)
    total_entities = sorted(list(set(total_heads) | set(total_tails)))
    total_relations = sorted(list(set(total_rels)))
    total_timestamps = sorted(list(set(total_times)))
    
    vocabs = {
        "train": {"E_vocab": train_entities},
        "total": {"E_vocab": total_entities, "R_vocab": total_relations, "T_vocab": total_timestamps}
    }

    # Map to IDs
    # Training IDs are relative to the training entity vocab, but use total relation/time vocabs
    train_ids = {
        "h": [train_entities.index(h) for h in train_heads],
        "r": [total_relations.index(r) for r in train_rels],
        "t": [train_entities.index(t) for t in train_tails],
        "ts": [total_timestamps.index(ts) for ts in train_times],
    }
    
    # Total IDs are relative to the total vocabs
    total_ids = {
        "h": [total_entities.index(h) for h in total_heads],
        "r": [total_relations.index(r) for r in total_rels],
        "t": [total_entities.index(t) for t in total_tails],
        "ts": [total_timestamps.index(ts) for ts in total_times],
    }
    
    return vocabs, {"train": train_ids, "total": total_ids}


def construct_global_graphs(dataset: str, ids, vocabs):
    """Constructs and saves static global graphs for train and test sets."""
    train_path = f"./{dataset}_Train_Global_Graph.bin"
    test_path = f"./{dataset}_Test_Global_Graph.bin"

    num_train_ent = len(vocabs['train']['E_vocab'])
    num_total_ent = len(vocabs['total']['E_vocab'])
    num_total_rel = len(vocabs['total']['R_vocab'])

    # Build and save the test (total) global graph
    if os.path.exists(test_path):
        print(f"Skipping existing file: {test_path}")
    else:
        print(f"Building Test Global Graph: {test_path}")
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_total_ent)
        g.add_edges(ids["total"]["h"], ids["total"]["t"])
        g.add_edges(ids["total"]["t"], ids["total"]["h"])  # Add reciprocal edges
        g.ndata["node_idx"] = torch.arange(num_total_ent, dtype=torch.long)
        
        relations = torch.tensor(ids["total"]["r"], dtype=torch.long)
        reciprocal_relations = relations + num_total_rel
        g.edata["relation_idx"] = torch.cat([relations, reciprocal_relations])
        save_graphs(test_path, [g])

    # Build and save the train global graph
    if os.path.exists(train_path):
        print(f"Skipping existing file: {train_path}")
    else:
        print(f"Building Train Global Graph: {train_path}")
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_train_ent)
        g.add_edges(ids["train"]["h"], ids["train"]["t"])
        g.add_edges(ids["train"]["t"], ids["train"]["h"])  # Add reciprocal edges
        
        # Map train entity IDs to their corresponding IDs in the total vocabulary
        g.ndata['node_idx'] = torch.tensor(
            [vocabs['total']['E_vocab'].index(e) for e in vocabs['train']['E_vocab']], dtype=torch.long
        )
        
        relations = torch.tensor(ids["train"]["r"], dtype=torch.long)
        reciprocal_relations = relations + num_total_rel
        g.edata['relation_idx'] = torch.cat([relations, reciprocal_relations])
        save_graphs(train_path, [g])


def _construct_temporal_graph_split(split_name: str, dataset: str, w: int, ids, vocabs):
    """Helper function to build a single split (train or test) of the temporal graph."""
    is_train = split_name == "Train"
    output_path = f"./window_dynamic_{w}_{dataset}_{split_name}_time_split_absolute_Graph.bin"

    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return

    print(f"Building {split_name} Temporal Graph: {output_path}")
    
    # Select the appropriate vocabs and IDs for the split
    v_split = vocabs['train'] if is_train else vocabs['total']
    id_split = ids['train'] if is_train else ids['total']
    num_entities = len(v_split['E_vocab'])
    
    num_times = len(vocabs['total']['T_vocab'])
    num_rels = len(vocabs['total']['R_vocab'])
    
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(num_entities * num_times)

    rel_stack, win_stack, abs_time_stack, diff_time_stack = [], [], [], []

    # Process forward and reciprocal edges
    for reciprocal in [False, True]:
        for h, r, t, ts in zip(id_split["h"], id_split["r"], id_split["t"], id_split["ts"]):
            src_node, target_node = (h, t) if not reciprocal else (t, h)
            rel = r if not reciprocal else r + num_rels
            
            # Determine the time window for connections, handling boundary cases
            start_ts = max(0, ts - w)
            end_ts = min(num_times, ts + w + 1)
            
            num_edges = end_ts - start_ts
            target_timestamps = list(range(start_ts, end_ts))
            
            src_nodes = np.array([num_times * src_node + ts] * num_edges)
            dst_nodes = np.array(target_timestamps) + np.array([num_times * target_node])

            g.add_edges(src_nodes, dst_nodes)
            
            # Append edge features
            rel_stack.extend([rel] * num_edges)
            abs_time_stack.extend([ts] * num_edges)
            diff_time_stack.extend(target_timestamps)
            
            # Calculate window size relative to the event time `ts`
            win_sizes = [abs(i - ts) for i in target_timestamps]
            win_stack.extend(win_sizes)

    # Assign node and edge data
    if is_train:
        train_e_total_ids = torch.tensor([vocabs["total"]["E_vocab"].index(x) for x in v_split["E_vocab"]], dtype=torch.long)
        node_idx_base = (train_e_total_ids * num_times).unsqueeze(1)
        g.ndata['node_idx'] = (node_idx_base + torch.arange(num_times, dtype=torch.long)).view(-1)
        g.ndata['entity_idx'] = train_e_total_ids.unsqueeze(1).repeat(1, num_times).view(-1)
    else:
        g.ndata['node_idx'] = torch.arange(num_entities * num_times, dtype=torch.long)
        g.ndata['entity_idx'] = torch.arange(num_entities, dtype=torch.long).unsqueeze(1).repeat(1, num_times).view(-1)

    g.edata['relation_idx'] = torch.tensor(rel_stack, dtype=torch.long)
    g.edata['window_size'] = torch.tensor(win_stack, dtype=torch.long)
    g.edata['absolute_time'] = torch.tensor(abs_time_stack, dtype=torch.long)
    g.edata['absolute_diffusion_time'] = torch.tensor(diff_time_stack, dtype=torch.long)
    
    save_graphs(output_path, [g])


def construct_temporal_graphs(dataset: str, w: int, ids, vocabs):
    """Constructs and saves time-split graphs for train and test sets."""
    _construct_temporal_graph_split("Train", dataset, w, ids, vocabs)
    _construct_temporal_graph_split("Test", dataset, w, ids, vocabs)


def _construct_window_graph_split(split_name: str, dataset: str, w: int, vocabs):
    """Helper function to build a single split (train or test) of the window graph."""
    is_train = split_name == "Train"
    output_path = f"./window{w}_{dataset}_{split_name}_Window_Graph.bin"
    
    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return

    print(f"Building {split_name} Window Graph: {output_path}")

    v_split = vocabs['train'] if is_train else vocabs['total']
    num_entities = len(v_split['E_vocab'])
    num_times = len(vocabs['total']['T_vocab'])
    
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(num_entities * num_times)
    
    # For each node, connect it to other nodes of the same entity within the window
    for i in tqdm(range(num_entities * num_times), desc=f"Building {split_name} Window Graph"):
        entity_id = i // num_times
        time_id = i % num_times

        start_ts = max(0, time_id - w)
        end_ts = min(num_times, time_id + w + 1)
        
        src_nodes = [entity_id * num_times + ts for ts in range(start_ts, end_ts) if ts != time_id]
        
        if src_nodes:
            g.add_edges(src_nodes, [i] * len(src_nodes))
            
    save_graphs(output_path, [g])


def construct_window_graphs(dataset: str, w: int, vocabs):
    """Constructs graphs connecting same-entity nodes across time."""
    _construct_window_graph_split("Train", dataset, w, vocabs)
    _construct_window_graph_split("Test", dataset, w, vocabs)


def main():
    """Main function to orchestrate the data building process."""
    args = parse_args()
    
    # --- 1. Load Data and Build Vocabularies ---
    t0 = time.perf_counter()
    train_q, valid_q, test_q = load_raw_data(args.dataset)
    vocabs, ids = build_vocabs_and_ids(train_q, valid_q, test_q)
    print(f"[+] Data loaded and processed in {time.perf_counter() - t0:.2f}s.")
    print(f"    Entities (Train/Total): {len(vocabs['train']['E_vocab'])} / {len(vocabs['total']['E_vocab'])}")
    print(f"    Relations (Total): {len(vocabs['total']['R_vocab'])}")
    print(f"    Timestamps (Total): {len(vocabs['total']['T_vocab'])}")

    # --- 2. Build Graphs ---
    t1 = time.perf_counter()
    
    if args.parallel:
        print("\n[+] Starting parallel graph construction...")
        max_workers = args.num_workers or (os.cpu_count() or 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(construct_global_graphs, args.dataset, ids, vocabs),
                executor.submit(_construct_temporal_graph_split, "Train", args.dataset, args.window_size, ids, vocabs),
                executor.submit(_construct_temporal_graph_split, "Test", args.dataset, args.window_size, ids, vocabs),
                executor.submit(_construct_window_graph_split, "Train", args.dataset, args.window_size, vocabs),
                executor.submit(_construct_window_graph_split, "Test", args.dataset, args.window_size, vocabs),
            ]
            wait(futures)
        print("[+] Parallel graph construction finished.")
    else:
        print("\n[+] Starting serial graph construction...")
        construct_global_graphs(args.dataset, ids, vocabs)
        construct_temporal_graphs(args.dataset, args.window_size, ids, vocabs)
        construct_window_graphs(args.dataset, args.window_size, vocabs)
        print("[+] Serial graph construction finished.")
        
    print(f"\n[+] All graphs built in {time.perf_counter() - t1:.2f}s.")


if __name__ == "__main__":
    main()
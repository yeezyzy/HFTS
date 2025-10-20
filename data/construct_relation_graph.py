import dgl
import torch
import argparse
from build_data_unified import load_raw_data, build_vocabs_and_ids

from tqdm import tqdm
import os
import numpy as np
import copy
import pickle

import numpy as np
from dgl.data.utils import save_graphs, load_graphs

arser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Constructing Temporal Knowledge Graph")
parser.add_argument("--dataset", default="YAGO11k", choices=["ICEWS14", "ICEWS0515", "YAGO11k"], help="dataset folder name, which has train.txt, test.txt, valid.txt in it")
parser.add_argument("--window_size", default=10, type=int, help="window size to read proper graph")
args = parser.parse_args()

data_path = './'
time_split_window_size=args.window_size
dataset=args.dataset

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
    
    return g

def load_graph():
    path = data_path+"window_dynamic_"+str(time_split_window_size)+'_'+dataset+"_"+"Train_time_split_absolute_Graph"+".bin"
    if os.path.exists(path):
        print("graph exist")
        return load_graphs(path)[0][0]
    
    print("graph not exist, constructing...")
    train_q, valid_q, test_q = load_raw_data(args.dataset)
    vocabs, ids = build_vocabs_and_ids(train_q, valid_q, test_q)
    return _construct_temporal_graph_split("Train", dataset, time_split_window_size, ids, vocabs)

G = load_graph()

edge_dict = {}
node_set = set()

cnt = 0

for node_id in tqdm(G.nodes().tolist()):
    in_nodes = set(G.in_edges(node_id)[0].tolist())
    in_timestamp = []
    in_relations = []
    in_rel_time = []
    for in_node in in_nodes:
        in_edge_id = G.edge_ids(in_node, node_id)[2]
        times = G.edata['absolute_time'][in_edge_id].tolist()
        relations = G.edata['relation_idx'][in_edge_id].tolist()
        for r, t in zip(relations, times):
            in_rel_time.append([r, t])

    if len(in_rel_time) == 0:
        continue
    out_nodes = set(G.out_edges(node_id)[1].tolist())
    out_timestamp = []
    out_relation = []
    out_rel_time = []
    for out_node in out_nodes:
        out_edge_id = G.edge_ids(node_id, out_node)[2]
        times = G.edata['absolute_time'][out_edge_id].tolist()
        relations = G.edata['relation_idx'][out_edge_id].tolist()
        for time, relation in zip(times, relations):
            out_rel_time.append([relation, time])
            for r, t in [(r, t) for r, t in in_rel_time if t <= time]:
                span = time - t
                if (r, relation) not in edge_dict.keys():
                    edge_dict[(r, relation)] = [span, 1]
                    node_set.add(r)
                    node_set.add(relation)
                # elif(span>0):
                else:
                    edge_dict[(r, relation)][0] += span
                    edge_dict[(r, relation)][1] += 1



Train_Relation_Graph = dgl.DGLGraph(multigraph=True)
Train_Relation_Graph.add_nodes(len(node_set))

time_span = []
for edge in edge_dict.keys():
    num, freq = edge_dict[edge]
    time_span.append(round(num/freq, 2))
    Train_Relation_Graph.add_edges(edge[0], edge[1])
Train_Relation_Graph.ndata['relation_idx'] = torch.tensor(range(460))
Train_Relation_Graph.edata['time_span'] = torch.tensor(time_span)

from dgl.data.utils import save_graphs
save_graphs("./window"+str(time_split_window_size)+'_'+dataset+"_Train_Relation_Graph_Timespan.bin", [Train_Relation_Graph])


import dgl
import torch

from dgl.data.utils import load_graphs
import pickle

import random
from tqdm import tqdm
import argparse

from utils import complex
from utils import distmult
from utils import transE
from utils import print_metrics_single
from utils import rank
from utils import get_historical_subgraph
from utils import get_future_subgraph
import model

torch.set_num_threads(3)

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Constructing Temporal Knowledge Graph")
parser.add_argument("--dataset", default="ICEWS14", choices=["ICEWS14", "ICEWS0515", "YAGO11k", "wikidata"],
                    help="dataset folder name, which has train.txt, test.txt, valid.txt in it")
parser.add_argument("--window_size", default="12", type=str, help="window size to read proper graph")
parser.add_argument("--rel_window_size", default="12", type=str, help="window size to read proper graph")
parser.add_argument("--time_split_window_size", default="12", type=str, help="window size to read proper graph")
parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
                    help="which gpu/cpu do you wanna use")
parser.add_argument("--aT_ratio", default=0.85, type=float, help="weighted sum ratio between TempGCN and aTempGCN")
parser.add_argument("--rel_ratio", default=0.2, type=float, help="ratio of RelGCN")
parser.add_argument("--SSL_ratio", default=1., type=float, help="ratio of Self Supervised Loss")
parser.add_argument("--score_function", default="distmult", choices=["complex", "distmult", "transE"],
                    help="choose score function")
parser.add_argument("--random_seed", default=1024, type=int, help="random_seed for random.random and torch")
parser.add_argument("--model_path", default="./data/model/best_ICEWS14.pth", type=str, help="load model path")
parser.add_argument("--T", default="O", choices=["O", "X"], help="relation_graph construct using T or not")


args = parser.parse_args()
random_seed = args.random_seed
random.seed(random_seed)
torch.manual_seed(random_seed)

data_name = args.dataset
window_size = args.window_size
rel_window_size = args.rel_window_size
time_split_window_size = args.time_split_window_size
device_0 = args.device
aT_ratio = args.aT_ratio
rel_ratio = args.rel_ratio
SSL_ratio = args.SSL_ratio
model_path = args.model_path
score_function = args.score_function
rel_T = args.T
likelihood = distmult

last_model_filename = None

if score_function == "complex":
    likelihood = complex
elif score_function == "distmult":
    likelihood = distmult
elif score_function == "transE":
    likelihood = transE

print('Loading datas')
with open('./data/data_absolution_' + data_name + '.pickle', 'rb') as f:
    data = pickle.load(f)
num_of_time, num_of_rel, num_of_ent, num_of_train_ent = data['nums']
train_dataloader_list = data['train_data']
valid_data = data['valid_data']
test_data = data['test_data']
G_train_facts = data['train_facts']

print("Loading graphs")
data_path = "./data/"
Test_Global_Graph = load_graphs(data_path + data_name + "_" + "Test_Global_Graph" + ".bin")[0][0]
Test_time_split_Graph = load_graphs(data_path + "window_dynamic_" + time_split_window_size + '_' + data_name + "_" + "Test_time_split_absolute_Graph" + ".bin")[0][0]
Test_Window_Graph = load_graphs(data_path + "window" + "1" + '_' + data_name + "_" + "Test_Window_Graph" + ".bin")[0][0]
Test_Relation_Graph = load_graphs(data_path + "window" + rel_window_size + '_' + data_name + "_" + "Train_Relation_Graph_Timespan" + ".bin")[0][0]

relation_time_span_weight = [[int(time_split_window_size)] * (num_of_rel * 2)] * (num_of_rel * 2)
for i in range(num_of_rel * 2):
    for j in range(num_of_rel * 2):
        if Test_Relation_Graph.has_edges_between(i, j).item() == 1:
            relation_time_span_weight[i][j] = Test_Relation_Graph.edata['time_span'][
                Test_Relation_Graph.edge_ids(i, j)]

emb_dim = 100
temperature = 0.1

model = model.T_aT_R1_GCN_SSL(num_of_ent, num_of_time, num_of_rel * 2, emb_dim, temperature, device_0, aT_ratio,
                                     rel_ratio, random_seed, time_split_window_size, relation_time_span_weight)

checkpoint = torch.load(model_path, map_location=device_0)
model.load_state_dict(checkpoint['state_dict'])

print("start testing")
model.eval()
with torch.no_grad():
    object_filtered_data_ranks = []
    subject_filtered_data_ranks = []
    r_object_filtered_data_ranks = []
    r_subject_filtered_data_ranks = []
    entity_index = list(range(num_of_ent))

    for s, r, o, t, o_filter_mask, s_filter_mask, _, __ in tqdm(test_data):
        historical_window_size = min(int(time_split_window_size), t) + 1
        future_window_size = min(int(time_split_window_size), num_of_time - t)
        historical_ratio = historical_window_size / (historical_window_size + future_window_size)
        future_ratio = future_window_size / (historical_window_size + future_window_size)

        entity_set = torch.tensor(entity_index) * num_of_time + torch.tensor([t])
        with Test_time_split_Graph.local_scope():
            with Test_Window_Graph.local_scope():
                with Test_Global_Graph.local_scope():
                    test_historical_graph = get_historical_subgraph(Test_time_split_Graph, t)
                    test_future_graph = get_future_subgraph(Test_time_split_Graph, t)
                    entity_embs, relation_emb = model(test_historical_graph,
                                                      test_future_graph,
                                                      Test_Window_Graph,
                                                      Test_Global_Graph,
                                                      Test_Relation_Graph, entity_set,
                                                      torch.tensor([r, r + num_of_rel]),
                                                      num_of_ent, historical_ratio, future_ratio)
        score = likelihood(entity_embs[s], relation_emb[0], entity_embs[o]).item()
        reciprocal_score = likelihood(entity_embs[o], relation_emb[1],
                                      entity_embs[s]).item()
        objects_score = likelihood(entity_embs[s].repeat(num_of_ent, 1),
                                   relation_emb[0].repeat(num_of_ent, 1),
                                   entity_embs)
        subjects_score = likelihood(entity_embs,
                                    relation_emb[0].repeat(num_of_ent, 1),
                                    entity_embs[o].repeat(num_of_ent, 1))

        filtered_objects_scores = objects_score[o_filter_mask].tolist()
        filtered_subjects_scores = subjects_score[s_filter_mask].tolist()
        object_filtered_rank = rank(sorted(filtered_objects_scores), score)
        subject_filtered_rank = rank(sorted(filtered_subjects_scores), score)
        object_filtered_data_ranks.append(object_filtered_rank)
        subject_filtered_data_ranks.append(subject_filtered_rank)
        r_object_filtered_rank = rank(sorted(filtered_objects_scores), reciprocal_score)
        r_subject_filtered_rank = rank(sorted(filtered_subjects_scores), reciprocal_score)
        r_object_filtered_data_ranks.append(r_object_filtered_rank)
        r_subject_filtered_data_ranks.append(r_subject_filtered_rank)
    MRR, h1, h3, h10, result = print_metrics_single(r_object_filtered_data_ranks, r_subject_filtered_data_ranks)
    print("test result\nMRR:", MRR, "\nHits@1:", h1, "\nHits@3:", h3, "\nHits@10:", h10, "\n")
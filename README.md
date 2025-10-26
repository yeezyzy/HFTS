# Enhancing Temporal Knowledge Graph Completion via Historical-Future Subgraphs and Time-Span Sensitivity

**requirements**

```bash
conda create -n THOR python=3.8
conda install pytorch=1.7.0 cudatoolkit=11.0 -c pytorch
pip install dgl==0.4.1
pip install tqdm
pip install seaborn
```

**data_process**

```bash
cd data
python data_processor.py --dataset ICEWS14
python construct_graph.py --dataset ICEWS14 --window_size 12
python construct_relation_graph.py --dataset ICEWS14 --window_size 12
```

**train**

ICEWS14

```bash
python ./train.py --dataset ICEWS14 --window_size 12 --device cuda:0 --aT_ratio 0.85 --rel_ratio 0.2
```

YAGP11k

```bash
python ./train.py --dataset YAGO11k --window_size 11 --device cuda:0 --aT_ratio 0.8 --rel_ratio 0.05
```

*In order to increase readability, the project code has been refactored. If the program cannot run, please try using the original version in the backup directory.*
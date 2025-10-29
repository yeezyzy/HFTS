# HFTS: Time-Span-Aware Historical–Future Modeling for Temporal Knowledge Graph Completion

## Prerequisites

Before cloning the repository, ensure you have [Git LFS](https://git-lfs.com/) installed. Use the following command to clone the repository with all its contents:

```bash
git lfs clone <repository-url>
```

## Requirements

To set up the environment, run the following commands:

```bash
conda create -n THOR python=3.8
conda install pytorch=1.7.0 cudatoolkit=11.0 -c pytorch
pip install dgl==0.4.1
pip install tqdm
pip install seaborn
```

## Data Processing

Preprocess the dataset (e.g., ICEWS14) with the following commands:

```bash
cd data
python data_processor.py --dataset ICEWS14
python construct_graph.py --dataset ICEWS14 --window_size 12
python construct_relation_graph.py --dataset ICEWS14 --window_size 12
```

## Training

### ICEWS14

Train the model on the ICEWS14 dataset:

```bash
python ./train.py --dataset ICEWS14 --window_size 12 --device cuda:0 --aT_ratio 0.85 --rel_ratio 0.2
```

### YAGO11k

Train the model on the YAGO11k dataset:

```bash
python ./train.py --dataset YAGO11k --window_size 11 --device cuda:0 --aT_ratio 0.8 --rel_ratio 0.05
```

## pre-trained
> You can download the pre-trained models from [Hugging Face](https://huggingface.co/moyeezy/HFTS/tree/main).

We provide a pre-trained model for the ICEWS14 dataset. To evaluate it, run:

```bash
python ./test.py --model_path ./data/model/saved_ICEWS14.pth
```
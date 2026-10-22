# EGMF-DTI

## Introduction

The official code implementation for EGMF-DTI from our paper, "Grouped Fusion of Multiscale Protein Structure
and Drug Topology and Geometry for
Drug–Target Interaction Prediction".

## Installation

We recommend using a conda environment with Python 3.8.

```bash
conda create -n egmfdti python=3.8
conda activate egmfdti
```

Install PyTorch (adjust the CUDA version according to your system):

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```text
descriptastorus==2.8.0
horovod==0.28.1
matplotlib==3.7.5
numpy==1.24.4
paddle==1.4.1
pandas==1.4.1
PyYAML==6.0.2
PyYAML==6.0.3
rdkit==2024.3.5
rdkit_pypi==2022.9.5
Requests==2.34.2
scikit_learn==1.3.2
scipy==1.18.1
seaborn==0.13.2
tensorflow==2.21.0
torch==2.1.0+cu121
torch_geometric==2.8.0.post1
torchdrug==0.2.1
tqdm==4.66.5
transformers==4.46.3
yacs==0.1.8
```

## Data Preparation

### Datasets

| Dataset  | Drugs | Protein | Interaction |
| :------- | :---: | :-----: | :---------: |
| DrugBank | 6,148 |  4,085  |   33,062    |
| Davis    |  68   |   379   |   25,772    |
| KIBA     | 2,068 |   225   |   116,350   |

### Directory Layout

Place the raw data under the `dataset/` directory. After running the feature extraction scripts, the expected structure will be:

```text
dataset/
├── drugbank/          # DrugBank features
│   ├── raw
│   ├── processed
│   ├── pocket_pdb
│   ├── pdb
│   ├── t_feature.npy
│   ├── d_2D_feature.npy
│   ├── d_3D_feature.npy
│   ├── t_3D_feature.npy
│   ├── pocket_feature.npy
│   ├── total_cid_unid_csv.csv
│   └── metadata.csv
├── davis/               # Davis features
│   ├── raw
│   ├── processed
│   ├── pocket_pdb
│   ├── pdb
│   ├── davis_t_feature.npy
│   ├── davis_d_2d_feature.npy
│   ├── davis_d_3d_feature.npy
│   ├── t_3d_feature.npy
│   ├── pocket_feature.npy
│   └── davis_total_cid_unid_csv.csv
└── KIBA/                # KIBA features
    └── ...
```

### Feature Extraction

Run the following scripts to extract features for each modality. Make sure the raw data files (CSV, PDB, etc.) are available and paths are correctly set.

**Protein sequence features** (using ProtT5):

```bash
python extract_p_emb.py
```

**Protein 3D structure and pocket features** (using GearNet and P2Rank):

```bash
python extract_t_structure_emd.py
```

**Drug 2D features** (using GROVER):

First, download the pretrained GROVER model (`grover_large.pt`) from the [official GROVER repository](https://github.com/tencent-ailab/grover) and place it under the `GROVER/` directory.

Then run the two commands below to generate the 2D molecular features (using the Davis dataset as an example):

```bash
cd GROVER

python scripts/save_features.py \
    --data_path davis.csv \
    --save_path rdkit_200.npz \
    --features_generator rdkit_2d_normalized \
    --restart

python main.py fingerprint \
    --data_path davis.csv \
    --features_path rdkit_200.npz \
    --checkpoint_path ./grover_large.pt \
    --fingerprint_source both \
    --batch_size 64 \
    --output davis_d_2d_feature.npy
```

**Drug 3D features** (using RDKit-generated conformers and geometric graphs):

```bash
python extract_drug_3d_emb.py
```

## Training

After feature extraction, train GroupDTI on each benchmark dataset. The trained weights and parameters will be saved under the `runs/` directory.

**DrugBank**:

```bash
python main_drugbank.py --cfg configs/drugbank.yaml --data drugbank --split random
```

**Davis**:

```bash
python main_davis.py --cfg configs/davis.yaml --data davis --split random
```

**KIBA**:

```bash
python main_KIBA.py --cfg configs/kiba.yaml --data kiba --split random
```

## Prediction

Once the model is trained, run inference using the saved checkpoint:

```bash
python inference.py
```

Make sure the checkpoint path in `inference.py` points to the desired model under `runs/`.

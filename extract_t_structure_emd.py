import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchdrug import data, models, layers
from torchdrug.layers import geometry

CSV_PATH = "./dataset/KIBA/KIBA_total_cid_unid.csv"
#'./dataset/davis/davis_total_cid_unid.csv' 'dataset/drugbank/metadata.csv'
CSV_UID_COL = "uid"
PDB_DIR = "./dataset/KIBA/pdb"
#"./dataset/KIBA/pocket_pdb"
OUTPUT_NPY = "./dataset/KIBA/KIBA_t_3D_feature.npy"
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

graph_construction_model = layers.GraphConstruction(
    node_layers=[geometry.AlphaCarbonNode()],
    edge_layers=[
        geometry.SequentialEdge(max_distance=2),
        geometry.SpatialEdge(radius=10.0, min_distance=5),
        geometry.KNNEdge(k=10, min_distance=5)
    ],
    edge_feature="gearnet"         
)

model = models.GearNet(
    input_dim=21,
    hidden_dims=[512, 512, 512, 512, 512, 512], 
    num_relation=7,
    edge_input_dim=59,
    num_angle_bin=8,
    short_cut=True,
    batch_norm=True,
    concat_hidden=True,             
    readout="sum"
)

print("Loading model weights mc_gearnet_edge.pth...")
checkpoint = torch.load("./models/mc_gearnet_edge.pth", map_location="cpu")
state_dict = checkpoint.get('model', checkpoint) 

model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

output_dim = 512 * 6
print(f"Model output feature dimension: {output_dim}")

def extract_features_from_pdb(pdb_path):
    try:
        protein = data.Protein.from_pdb(
            pdb_path,
            atom_feature='position',
            bond_feature='length',
            residue_feature='symbol'
        )
        packed = data.Protein.pack([protein])
        graph = graph_construction_model(packed).to(DEVICE)
        node_feature = graph.residue_feature.float()   # shape [N, 21]
        
        with torch.no_grad():
            output = model(graph, node_feature)
            graph_feature = output['graph_feature'].cpu().numpy().flatten()
        return graph_feature
    except Exception as e:
        print(f"  Error processing {os.path.basename(pdb_path)}: {e}")
        return None

df = pd.read_csv(CSV_PATH)
uids = df[CSV_UID_COL].astype(str).tolist()
print(f"Total proteins to process: {len(uids)}")

features_list = []   
missing_indices = [] 
valid_uids = []

for i, uid in enumerate(tqdm(uids, desc="Extracting valid features")):
    pdb_file = os.path.join(PDB_DIR, f"{uid}.pdb")
    if not os.path.exists(pdb_file):
        missing_indices.append(i)
        continue
    
    feat = extract_features_from_pdb(pdb_file)
    if feat is not None:
        features_list.append(feat)
        valid_uids.append(uid)
    else:
        missing_indices.append(i)

features_array_valid = np.stack(features_list) 
print(f"Valid features: {len(features_list)}, Missing: {len(missing_indices)}")

mean_feature = features_array_valid.mean(axis=0)  
print(f"Mean feature dimension: {mean_feature.shape}")

all_features = []
valid_idx = 0
for i in range(len(uids)):
    if i in missing_indices:
        all_features.append(mean_feature.copy()) 
    else:
        all_features.append(features_array_valid[valid_idx])
        valid_idx += 1

features_array = np.stack(all_features)
print(f"Feature matrix shape: {features_array.shape}")
np.save(OUTPUT_NPY, features_array)
print(f"Features saved to: {OUTPUT_NPY}")
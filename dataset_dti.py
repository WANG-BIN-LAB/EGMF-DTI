import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdmolops

class DatasetForExtraction:
    def __init__(self, csv_path, smiles_field='SMILES', id_field='ID', addH=True, batch_size=32):
        """
        用于特征提取的数据集类，按原始顺序读取所有分子，不划分训练/验证/测试。
        Args:
            csv_path: CSV文件路径
            smiles_field: SMILES列名
            id_field: ID列名（可选）
            addH: 是否添加氢原子
            batch_size: 批次大小（用于模型推理，不会打乱顺序）
        """
        self.csv_path = csv_path
        self.smiles_field = smiles_field
        self.id_field = id_field
        self.addH = addH
        self.batch_size = batch_size

        # 读取CSV
        self.data = pd.read_csv(csv_path)
        self.smiles_list = self.data[self.smiles_field].values
        if self.id_field in self.data.columns:
            self.ids = self.data[self.id_field].values
        else:
            self.ids = np.arange(len(self.smiles_list))

        # 预处理所有分子，得到原子序列和邻接矩阵
        self.atom_seqs = []
        self.adj_matrices = []
        self.valid_indices = []
        for idx, smi in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"警告: 无效 SMILES '{smi}'，跳过该样本")
                continue
            if self.addH:
                mol = Chem.AddHs(mol)
            atom_seq = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            adj = rdmolops.GetAdjacencyMatrix(mol).astype(np.int32)
            self.atom_seqs.append(atom_seq)
            self.adj_matrices.append(adj)
            self.valid_indices.append(idx)

        # 更新有效样本的ID和标签（如果需要）
        self.ids = self.ids[self.valid_indices]
        self.num_samples = len(self.atom_seqs)
        print(f"有效分子数: {self.num_samples}")

    def _generator(self):
        """生成器，逐个返回 (原子序列, 邻接矩阵, 原始索引)"""
        for idx, (seq, adj) in enumerate(zip(self.atom_seqs, self.adj_matrices)):
            yield seq, adj, idx  # idx 是有效样本的连续索引，用于保持顺序

    def _pad_batch(self, seqs, adjs):
        """填充一个批次到相同长度"""
        max_len = max(len(s) for s in seqs)
        padded_seqs = []
        padded_adjs = []
        for seq, adj in zip(seqs, adjs):
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_adj = np.zeros((max_len, max_len), dtype=np.int32)
            padded_adj[:len(seq), :len(seq)] = adj
            padded_seqs.append(padded_seq)
            padded_adjs.append(padded_adj)
        return np.array(padded_seqs, dtype=np.int32), np.array(padded_adjs, dtype=np.int32)

    def get_dataset(self):
        """返回 tf.data.Dataset，按原始顺序，批次内顺序不变"""
        # 构建数据集，每个元素为 (seq, adj, idx)
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([None,], [None, None], [])
        )
        # 按批次聚合，填充到批次内最大长度
        def batch_pad(seq, adj, idx):
            seqs, adjs, idxs = list(seq), list(adj), list(idx)
            padded_seqs, padded_adjs = self._pad_batch(seqs, adjs)
            return tf.constant(padded_seqs, dtype=tf.int32), tf.constant(padded_adjs, dtype=tf.int32), tf.constant(idxs, dtype=tf.int32)

        dataset = dataset.batch(self.batch_size).map(batch_pad)
        return dataset, self.ids
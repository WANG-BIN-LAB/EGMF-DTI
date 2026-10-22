import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem


def rdchem_enum_to_list(values):
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


class CompoundKit(object):
    atom_vocab_dict = {
        "atomic_num": list(range(1, 200)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167
    period_table = Chem.GetPeriodicTable()

    @staticmethod
    def get_atom_value(atom, name):
        if name == 'atomic_num': return atom.GetAtomicNum()
        elif name == 'chiral_tag': return atom.GetChiralTag()
        elif name == 'degree': return atom.GetDegree()
        elif name == 'explicit_valence': return atom.GetExplicitValence()
        elif name == 'formal_charge': return atom.GetFormalCharge()
        elif name == 'hybridization': return atom.GetHybridization()
        elif name == 'implicit_valence': return atom.GetImplicitValence()
        elif name == 'is_aromatic': return int(atom.GetIsAromatic())
        elif name == 'mass': return int(atom.GetMass())
        elif name == 'total_numHs': return atom.GetTotalNumHs()
        elif name == 'num_radical_e': return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring': return int(atom.IsInRing())
        elif name == 'valence_out_shell': return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else: raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        assert name in CompoundKit.atom_vocab_dict
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        assert name in CompoundKit.atom_vocab_dict
        return len(CompoundKit.atom_vocab_dict[name])

    @staticmethod
    def get_bond_value(bond, name):
        if name == 'bond_dir': return bond.GetBondDir()
        elif name == 'bond_type': return bond.GetBondType()
        elif name == 'is_in_ring': return int(bond.IsInRing())
        elif name == 'is_conjugated': return int(bond.GetIsConjugated())
        elif name == 'bond_stereo': return bond.GetStereo()
        else: raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        assert name in CompoundKit.bond_vocab_dict
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        assert name in CompoundKit.bond_vocab_dict
        return len(CompoundKit.bond_vocab_dict[name])

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=CompoundKit.morgan_fp_N)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=CompoundKit.morgan2048_fp_N)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        fg_counts = []                    # ← 改回普通 list
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        rings = mol.GetRingInfo()
        rings_info = [r for r in rings.AtomRings()]
        ring_list = []                    # ← 改回普通 list
        for atom in mol.GetAtoms():
            atom_result = []              # ← 改回普通 list
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = sum(1 for r in rings_info if len(r) == ringsize and atom.GetIdx() in r)
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        atom_features_dicts = []          # ← 改回普通 list
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))
        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])
        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc: pc = 0
        if pc == float('inf'): pc = 10
        return pc


class AtomEmbedding(nn.Module):
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim)
            self.embed_list.append(embed)

    def forward(self, node_features):
        # ★ 强制输入跟权重同设备
        target_device = self.embed_list[0].weight.device
        node_features = node_features.to(target_device)
    
        out_embed = self.embed_list[0](node_features.T[0])
        for i in range(1, len(self.embed_list)):
            out_embed = torch.cat((out_embed, self.embed_list[i](node_features.T[i])), dim=1)
        return out_embed


class BondEmbedding(nn.Module):
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        # ★ 强制输入跟权重同设备
        target_device = self.embed_list[0].weight.device
        edge_features = edge_features.to(target_device)
    
        out_embed = self.embed_list[0](edge_features.T[0])
        for i in range(1, len(self.embed_list)):
            out_embed = torch.cat((out_embed, self.embed_list[i](edge_features.T[i])), dim=1)
        return out_embed


class RBF(nn.Module):
    """Radial Basis Function"""

    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        self.register_buffer(
            'centers',
            torch.reshape(torch.tensor(centers, dtype=torch.float32), [1, -1])
        )
        self.gamma = gamma

    def forward(self, x):
        # ★ 加 dtype 和 device 双保险
        x = torch.reshape(x, [-1, 1]).to(dtype=torch.float32, device=self.centers.device)
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(nn.Module):
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names
        if rbf_params is None:
            self.rbf_params = {'bond_length': (np.arange(0, 2, 0.1), 10.0)}
        else:
            self.rbf_params = rbf_params
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            self.rbf_list.append(RBF(centers, gamma))
            self.linear_list.append(nn.Linear(len(centers), embed_dim))

    def forward(self, bond_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            rbf_x = self.rbf_list[i](bond_float_features).float()
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Module):
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names
        if rbf_params is None:
            self.rbf_params = {'bond_angle': (np.arange(0, np.pi, 0.1), 10.0)}
        else:
            self.rbf_params = rbf_params
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            self.rbf_list.append(RBF(centers, gamma))
            self.linear_list.append(nn.Linear(len(centers), embed_dim))

    def forward(self, bond_angle_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            rbf_x = self.rbf_list[i](bond_angle_float_features).float() 
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class DistanceFloatRBF(nn.Module):
    def __init__(self, distance_float_names, embed_dim, rbf_params=None):
        super(DistanceFloatRBF, self).__init__()
        self.distance_float_names = distance_float_names
        if rbf_params is None:
            self.rbf_params = {'distance': (np.arange(0, 9.5, 0.1), 10.0)}
        else:
            self.rbf_params = rbf_params
        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.distance_float_names:
            centers, gamma = self.rbf_params[name]
            self.rbf_list.append(RBF(centers, gamma))
            self.linear_list.append(nn.Linear(len(centers), embed_dim))

    def forward(self, distance_float_features):
        out_embed = 0
        for i, name in enumerate(self.distance_float_names):
            rbf_x = self.rbf_list[i](distance_float_features).float() 
            out_embed += self.linear_list[i](rbf_x)
        return out_embed
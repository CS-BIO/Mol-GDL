# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:41:46 2021

@author: 申聪
"""
import numpy as np
from rdkit import Chem
from numpy import random
from typing import List, Tuple, Union

# DeepDDS deep graph neural network with attention mechanism to predict synergistic drug combinations
def atom_features3(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.ADDING_H = False

PARAMS = Featurization_parameters()

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


#A Deep Learning Approach to Antibiotic Discovery
def atom_features2(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


def one_of_k_encoding1(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def atom_features1(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding1(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)

class mol_feature(object):
        
    def Calculate_feature1(smile): #133维
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
#        print("分子中原子的数量：",mol.GetNumAtoms())
        features = []
        for atom in mol.GetAtoms():
#            print(atom)
            feature = atom_features1(atom)
#            print(len(feature))
#            print(feature)
            features.append(list(np.array(feature,dtype=float) / sum(feature)))
#        print(features)
        return features
    
    
    def Calculate_feature2(smile): #133维
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
#        print("分子中原子的数量：",mol.GetNumAtoms())
        features = []
        for atom in mol.GetAtoms():
#            print(atom)
            feature = atom_features2(atom)
#            print(len(feature))
#            print(feature)
            features.append(list(np.array(feature,dtype=float) / sum(feature)))
#        print(features)
        return features
    
    
    
    def Calculate_feature3(smile):
        mol = Chem.MolFromSmiles(smile)   
        mol = Chem.AddHs(mol)
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features3(atom)
            features.append(feature / sum(feature))
        return features
    
    
    def Calculate_feature(dis_mat,all_atoms):
#        print(dis_mat)
#        print(all_atoms)
        Feat_mat = np.zeros((len(all_atoms),62),dtype=float)
#        Feat_mat = random.random(size = (len(all_atoms),args.dim_init))
#        print(dis_mat)
        for a in range(len(dis_mat[0,:])):
            for b in range(len(dis_mat[:,0])):
                if dis_mat[a][b] == 0:
                    continue
                if all_atoms[b] == 'C':
                    if dis_mat[a][b] >= 10:
                        Feat_mat[a,18] += 1
                    else:
                        Feat_mat[a,int((dis_mat[a,b]-1)*2)] += 1
                elif all_atoms[b] == 'H':
                    if dis_mat[a][b] >= 10:
                        Feat_mat[a,37] += 1
                    else:
                        Feat_mat[a,19+int((dis_mat[a,b]-1)*2)] += 1
                elif all_atoms[b] == 'O':
                    if dis_mat[a][b] <= 2.5:
                        Feat_mat[a,38] += 1
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,39] += 1
                    if dis_mat[a][b] <= 7.5:
                        Feat_mat[a,40] += 1
                    else:
                        Feat_mat[a,41] += 1
                elif all_atoms[b] == 'N':
                    if dis_mat[a][b] <= 2.5:
                        Feat_mat[a,42] += 1
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,43] += 1
                    if dis_mat[a][b] <= 7.5:
                        Feat_mat[a,44] += 1
                    else:
                        Feat_mat[a,45] += 1    
                elif all_atoms[b] == 'P':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,46] += 1
                    else:
                        Feat_mat[a,47] += 1
                elif all_atoms[b] == 'Cl'or all_atoms[a] == 'CL':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,48] += 1
                    else:
                        Feat_mat[a,49] += 1
                elif all_atoms[b] == 'F':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,50] += 1
                    else:
                        Feat_mat[a,51] += 1
                elif all_atoms[b] == 'Br':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,52] += 1
                    else:
                        Feat_mat[a,53] += 1
                elif all_atoms[b] == 'S':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,54] += 1
                    else:
                        Feat_mat[a,55] += 1
                elif all_atoms[b] == 'Si':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,56] += 1
                    else:
                        Feat_mat[a,57] += 1
                elif all_atoms[b] == 'I':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,58] += 1
                    else:
                        Feat_mat[a,59] += 1
                elif all_atoms[b] == 'As' or all_atoms[a] == 'AS':
                    if dis_mat[a][b] <= 5:
                        Feat_mat[a,60] += 1
                    else:
                        Feat_mat[a,61] += 1
#            temp_sum = np.sum(Feat_mat[a,:])
#            Feat_mat[a,:] = Feat_mat[a,:]/temp_sum
        return Feat_mat

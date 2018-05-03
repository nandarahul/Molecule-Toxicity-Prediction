import os
from rdkit import Chem
import glob
import json
import numpy as np
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem import Fragments, Lipinski
from rdkit.Chem import Crippen, MolSurf


bond_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, "AROMATIC": 4}
el = ['Ni', 'Pt', 'Na', 'Nd', 'Mg', 'Al', 'Pb', 'Pd', 'Dy', 'Be', 'Ba', 'Bi', 'Fe', 'Br', 'Sr', 'Hg', 'C', 'B', 'F', 'I', 'H', 'K', 'Mn', 'O', 'N', 'P', 'Si', 'Sn', 'V', 'Sb', 'Li', 'Se', 'Zn', 'Co', 'Ge', 'Ag', 'Cl', 'Ca', 'S', 'Cd', 'As', 'Au', 'Zr', 'In', 'Cr', 'Cu']
es = set([])
def to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
    return edges


def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))


import csv
def main():
    infile = open("molecule_training.csv", 'r')
    infile.readline()

    with open('train_molecule_new_features.csv', 'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['index', 'Maximum Degree', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
        #                  'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area', 'Graph', 'smiles',
        #                  'target'])
        writer.writerow(['index', 'Maximum Degree', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
                         'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area', 'fr_phos',
                         'aromatic_carbocycles', 'MolLogP', 'PEOE_VSA1', 'Fingerprint', 'smiles', 'target'])
        for line in infile:
            line = line.strip('\n\r ')
            line = line.split(",")
            smiles = line[10].strip()
            #edge_list = to_graph(smiles)
            mol = Chem.MolFromSmiles(smiles)
            # fingerprint_explicit_bitvector = RDKFingerprint(mol)
            # fingerprint_bit_string = fingerprint_explicit_bitvector.ToBitString()
            fingerprint_bit_string = GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            #writer.writerow(line[:8] + [fingerprint_bit_string, line[10], line[11]])
            #writer.writerow(line[:8] + [edge_list] + [line[10], line[11]])
            fr_phos = Fragments.fr_phos_acid(mol) + Fragments.fr_phos_ester(mol)
            aromatic_cc = Lipinski.NumAromaticCarbocycles(mol)
            molLogP = Crippen.MolLogP(mol)
            peoe_vsa1 = MolSurf.PEOE_VSA1(mol)
            writer.writerow(line[:8] + [fr_phos, aromatic_cc, molLogP, peoe_vsa1, fingerprint_bit_string,
                                        line[10], line[11]])

    infile.close()

    infile = open("molecule_TestFeatures.csv", 'r')
    infile.readline()

    with open('test_molecule_new_features.csv', 'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['index', 'Maximum Degree', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
        #                  'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area', 'Graph', 'smiles',
        #                  'target'])
        writer.writerow(['index', 'Maximum Degree', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
                         'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area', 'fr_phos',
                         'aromatic_carbocycles', 'MolLogP', 'PEOE_VSA1', 'Fingerprint', 'smiles'])
        for line in infile:
            line = line.strip('\n\r ')
            line = line.split(",")
            smiles = line[10].strip()
            # edge_list = to_graph(smiles)
            mol = Chem.MolFromSmiles(smiles)
            # fingerprint_explicit_bitvector = RDKFingerprint(mol)
            # fingerprint_bit_string = fingerprint_explicit_bitvector.ToBitString()
            fingerprint_bit_string = GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            fr_phos = Fragments.fr_phos_acid(mol) + Fragments.fr_phos_ester(mol)
            aromatic_cc = Lipinski.NumAromaticCarbocycles(mol)
            molLogP = Crippen.MolLogP(mol)
            peoe_vsa1 = MolSurf.PEOE_VSA1(mol)
            writer.writerow(line[:8] + [fr_phos, aromatic_cc, molLogP, peoe_vsa1, fingerprint_bit_string, line[10]])
            # writer.writerow(line[:8] + [edge_list] + [line[10], line[11]])

    infile.close()

main()

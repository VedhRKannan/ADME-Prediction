import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import matplotlib.pyplot as plt




# Load model
model = joblib.load("saved_models/SVR.pkl")

# Make predictions
def mol_to_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
predictions = model.predict([mol_to_fp("CC(=O)Nc1ccc(cc1)O")])

print(predictions)

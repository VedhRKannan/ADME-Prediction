import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import matplotlib.pyplot as plt




# Load model
# model = joblib.load("saved_models/SVR.pkl")

# Make predictions
def mol_to_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



# Load all models
names = ["lip", "sol"]
loaded_models = {name: joblib.load(f"saved_models/{name}.pkl") for name in names}

smiles = "CC(=O)Nc1ccc(cc1)O"
smiles = [mol_to_fp(smiles)]
# Make predictions with all models
for name, model in loaded_models.items():
    preds = model.predict(smiles)
    print(f"Predictions from {name}: {preds[:5]}")  # Print first 5 predictions

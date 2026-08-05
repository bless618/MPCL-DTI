# MPCL-DTI

Drug–target interaction (DTI) prediction with multi-source similarity fusion, metapath-based PU learning, GCN contrastive learning, and an MLP classifier.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 0 | `similarity_fusion.py` | `drug_fusion_similarity.txt`, `target_fusion_similarity.txt`|
| 1 | `Metapath_PU_Learning.py` | PU indices, `Y_fused_Meta.txt` |
| 2 | `xiaorongshiyangroup_divide.py` | 5-fold split files under `divide_result/` |
| 3 | `extract_smiles_protein_features.py` | `drug_smiles_feat.npy`, `protein_seq_feat.npy` |
| 4 | `model_GCN_original_structure.py` | `predict_result/embedding_cl_seq{fold}.txt` |
| 5 | `xiaorongshiyanclassifier.py` | `results_<timestamp>/fold_{f}_results.json` |
| 6 | `xiaorongshiyanaggregate.py` | `final_results.json` |

**Split policy (fixed):** 5-fold cross-validation.

## Installation

MPCL-DTI - pinned GPU environment
Target: Ubuntu 20.04 x86_64, Python 3.9.7, NVIDIA driver 535.216.03
GPU runtime: PyTorch 2.1.2 with CUDA 12.1 wheels (cu121)
#
The driver reports CUDA 12.2 and is backward-compatible with the cu121
runtime bundled in these wheels. A system-wide CUDA toolkit is not required.
#
Install in a fresh virtual environment with:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Data layout

Point the pipeline at a data directory containing:

**Required for Step 0 (similarity fusion):**
- `mat_drug_protein.txt` — DTI label matrix (`n_drugs × n_targets`)
- `mat_drug_drug.txt`, `mat_drug_disease.txt`, `mat_drug_se.txt`
- `mat_protein_protein.txt`, `mat_protein_disease.txt`

**Required for Step 3 (sequence features):**
- A drug SMILES CSV: `drug_info.csv`, `luo_drug_smiles.csv`, or `drug_smiles.csv`
- A protein sequence CSV: `protein_sequence.csv`, `protein_info.csv`, `luo_protein_sequences.csv`, or `protein_sequences.csv`

### Bundled datasets

| Directory | DTI matrix |
|-----------|------------|
| `data/` | 708 × 1512 |
| `data1/` | 2214 × 1968 |

## Quick start

Run the full pipeline on the default `data/` folder:

```bash
python run.py
```

Use `other data`:

```bash
python run.py --data-dir data_name

### Common options
```bash
# test with fewer GCN epochs
python run.py --gcn-epochs 50

## Results
Each `run.py` invocation writes a timestamp to `.current_run_timestamp.txt`. Classifier outputs go to `results_<timestamp>/`; Step 6 aggregates them into `final_results.json`.
# Interpretable Prediction of circRNA-Disease Links using Multi-Graph GNN Fusion

## 📌 Overview

Circular RNAs (circRNAs) are non-coding RNA molecules that play a significant role in gene regulation and disease mechanisms. Identifying circRNA–disease associations experimentally is expensive and time-consuming.

This project proposes a **multi-graph Graph Neural Network (GNN) framework** to predict and interpret novel circRNA–disease associations by integrating multiple biological interaction networks. The model not only predicts associations but also provides **interpretability**, making it useful for biological research and hypothesis generation.

---

## 🎯 Objectives

- Perform exploratory analysis to identify effective GNN models
- Predict circRNA–disease associations
- Integrate multiple biological networks using multi-graph fusion
- Identify key circRNAs involved in diseases
- Provide interpretable insights using explainability techniques :contentReference[oaicite:1]{index=1}

---

## 🧠 Key Idea

Instead of combining all biological relationships into a single graph, this project:

- Builds **separate graphs** for each biological interaction:
  - circRNA–miRNA
  - miRNA–disease
  - circRNA–disease
- Learns **relation-specific embeddings**
- Combines them using a **fusion mechanism**
- Predicts associations using learned representations

This preserves biological semantics and improves prediction quality.

---

## 🏗️ System Architecture

1. Data Preprocessing  
2. Multi-Graph Construction  
3. GNN Encoding (GraphSAGE / GAT)  
4. Multi-Graph Fusion  
5. Link Prediction  
6. Evaluation & Interpretability :contentReference[oaicite:2]{index=2}

---

## 📊 Datasets

The system uses multiple biological datasets:

- **HGCNCDA Dataset** (Primary)
- **HMDD** (miRNA–disease)
- **circR2Disease / Circ2Disease** (circRNA–disease)

### Dataset Statistics

| Entity Type | Count (Approx.) |
|------------|----------------|
| circRNA    | ~800           |
| miRNA      | ~800           |
| Disease    | ~150           |

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Data cleaning and normalization
- Duplicate removal
- Edge list creation
- Negative sampling
- Train/Validation/Test split (70/15/15)

### 2. Feature Engineering
- Degree & Log-Degree
- Betweenness Centrality
- Node-type One-hot Encoding
- Graph topology features

### 3. GNN Encoding
- **GraphSAGE** → neighborhood aggregation
- **GAT** → attention-based weighting

Each subgraph is encoded separately:
Ei = GNNEncoder(Gi, Xi)

### 4. Multi-Graph Fusion
Embeddings are combined using weighted fusion:
E_final(v) = Σ αi * Ei(v)

### 5. Link Prediction
Score(c, d) = σ(Ecᵀ Ed)


### 6. Interpretability
- GNNExplainer
- Feature importance analysis
- Subgraph explanations :contentReference[oaicite:3]{index=3}

---

## 📈 Results

- **Best Model:** GraphSAGE (4-layer) + Weighted Fusion
- **AUC:** ~0.814  
- **AUPR:** ~0.820  

### Key Observations
- Feature engineering significantly improves performance
- Multi-graph fusion outperforms single-graph models
- GIP similarity improves performance for sparse nodes
- Interpretability reveals biologically meaningful patterns :contentReference[oaicite:4]{index=4}

---

## 🔍 Example Predictions

| circRNA | Disease | Score |
|--------|--------|------|
| cdr1as | Nasopharyngeal carcinoma | 0.279 |
| cdr1as | Osteosarcoma | 0.279 |
| cdr1as | Stomach neoplasms | 0.279 |

These predictions highlight potential biological relevance and require further validation.

---

## 🧰 Tech Stack

- **Language:** Python  
- **Frameworks:** PyTorch, PyTorch Geometric  
- **Libraries:** NumPy, Pandas, NetworkX  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / VS Code :contentReference[oaicite:5]{index=5}

---

## 📦 Installation

```bash
git clone <your-repo-link>
cd <project-folder>

pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Run training
python train.py

# Evaluate model
python evaluate.py

# Run inference
python predict.py
```
## 🧪 Evaluation Metrics

- AUC (Area Under ROC Curve)  
- AUPR (Area Under Precision-Recall Curve)  
- 5-Fold Cross Validation  

---

## 🚀 Future Scope

- Attention-based fusion instead of fixed weights  
- Integration of more biological entities (proteins, lncRNA)  
- Deployment as a web-based tool  
- Advanced interpretability (PathExplainer)  
- Application to drug discovery and repositioning  

---

## 📚 References

1. Liu et al., High-order GCN for circRNA-disease prediction (2025)  
2. Lu & Li, GraphSAGE for metabolite-disease prediction (2024)  
3. Wang & Zhong, GAT for lncRNA-disease prediction (2022)  
4. Li et al., HGAN for drug-target prediction (2022)  
5. Ying et al., GNNExplainer (NeurIPS 2019)  

---

## 👩‍💻 Team

- Ayisha Liya 
- Sneha Sudheesh   
- Hina Parveen  
- Isha Thahaniya  
- Amna

**Guides:** Dr. Ajish Kumar K.S  and Dr. Manu Madhavan

---

## 📌 Final Note

This project bridges computational biology and graph deep learning, providing an interpretable system for discovering novel biological relationships—something that’s actually useful beyond just "good accuracy numbers".
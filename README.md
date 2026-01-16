# 📈 Cervical Cancer Risk Detection — End-to-End ML Pipeline

**Author:** Juvénis Kaboré  
EFREI Paris | Data Engineering & AI Student  
🌍 Currently in Malaysia  
🔗 [LinkedIn](https://www.linkedin.com/in/fortune-kabore) • [GitHub](https://github.com/Spykabore15)

---

## 🩺 Project Overview

This repository provides a robust **end-to-end machine learning pipeline** for predicting **cervical cancer risk** using clinical and behavioral factors. The pipeline leverages **Microsoft Fabric** for orchestration and reproducibility, integrates with Git for version control, and is built with production readiness in mind.

### Key Features
- **Modular cloud-native workflow**: ETL, modeling, and deployment stages are clearly separated & automated.
- **Clinical interpretability**: Integration of SHAP for explaining model decisions.
- **Deployment ready**: Exported CatBoost model compatible with API, dashboard, and pipeline integration.

---

## 🗂️ Repository Structure

```plaintext
Cervical-Cancer-Risk-Detection/
├── Dataflow_2/                   # Data preprocessing with Dataflow Gen2
│   └── cervical_cancer_data_cleaned  # Clean dataset (output)
├── data_training_notebook.py     # Research notebook (EDA, feature selection, model benchmarking)
├── Final_notebook.py             # Production notebook (CatBoost training + export)
├── Final_pipeline/               # Fabric orchestration scripts
│   ├── Dataflow_2                # ETL execution
│   └── Final_notebook.py         # Model training & registry
└── README.md                     # Documentation
```

---

## 🖥️ Microsoft Fabric Workspace: `cancer_detection_analysis`

- **Dataflow** — ETL & Data Cleaning
- **Notebooks** — Research, training, and deployment pipeline
- **Pipeline** — Workflow automation within Fabric
- **Lakehouse** — Unified storage for analytics and models
- **Git Integration** — Collaboration & version control

---

## ⚙️ End-to-End Workflow

### 1️⃣ Data Preprocessing (Dataflow Gen2)
- Missing value handling
- Outlier filtering
- Feature normalization & schema validation
- Exporting cleaned dataset to Lakehouse

### 2️⃣ Model Development (Research Notebook)
- **EDA**: Visualization of risk factors with pandas, seaborn, matplotlib
- **Feature selection:** RandomForest + SelectFromModel
- **Class imbalance:** SMOTE, ADASYN, RandomOverSampler (imbalanced-learn)
- **Model comparison:** RandomForest, XGBoost, CatBoost, SVM
- **Interpretability:** SHAP values for important features
- **Outcome:** `CatBoostClassifier` selected for performance & explainability

### 3️⃣ Pipeline Orchestration (Fabric)
- **Automation:** Microsoft Fabric orchestrates ETL + model training + registry in one pipeline
- **Sequence:** Dataflow → Notebook (training) → Export model

---

## 🤖 ML Model & Deployment

- **Best Model:** CatBoostClassifier (.cbm format)
- **Key features:** Age, Number of pregnancies, Smoking history, HPV test results, etc.
- **Interpretability:** SHAP plots highlight feature contribution (see notebooks for examples)
- **Integration options:**
    - 🟢 REST APIs (Flask, FastAPI)
    - 🟣 Clinical dashboards
    - ☁️ Azure ML or Microsoft Fabric batch inference

---

## 🛠️ Tech Stack

| Category | Tools & Frameworks |
|----------|--------------------|
| Platform | Microsoft Fabric (Dataflow Gen2, Pipelines, Lakehouse, Git Integration) |
| Language | Python |
| Libraries | pandas, scikit-learn, imbalanced-learn, xgboost, catboost, shap |
| Visualization | matplotlib, seaborn |
| MLOps | Git integration, automated Fabric pipelines |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.8+
- Access to Microsoft Fabric workspace with Dataflow, Lakehouse, and Pipeline capabilities

### Setup

```bash
# Clone repository
git clone https://github.com/Spykabore15/Cervical-Cancer-Risk-Detection.git

# (Optional) Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn imbalanced-learn xgboost catboost shap matplotlib seaborn

# Follow notebooks in order (EDA → feature selection → modeling → export)
# See Dataflow_2/ for preprocessing logic
```

### Running Model Training

1. Execute `Dataflow_2` for cleansing and export data to Lakehouse.
2. Open `data_training_notebook.py` and work through EDA, feature selection, class imbalance, and model benchmarking.
3. Use `Final_notebook.py` for CatBoost model training and export.
4. Exported models can be integrated into APIs, dashboards, or batch inference pipelines.

---

## 🧪 Testing

- **Validation:** Accuracy, F1, confusion matrix, ROC-AUC reported in research/model notebooks.
- **Robustness:** Next steps include testing with external datasets, monitoring, and continuous retraining via MLOps.

---

## 📓 Example Results

Feature Importance Example (SHAP plot):
```
Top features: Age, NumOfPregnancies, SmokesPacksYear, DxCancer, HPV related
Model: CatBoostClassifier
See `Final_notebook.py` / SHAP summary plots for explainability
```

Sample performance:
- F1 score (cross-validated): *provided in confusion matrices in notebooks*
- Interpretability: *SHAP values and EDA shown in included notebooks*

---

## 📋 Next Steps

- Hyperparameter tuning and model selection refinement
- 🐳 Containerization with Docker + Azure ML
- 🔁 Automated retraining & drift monitoring
- 🧑‍⚕️ Real-time API for clinical application

---

## 👨‍💻 Author

**Juvénis Kaboré**
EFREI Paris — Data Engineering & AI Student
🌏 [LinkedIn](https://www.linkedin.com/in/fortune-kabore) • [Portfolio](https://juvenis.lovable.app/)

*“Data becomes powerful when it drives meaningful change.”*

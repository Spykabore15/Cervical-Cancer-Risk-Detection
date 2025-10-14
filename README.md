
#📊 Cervical Cancer Detection – End-to-End ML Pipeline
📌 Project Overview

This project develops an end-to-end machine learning pipeline for predicting cervical cancer based on clinical and behavioral risk factors.

The pipeline is built and orchestrated in Microsoft Fabric with Git integration, ensuring reproducibility, automation, and version control.

Key steps include:

Data preprocessing using Dataflow Gen2 (Dataflow 2)

Exploratory Data Analysis (EDA) & model benchmarking in a dedicated research notebook

Final orchestrated pipeline combining Dataflow preprocessing and CatBoost model training

Deployment-ready model export for integration into downstream applications

📂 Repository & Workspace Structure
├── Dataflow_2/                        # Dataflow Gen2 pipeline (data preprocessing)
│   └── cervical_cancer_data_cleaned   # Cleaned dataset (output)
├── data training notebook.py           # Research notebook: EDA, feature selection, model comparison
├── Final_notebook.py                   # Final deployable ML model (CatBoost)
├── Final_pipeline/                     # Data pipeline orchestration
│   ├── Dataflow_2                      # Executes preprocessing
│   └── Final_notebook.py               # Runs final model training & saving
└── README.md                           # Project documentation (this file)


In Fabric Workspace (cancer_detection_analysis):

✅ Dataflow (ETL & cleaning)

✅ Notebooks (training & final pipeline)

✅ Pipeline (orchestration)

✅ Lakehouse (storage & analytics)

✅ Git integration (version control, collaboration)

⚙️ Workflow
1. Data Preprocessing – Dataflow 2

Data ingestion and cleaning via Fabric Dataflow Gen2.

Steps include: missing value handling, outlier filtering, data normalization.

Output stored as cervical_cancer_data_cleaned in Lakehouse.

2. Model Development – Research Notebook

Performed EDA with Pandas, Seaborn, Matplotlib.

Feature selection with RandomForest + SelectFromModel.

Class imbalance handling (RandomOverSampler, SMOTE, ADASYN).

Benchmarked models: RandomForest, XGBoost, CatBoost, SVM.

Feature importance & interpretability with SHAP values.

➡️ CatBoostClassifier selected as final model (best performance & interpretability).

3. Final Pipeline – Orchestration in Fabric

Pipeline runs sequentially:

Dataflow (cleaning)

Notebook (CatBoost training & evaluation)

Trained model saved as:

/lakehouse/default/Files/models/cancerPredictionModelDeployed_catBoost.cbm


Pipeline status: ✅ Succeeded (automated & repeatable).

📊 Results

Best model: CatBoostClassifier

Key metrics: Precision, Recall, F1-score, ROC-AUC

SHAP interpretability highlights key clinical risk factors (e.g., Age, Number of pregnancies, Smoking history, HPV tests).

🚀 Deployment

Model exported in .cbm format (CatBoost native).

Ready for integration into:

REST APIs (Flask / FastAPI)

Clinical dashboards

Batch inference pipelines in Azure ML / Fabric

🔧 Tech Stack

Microsoft Fabric (Dataflow Gen2, Pipelines, Lakehouse, Git integration)

Python: pandas, scikit-learn, imbalanced-learn, xgboost, catboost, shap

Visualization: matplotlib, seaborn

MLOps: Git integration, automated pipeline execution

📌 Next Steps

Containerize final pipeline with Docker + Azure ML

Automate retraining with Fabric Pipelines & MLOps best practices

Validate on external datasets for robustness

Develop a real-time inference API for clinical deployment

# 📊 Cervical Cancer Detection – End-to-End ML Pipeline  

**📍 EFREI Paris | Author: Juvénis Kaboré**  
🧠 *Predicting cervical cancer risk using machine learning and Microsoft Fabric orchestration.*

---

## 📌 Project Overview  

This project develops a complete **end-to-end machine learning pipeline** for predicting **cervical cancer** based on clinical and behavioral risk factors.  

The pipeline is **built and orchestrated in Microsoft Fabric** with **Git integration**, ensuring full reproducibility, automation, and version control across all stages of the workflow.  

### 🔑 Key Steps
- **Data preprocessing** using **Dataflow Gen2 (Dataflow 2)**  
- **Exploratory Data Analysis (EDA)** & model benchmarking in a dedicated research notebook  
- **Final orchestrated pipeline** combining Dataflow preprocessing and CatBoost model training  
- **Deployment-ready model export** for integration into downstream applications  

---

## 📂 Repository & Workspace Structure  

├── Dataflow_2/ # Dataflow Gen2 pipeline (data preprocessing)
│ └── cervical_cancer_data_cleaned # Cleaned dataset (output)
├── data_training_notebook.py # Research notebook: EDA, feature selection, model comparison
├── Final_notebook.py # Final deployable ML model (CatBoost)
├── Final_pipeline/ # Data pipeline orchestration
│ ├── Dataflow_2 # Executes preprocessing
│ └── Final_notebook.py # Runs final model training & saving
└── README.md # Project documentation


---

## 🧭 Microsoft Fabric Workspace: `cancer_detection_analysis`

✅ **Dataflow** – ETL & data cleaning  
✅ **Notebooks** – Training and final model pipeline  
✅ **Pipeline** – End-to-end orchestration  
✅ **Lakehouse** – Unified storage & analytics layer  
✅ **Git Integration** – Version control & collaboration  

---

## ⚙️ Workflow  

### 1️⃣ Data Preprocessing – *Dataflow Gen2*  
Data ingestion and cleaning performed through Fabric’s **Dataflow Gen2**.  
**Steps include:**  
- Handling missing values  
- Filtering outliers  
- Feature normalization and schema validation  
- Exporting the cleaned dataset (`cervical_cancer_data_cleaned`) to **Lakehouse**

---

### 2️⃣ Model Development – *Research Notebook*  
- Conducted **Exploratory Data Analysis (EDA)** with `Pandas`, `Seaborn`, and `Matplotlib`.  
- Performed **feature selection** using `RandomForest` + `SelectFromModel`.  
- Managed **class imbalance** via `SMOTE`, `ADASYN`, and `RandomOverSampler`.  
- Benchmarked multiple models: `RandomForest`, `XGBoost`, `CatBoost`, and `SVM`.  
- Used **SHAP values** for feature importance and interpretability.  
- **Result:** `CatBoostClassifier` selected as the **final model** for its superior performance and explainability.  

---

### 3️⃣ Final Pipeline – *Fabric Orchestration*  
A complete **Fabric pipeline** was developed to automate the full ML workflow:  

**Pipeline sequence:**  
1. Dataflow → Cleans and pre-processes the data  
2. Notebook → Trains and evaluates the CatBoost model  


**Best Model:** `CatBoostClassifier`  
**Interpretability:** SHAP values highlighted critical risk factors including *Age*, *Number of pregnancies*, *Smoking history*, and *HPV test results*.  

---

## 🚀 Deployment  

The final model (`.cbm` format) is **deployment-ready** and can be integrated into:  
- 🌐 **REST APIs** (Flask / FastAPI)  
- 📈 **Clinical dashboards** for healthcare analytics  
- ☁️ **Batch inference pipelines** in Azure ML / Fabric  

---

## 🔧 Tech Stack  

| Category | Tools & Frameworks |
|-----------|--------------------|
| Platform | Microsoft Fabric (Dataflow Gen2, Pipelines, Lakehouse, Git Integration) |
| Language | Python |
| Libraries | pandas, scikit-learn, imbalanced-learn, xgboost, catboost, shap |
| Visualization | matplotlib, seaborn |
| MLOps | Git integration, automated Fabric pipelines |

---

## 📌 Next Steps  

- Parameters tunning
- 🐳 Containerize the final pipeline with **Docker + Azure ML**  
- 🔁 Automate retraining and monitoring via **Fabric Pipelines (MLOps best practices)**  
- 🧪 Validate model performance on **external datasets** for robustness  
- 🌐 Develop a **real-time inference API** for clinical deployment  

---

## 🧠 Key Learnings  

- Building modular **end-to-end ML pipelines** using cloud-native tools  
- Applying **data engineering principles** in healthcare data processing  
- Balancing model accuracy, interpretability, and automation  
- Leveraging **Microsoft Fabric** for reproducible and scalable ML workflows  

---

## 👤 Author  

**Juvénis Kaboré**  
🎓 Data Engineering & AI Student – EFREI Paris  
📍 Currently in Malaysia | Passionate about AI, MLOps, and cloud data platforms  
🔗 [LinkedIn](https://www.linkedin.com/in/fortune-kabore) • [GitHub](https://github.com/Spykabore15)

---

⭐ *“Data becomes powerful when it drives meaningful change.”*

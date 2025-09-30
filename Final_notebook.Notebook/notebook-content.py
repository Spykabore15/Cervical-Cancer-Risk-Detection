# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "b067b728-cb34-43d6-a21c-50593a225270",
# META       "default_lakehouse_name": "lkhouse_1",
# META       "default_lakehouse_workspace_id": "1d3edf85-29fb-4335-8d73-4dd50e998b19",
# META       "known_lakehouses": [
# META         {
# META           "id": "b067b728-cb34-43d6-a21c-50593a225270"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load the data
spark_df = spark.read.table("cervical_cancer_data_cleaned")
display(spark_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pandas_df = spark_df.toPandas()
filtered_df = pandas_df[
    (pandas_df["Age"] < 55) &
    (pandas_df["NumOfPregnancies"] < 7) &
    (pandas_df["SmokesYears"] < 5) &
    (pandas_df["SmokesPacksYear"] < 21) &
    (pandas_df["IUDYears"] < 12) &
    (pandas_df["STDsNumber"] < 5) &
    (pandas_df["STDsNumberOfDiagnosis"] < 2)
]
filtered_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def optimize_memory(df):
    df = df.copy()  # Ensure we're not modifying a view
    for col in df.columns:
        if df[col].dtype == 'float64':
            df.loc[:, col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df.loc[:, col] = df[col].astype('int32')
    return df

filtered_df = optimize_memory(filtered_df)
print("Memory optimized")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Split the dataset
y = filtered_df["Biopsy"]
X = filtered_df.drop(columns = ["Biopsy"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Apply SelectFromModel
selector_model = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)

X_train_model = selector_model.fit_transform(X_train, y_train)
X_test_model = selector_model.transform(X_test)

#Retrieve the names of the selected columns
selected_columns = selector_model.get_feature_names_out()

# Recreate dataframes
X_train = pd.DataFrame(X_train_model, columns=selected_columns)
X_test = pd.DataFrame(X_test_model, columns=selected_columns)

#Selected columns
print(selected_columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Model training

catboost = CatBoostClassifier(
    auto_class_weights='Balanced', 
    random_seed=42,
    depth=5,
    verbose=0, 
    thread_count=1
)

print("Training...")
catboost.fit(X_train, y_train)

y_pred_catboost = catboost.predict(X_test)

print(classification_report(y_test, y_pred_catboost))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Features importances analysis

explainer = shap.TreeExplainer(catboost)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type='bar')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save the model
catboost.save_model('/lakehouse/default/Files/models/cancerPredictionModelDeployed_catBoost.cbm')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

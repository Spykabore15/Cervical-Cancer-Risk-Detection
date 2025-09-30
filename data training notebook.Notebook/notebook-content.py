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

# Load the data
spark_df = spark.read.table("cervical_cancer_data_cleaned")
display(spark_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pandas_df = spark_df.toPandas()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

numerical_columns = ['Age', 'NumberOfSexualPartners', 'FirstSexualIntercourse',
       'NumOfPregnancies', 'SmokesYears', 'SmokesPacksYear', 'HormonalContraceptivesYears', 'STDsNumber', 'IUDYears', 'STDsNumberOfDiagnosis']
for v in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=pandas_df, y=v)
    plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

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

categorical_columns = ['STDsNumber','HormonalContraceptives', 'IUD', 'STDs', 'STDsCondylomatosis',
       'STDsVulvoPerinealCondylomatosis', 'STDsGenitalHerpes', 'STDsHIV', 'DxCancer', 'DxHPV', 'Dx',
       'Hinselmann', 'Schiller', 'Citology', 'Biopsy']

for v in categorical_columns:
    plt.figure(figsize=(5,3))
    sns.boxplot(data=filtered_df, x=v)
    plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Remarque :**
# This behavior of the categorical variables is normal and expected. We have to deal with them like this and no further transformations.

# CELL ********************

corr_matrix = filtered_df.corr()
plt.figure(figsize=(16, 8))  # Taille de la figure
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _There are some correlations and that doesn't shock us. Still, I prefer keep all the variables for now._

# MARKDOWN ********************

# **Let's optimize the memory**

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

# MARKDOWN ********************

# **Time to split the dataset**:
#  _We'll apply **stratify = y** since we are dealing with an imbalance dataset in order for the minority class to be well distributed_

# CELL ********************

# Split the dataset
from sklearn.model_selection import train_test_split

y = filtered_df["Biopsy"]
X = filtered_df.drop(columns = ["Biopsy"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_train.value_counts(normalize=True)*100

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_test.value_counts(normalize=True)*100

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Well distributed.**
# - **Features importances :**
#  _Even after all of these preprocessing, there is still 25 variables to deal with, which are quiet a lot. We'll now try to identify and select only the important features by using **SelectFromModel** and **RandomForestClassifier**_

# CELL ********************

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

# MARKDOWN ********************

# **Selected features :**
# 1. 'Age' 
# 2. 'NumberOfSexualPartners'
# 3. 'FirstSexualIntercourse'
# 4. 'NumOfPregnancies'
# 5. 'SmokesPacksYear'
# 6. 'HormonalContraceptives'
# 7. 'HormonalContraceptivesYears'
# 8. 'IUDYears'
# 9. 'STDsGenitalHerpes'
# 10. 'DxCancer'
# 11. 'Hinselmann'
# 12. 'Schiller'
# 13. 'Citology'

# MARKDOWN ********************

# _Let's now deal with the problem of imbalance.
# Let's try **RandomOverSampling, SMOTE and ADASYN** and see which one perform best_.

# CELL ********************

!pip install imbalanced-learn
!pip install -U --force-reinstall scikit-learn==1.3.2 imbalanced-learn==0.11.0


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

print("Before oversampling: ", Counter(y_train) )
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
print("After oversampling:", Counter(y_res))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_res.value_counts(normalize=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from imblearn.over_sampling import SMOTE
from collections import Counter
print("Before oversampling: ", Counter(y_train) )
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("After oversampling:", Counter(y_train_sm))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from imblearn.over_sampling import ADASYN
from collections import Counter

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Check new class distribution
print("After ADASYN oversampling:", Counter(y_resampled))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _**Great !**_

# MARKDOWN ********************

# **Time for model selection:** The three models will be tested 
# 1. RandomForestClassifier
# 2. XgBoost
# 3. CatBoostClassifier
# 4. SVM


# MARKDOWN ********************

# _And will be trained on all the generated training ressources and also to the pure data._

# CELL ********************

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.stats import randint, uniform



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Random Forest Classifier

# CELL ********************

rf_model = RandomForestClassifier(class_weight='balanced',random_state=42)
params_dist = {
    "n_estimators": randint(100,500),
    "max_depth":[15, 18, 19, 20, None],
    "max_features": ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_clf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=params_dist, 
    n_iter=10, 
    cv=3,
     random_state=42,
     n_jobs=1
)
rf_clf.fit(X_res, y_res)

#Best score achieved
print(f"Best cross-validation score: {rf_clf.best_score_:.4f}")

# Best parameters found
print("Best parameters:", rf_clf.best_params_)
y_pred = rf_clf.predict(X_test)

y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)

clf_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Classification Report:\n {clf_report}")
print("\nConfusion Matrix:")
print(conf_matrix)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Features importances analysis
import shap


explainer = shap.TreeExplainer(rf_clf.best_estimator_)
shap_values = explainer.shap_values(X_res)

shap.summary_plot(shap_values, X_res, plot_type='bar', plot_size=(12,6))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _**After training all the ressources, the random forest data turns out to be the one that perform best.**_

# MARKDOWN ********************

# ## XbBoost

# CELL ********************

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


scale_pos_weight = y_train.value_counts(normalize=True)[0]/y_train.value_counts(normalize=True)[1]
print(f"Origin imbalance ratio : {scale_pos_weight:.2f}")

xgb_simple = XGBClassifier(
    random_state=42
)
xgb_simple.fit(X_res, y_res)
y_pred_xg = xgb_simple.predict(X_test)
print(classification_report(y_test, y_pred_xg))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Features importances analysis

explainer = shap.TreeExplainer(xgb_simple)
shap_values = explainer.shap_values(X_res)
shap.summary_plot(shap_values, X_res, plot_type='dot', plot_size=(10,6))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.model_selection import RandomizedSearchCV


scale_pos_weight = y_train.value_counts(normalize=True)[0]/y_train.value_counts(normalize=True)[1]
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 6, 7, 10],
    
}

# RandomizedSearchCV pour XGBoost
xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=42,
    ),
    param_distributions=param_dist_xgb,
    n_iter=20, 
    cv=4,
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("Hyperparameters search loading...")
xgb_search.fit(X_res, y_res)
y_pred_xg2 = xgb_search.predict(X_test)
print(classification_report(y_test, y_pred_xg2 ))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Features importances analysis

explainer = shap.TreeExplainer(xgb_search.best_estimator_)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type='bar')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _**Despites hyperparameters tweeking, we didn't get a better performance than that with this model**_

# MARKDOWN ********************

# ### CatBoostClassifier

# CELL ********************


from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix


catboost_simple = CatBoostClassifier(
    auto_class_weights='Balanced', 
    random_seed=42,
    depth=5,
    verbose=0, 
    thread_count=1
)

print("Training...")
catboost_simple.fit(X_train, y_train)

y_pred_catboost = catboost_simple.predict(X_test)

print(classification_report(y_test, y_pred_catboost))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _Exactly what we were looking for. We finally find our model_

# CELL ********************

# Features importances analysis

explainer = shap.TreeExplainer(catboost_simple)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type='bar')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Let's _try to tune a little bit the hyperparameters_

# CELL ********************

from catboost import CatBoostClassifier


# Improved version of catboost
catboost_improved = CatBoostClassifier(
    auto_class_weights ='Balanced',
    learning_rate=0.1,
    depth=5,
    iterations=1000,
    early_stopping_rounds=50,
    eval_metric='F1',
    random_seed=42,
    verbose=100,  # See progression
    thread_count=1
)

print("Training in progress...")
catboost_improved.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred_catboost = catboost_improved.predict(X_test)


print(classification_report(y_test, y_pred_catboost))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Features importances analysis

explainer = shap.TreeExplainer(catboost_improved)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type='bar')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **_The hyperparameters tuning doesn't change the output. We may have reached the maximum performance _**

# MARKDOWN ********************

# ### SVM

# CELL ********************

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# SVM is sensible in features' scaling. So,
scaler = StandardScaler()

#Standardisation
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel='linear',
    class_weight='balanced',  
    probability=True,    
    random_state=42
)

print("Training...")
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print(classification_report(y_test, y_pred_svm))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# _This model also performs well, but a little bit least than CatBoostClassifier._

# MARKDOWN ********************

# _**I tried to finetune the kernel hyperparameter to 'rbf' and 'poly', but its the best the model can do.**_

# MARKDOWN ********************

# ### Conclusion
# We will choose the CatBoostClassifier as final model.
# Let's save the model

# CELL ********************


catboost_improved.save_model('/lakehouse/default/Files/models/cancerPredictionModel_catBoost.cbm')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

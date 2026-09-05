import pandas as pd

df =pd.read_csv("diabetic_data.csv")
print(df)
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df["readmitted"].value_counts())

df =df.replace("?" ,pd.NA)
missing =df.isnull().mean().sort_values(ascending =False) *100
print("Top missing columns:  ")
print(missing.head(10))

df["readmitted_binary"] =(df["readmitted"] == "<30").astype(int)

print(df["readmitted_binary"].value_counts())
print("persentage posetive class: " ,df["readmitted_binary"].mean() *100 , "%")

columns_to_drop =["weight","max_glu_serum","A1Cresult","medical_specialty","payer_code","encounter_id","patient_nbr","readmitted"]

df_clean =df.drop(columns= columns_to_drop)
print("shape after dropping" ,df_clean)

numeric_columns =df_clean.select_dtypes(include=["int64","float64"]).columns.tolist()
numeric_columns.remove("readmitted_binary")

for columns in numeric_columns :
    df_clean[columns] =df_clean[columns].fillna(df_clean[columns].mean())

categorical_columns =df_clean.select_dtypes(include=["object"]).columns.tolist()
for columns in categorical_columns :
    df_clean[columns] =df_clean[columns].fillna(df_clean[columns].mode()[0])

print(df_clean.isnull().sum().sum())
print("Final columns: ",df_clean.columns.tolist())
print("fibal shape: " ,df_clean.shape)

x =df_clean.drop(columns=["readmitted_binary"])
y =df_clean["readmitted_binary"]

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
categorical_columns =x.select_dtypes(include=["object"]).columns.tolist()
categorical_columns =x.select_dtypes(include=["int64" ,"float64"]).columns.tolist()
encoder = OneHotEncoder(drop="first" ,handle_unknown= "ignore" ,sparse_output=False)
encoded_array =encoder.fit_transform(x[categorical_columns])
encoded_names =encoder.get_feature_names_out(categorical_columns)
encoded =pd.DataFrame(encoded_array ,columns=encoded_names ,index=x.index)
x_encoded =pd.concat([x[numeric_columns],encoded],axis =1)
print(x.shape)
print(x_encoded.shape)

# Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x_encoded,y ,random_state= 42 ,test_size= 0.2 ,stratify= y)

print("Train shape: ",x_train.shape)
print("Test shape: ",x_test.shape)
print("Train target distribution: ",y_train.value_counts(normalize =True) *100)
print("Test target distribution: ",y_test.value_counts(normalize =True) *100)

#Scale 
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
x_train_scaled =scaler.fit_transform(x_train)
x_test_scaled =scaler.transform(x_test)


#logestic Regression
from sklearn.linear_model import LogisticRegression
log_reg =LogisticRegression(max_iter= 2000 ,class_weight= "balanced" ,random_state= 42)
log_reg.fit(x_train_scaled , y_train)
y_pred = log_reg.predict(x_test_scaled)
y_proba =log_reg.predict_proba(x_test_scaled)[: , 1]

#metrics 
from sklearn.metrics import accuracy_score ,recall_score , precision_score ,f1_score ,confusion_matrix ,roc_auc_score
print("Accuracy: " ,accuracy_score(y_pred ,y_test))
print("Recall:  ",recall_score(y_pred ,y_test))
print("Precision:  ",precision_score(y_pred ,y_test))
print("f1_score:  ",f1_score(y_pred ,y_test))
print("Confusion Matrix:  ",confusion_matrix(y_pred ,y_test))

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(n_estimators= 100 , class_weight= "balanced" ,random_state=42 ,n_jobs= 1)
rf.fit(x_train ,y_train)
y_pred_rf =rf.predict(x_test)
y_proba_rf =rf.predict_proba(x_test)[: ,1]

print("Random Forest Results")
print("Accuracy: ",accuracy_score(y_test, y_pred_rf))
print("Precision: ",precision_score(y_test ,y_pred_rf))
print("Recall: ",recall_score(y_test ,y_pred_rf))
print("f1_score: ",f1_score(y_test ,y_pred_rf))
print("Confusion Matrix: ",confusion_matrix(y_test ,y_pred_rf))
print("AUS-ROC: ",roc_auc_score(y_test ,y_proba_rf))

#XGBoost
from xgboost import XGBClassifier
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb =XGBClassifier (n_estimators = 100 ,scale_pos_weight =scale_pos_weight ,random_state =42 ,eval_metric ="logloss")
xgb.fit(x_train,y_train)
y_pred_xgb =xgb.predict(x_test)
y_proba_xgb =xgb.predict_proba(x_test)[: ,1]

print("XGBoost Result")
print("Accuracy: ",accuracy_score(y_test,y_pred_xgb))
print("Recall: ",recall_score(y_test,y_pred_xgb))
print("f1_score: ",f1_score(y_test,y_pred_xgb))
print("Precision: ",precision_score(y_test,y_pred_xgb))
print(confusion_matrix(y_test,y_pred_xgb))
print("AUC-ROC: ",roc_auc_score(y_test,y_proba_xgb))

# DDN
from tensorflow import keras
from tensorflow.keras import layers

dnn =keras.Sequential([layers.Input(shape=(x_train_scaled.shape[1],)),
                       layers.Dense(64,activation="relu"),
                       layers.Dropout(0.3),
                       layers.Dense(32,activation="relu"),
                       layers.Dropout(0.3),
                       layers.Dense(1,activation= "sigmoid")
                       ])

dnn.compile(optimizer ="adam" ,loss ="binary_crossentropy",metrics =["accuracy"])
class_weight_dict ={0 : 1.0 , 1 :(y_train == 0).sum() / (y_train == 1).sum()}
history =dnn.fit(x_train_scaled ,y_train ,epochs = 20 ,batch_size =256 , validation_split =0.1 ,class_weight =class_weight_dict ,verbose =1)

y_proba_dnn =dnn.predict(x_test_scaled).flatten()
y_pred_dnn =(y_proba_dnn >= 0.5).astype(int)

print("DNN results")
print("Accuracy: ",accuracy_score(y_test ,y_pred_dnn))
print("Recall: ",recall_score(y_test ,y_pred_dnn))
print("Percision: ",precision_score(y_test ,y_pred_dnn))
print("f1 score: ",f1_score(y_test ,y_pred_dnn))
print("AUC-ROC: ",roc_auc_score(y_test ,y_pred_dnn))
print(confusion_matrix(y_test ,y_pred_dnn))

# SHAP
import shap 
import matplotlib.pyplot as plt

explainer =shap.TreeExplainer(xgb)
x_test_sample =x_test.sample(1000, random_state =42)
shap_values =explainer.shap_values(x_test_sample)

plt.figure()
shap.summary_plot(shap_values ,x_test_sample ,show =False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png",dpi = 150)
print("Shap summary plot saved as shap_summary_plot.png")

import numpy as np
features_importance =np.abs(shap_values).mean(axis = 0)
top_features =pd.Series(features_importance ,index =x_test_sample.columns).sort_values(ascending=False)
print("Top 10 Most important features",top_features.head(10))

import joblib
joblib.dump(log_reg ,"model_logestic_regression.pkl")
joblib.dump(rf ,"model_random_forest.pkl")
joblib.dump(xgb,"model_xgboost.pkl")
dnn.save("model_dnn.keras")
joblib.dump(scaler,"scaler.pkl")
print("All models saved")

#json
import json

results ={
    "Logestic Regression": {
        "accuracy": accuracy_score(y_test ,y_pred),
        "precision":precision_score(y_test ,y_pred),
        "recall":recall_score(y_test,y_pred),
        "f1 score":f1_score(y_test ,y_pred),
        "AUC_ROC":roc_auc_score(y_test,y_pred)
    },
    "Random Forest":{ 
            "accuracy": accuracy_score(y_test ,y_pred_rf),
            "precision":precision_score(y_test ,y_pred_rf),
            "recall":recall_score(y_test,y_pred_rf),
            "f1 score":f1_score(y_test ,y_pred_rf),
            "AUC_ROC":roc_auc_score(y_test,y_pred_rf)
    },
    "XGBoost":{
                "accuracy": accuracy_score(y_test ,y_pred_xgb),
                "precision":precision_score(y_test ,y_pred_xgb),
                "recall":recall_score(y_test,y_pred_xgb),
                "f1 score":f1_score(y_test ,y_pred_xgb),
                "AUC_ROC":roc_auc_score(y_test,y_pred_xgb)
    },
    "DNN":{
                "accuracy": accuracy_score(y_test ,y_pred_dnn),
                "precision":precision_score(y_test ,y_pred_dnn),
                "recall":recall_score(y_test,y_pred_dnn),
                "f1 score":f1_score(y_test ,y_pred_dnn),
                "AUC_ROC":roc_auc_score(y_test,y_pred_dnn)
    }
}

with open("results.json" ,"w") as f :
    json.dump(results,f,indent= 4)
print("results.json")


x_test.head(20).to_csv("x_test_sample.csv" ,index =False)
y_test.head(20).to_csv("y_test_sample.csv" ,index =False)

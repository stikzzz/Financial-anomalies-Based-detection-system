import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

df = pd.read_csv("fraud_data.xls")

df = df[["nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]]
df = df.sample(n=100000, random_state=42)

df = df.dropna()

# Feature Engineering
df['transactionAmount'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceChangeRatioOrig'] = (df['oldbalanceOrg'] - df['newbalanceOrig']) / (df['oldbalanceOrg'] + 1)
df['balanceChangeRatioDest'] = (df['oldbalanceDest'] - df['newbalanceDest']) / (df['oldbalanceDest'] + 1)

df = df[df['transactionAmount'] >= 0]

# Encode categorical variables
label_encoder_orig = LabelEncoder()
label_encoder_dest = LabelEncoder()
df['nameOrig'] = label_encoder_orig.fit_transform(df['nameOrig'])
df['nameDest'] = label_encoder_dest.fit_transform(df['nameDest'])

# Define features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Assuming 'model' is your trained RandomForestClassifier
filename = 'detection.pkl'

# Open a file for writing in binary mode
with open(filename, 'wb') as file:
    # Use pickle.dump to serialize the model to the file
    pickle.dump(model, file)

# Open the file for reading in binary mode and load the model
with open('detection.pkl', 'rb') as file:
    detection_model = pickle.load(file)

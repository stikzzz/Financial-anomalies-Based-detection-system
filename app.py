from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key

# Load the model
with open('detection.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
df = pd.read_csv("fraud_data.xls")  # Changed to read CSV

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    nameDest = request.form['nameDest']
    type = request.form['type']
    amount = float(request.form['amount'])

    # Filter the dataset based on the nameDest
    user_data = df[df['nameDest'] == nameDest]

    if user_data.empty:
        return jsonify({'prediction': 'none', 'message': 'No transactions found for the provided nameDest.'})

    fraud_transactions = user_data[user_data['isFraud'] == 1]

    if not fraud_transactions.empty:
        return jsonify({'prediction': 'fraud', 'message': 'Warning: The user has one or more fraudulent transactions. Please review the transactions for further action.'})
    else:
        return jsonify({'prediction': 'safe', 'message': 'Good news! No fraudulent transactions found for this user. You are safe to proceed.'})

@app.route('/admin', methods=['POST'])
def admin():
    nameDest = request.form['nameDest']
    
    # Update the dataset
    global df
    if nameDest in df['nameDest'].values:
        df.loc[df['nameDest'] == nameDest, 'isFraud'] = 1
    else:
        new_data = pd.DataFrame({
            'nameOrig': [None],
            'oldbalanceOrg': [None],
            'newbalanceOrig': [None],
            'nameDest': [nameDest],
            'oldbalanceDest': [None],
            'newbalanceDest': [None],
            'isFraud': [1]
        })
        df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the updated dataset
    df.to_csv("fraud_data.xls", index=False)
    
    # Retrain the model
    retrain_model()

    # Return to index page with success message
    flash('Successfully marked as fraud and retrained the model.')
    return redirect(url_for('home'))

def retrain_model():
    global model
    df = pd.read_csv("fraud_data.xls")

    df = df[["nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]]
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

    # Save the updated model
    filename = 'detection.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    app.run(debug=True)

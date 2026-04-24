import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split

# Load dataset and fit scaler on training data
dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
x = dataset.iloc[:, : 10].values
y = dataset.iloc[:, 10].values

# Split data like in the notebook
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

# Fit scaler on training data only
scaler = StandardScaler()
scaler.fit(x_train)

model = tf.keras.models.load_model('filesuse/project_model1.h5')

app = Flask(__name__)

def detect_fraud_hybrid(features):
    """
    Hybrid fraud detection using both ML model and rule-based approach
    features: [hour, day, month, year, category, upi_number, age, amount, state, zip]
    """
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = features
    
    # Get ML model prediction
    x_scaled = scaler.transform([features])
    y_pred = model.predict(x_scaled, verbose=0)
    ml_score = y_pred[0][0]
    
    # Rule-based fraud indicators - AGGRESSIVE DETECTION
    fraud_score = 0
    fraud_reasons = []
    
    # Rule 1: Very high transaction amount (> 200)
    if v8 > 200:
        fraud_score += 0.5
        fraud_reasons.append(f"Very high amount: ₹{v8}")
    
    # Rule 2: High transaction amount (> 100)
    elif v8 > 100:
        fraud_score += 0.3
        fraud_reasons.append(f"High amount: ₹{v8}")
    
    # Rule 3: Unusual transaction time (0-6 AM)
    if 0 <= v1 <= 6:
        fraud_score += 0.3
        fraud_reasons.append(f"Unusual time: {v1}:00 AM")
    
    # Rule 4: Multiple high-risk factors
    if v8 > 80 and 0 <= v1 <= 6:
        fraud_score += 0.3
        fraud_reasons.append("High amount + unusual time")
    
    # Rule 5: Very high amount at any time
    if v8 > 250:
        fraud_score += 0.2
        fraud_reasons.append(f"Extremely high amount: ₹{v8}")
    
    # Combine ML and rule-based scores - RULE-BASED WEIGHTED MORE
    final_score = (ml_score * 0.3) + (fraud_score * 0.7)
    
    print(f"\n=== FRAUD DETECTION ANALYSIS ===")
    print(f"Transaction Amount: ₹{v8}")
    print(f"Transaction Time: {v1}:00")
    print(f"ML Model Score: {ml_score:.4f}")
    print(f"Rule-Based Score: {fraud_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"Fraud Indicators: {fraud_reasons if fraud_reasons else 'None'}")
    print(f"Threshold: 0.5")
    
    return final_score, fraud_reasons

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
        v1 = trans_datetime.hour
        v2 = trans_datetime.day
        v3 = trans_datetime.month
        v4 = trans_datetime.year
        v5 = int(request.form.get("category"))
        v6 = float(request.form.get("card_number"))
        dob = pd.to_datetime(request.form.get("dob"))
        v7 = np.round((trans_datetime - dob).days / 365.25)
        v8 = float(request.form.get("trans_amount"))
        v9 = int(request.form.get("state"))
        v10 = int(request.form.get("zip"))
        
        x_test_input = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
        
        # Get fraud detection score
        final_score, fraud_reasons = detect_fraud_hybrid(x_test_input)
        
        # Determine result based on threshold
        # Threshold: 0.5 means fraud
        if final_score > 0.5:
            result = "FRAUD TRANSACTION"
            print(f"Result: FRAUD (score {final_score:.4f} > 0.5)")
        else:
            result = "VALID TRANSACTION"
            print(f"Result: VALID (score {final_score:.4f} <= 0.5)")
        
        print("=" * 35 + "\n")
        
        return render_template('result.html', OUTPUT='{}'.format(result))
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', OUTPUT='ERROR: Invalid input')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)



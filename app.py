import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ------------------ API CONFIG ------------------
# 🔴 PASTE YOUR REAL NVIDIA API KEY BELOW (from https://integrate.api.nvidia.com)
API_KEY = "YOUR_NVIDIA_API_KEY_HERE"

# ✅ Use a valid NVIDIA NIM model — options: meta/llama-3.1-8b-instruct, mistralai/mistral-7b-instruct-v0.3
MODEL_NAME = "meta/llama-3.1-8b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= "Your API KEY"
)

# ------------------ KERAS FIX ------------------
from keras.layers import InputLayer
original_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        batch_shape = kwargs.pop("batch_shape")
        kwargs["input_shape"] = batch_shape[1:]
    return original_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

from keras.utils import custom_object_scope

# ------------------ LOAD DATA ------------------
dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)

x = dataset.iloc[:, :10].values
y = dataset.iloc[:, 10].values

x_train, _, _, _ = train_test_split(x, y, test_size=0.15, random_state=0)

scaler = StandardScaler()
scaler.fit(x_train)

# ------------------ LOAD MODEL ------------------
with custom_object_scope({
    'DTypePolicy': tf.keras.mixed_precision.Policy
}):
    model = tf.keras.models.load_model(
        'filesuse/project_model1.h5',
        compile=False
    )

# ------------------ FLASK ------------------
app = Flask(__name__)

# ------------------ FRAUD DETECTION ------------------
def detect_fraud_hybrid(features):
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = features

    x_scaled = scaler.transform([features])
    ml_score = model.predict(x_scaled, verbose=0)[0][0]

    fraud_score = 0

    if v8 > 200:
        fraud_score += 0.5
    elif v8 > 100:
        fraud_score += 0.3

    if 0 <= v1 <= 6:
        fraud_score += 0.3

    if v8 > 80 and 0 <= v1 <= 6:
        fraud_score += 0.3

    if v8 > 250:
        fraud_score += 0.2

    final_score = (ml_score * 0.3) + (fraud_score * 0.7)

    return final_score

# ------------------ AI EXPLANATION ------------------
def get_fraud_explanation(transaction_data):
    try:
        prompt = f"""You are a fraud detection assistant. Analyze this UPI transaction and explain in exactly 3-4 sentences why it appears fraudulent.

Transaction Details:
- Amount: ₹{transaction_data['amount']}
- Customer Age: {transaction_data['age']} years
- Transaction Hour: {transaction_data['hour']}:00

Write a single clear paragraph (3-4 sentences) explaining the fraud indicators. Be specific and concise. Do not use bullet points or headers."""

        print(f"[AI] Sending request to NVIDIA API with model: {MODEL_NAME}")

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fraud detection expert. Always respond with a single paragraph of 3-4 sentences. Never use bullet points."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200,
            stream=False
        )

        # ✅ Correctly extract the response text
        response_text = completion.choices[0].message.content

        print(f"[AI] Raw response:\n{response_text}")

        if response_text and response_text.strip():
            return response_text.strip()
        else:
            print("[AI] Empty response received.")
            return "AI explanation could not be generated (empty response)."

    except Exception as e:
        print(f"[AI ERROR] Type: {type(e).__name__}, Message: {e}")
        return f"AI explanation unavailable: {str(e)}"

# ------------------ ROUTES ------------------
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
        v6 = float(request.form.get("card_number")) % 1000

        dob = pd.to_datetime(request.form.get("dob"))
        v7 = np.round((trans_datetime - dob).days / 365.25)

        v8 = float(request.form.get("trans_amount"))
        v9 = int(request.form.get("state"))
        v10 = int(request.form.get("zip"))

        x_input = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])

        final_score = detect_fraud_hybrid(x_input)

        if final_score > 0.5:
            result = "FRAUD TRANSACTION"

            transaction_data = {
                "hour": v1,
                "age": int(v7),
                "amount": v8
            }

            explanation = get_fraud_explanation(transaction_data)

        else:
            result = "VALID TRANSACTION"
            explanation = "Transaction looks normal. No suspicious indicators were detected."

        return render_template(
            'result.html',
            OUTPUT=result,
            explanation=explanation
        )

    except Exception as e:
        print(f"[ROUTE ERROR] {type(e).__name__}: {e}")
        return render_template('result.html', OUTPUT="ERROR", explanation=str(e))

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
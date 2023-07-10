from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Define the feature order that the model and preprocessor expect
FEATURE_ORDER = [
    'age', 'job', 'marital_status', 'educational_level',
    'credit_default', 'annual_balance', 'housing_loan',
    'personal_loan', 'contact', 'day', 'month', 'duration',
    'present_campaign', 'pdays', 'previous_campaign', 'marketing_outcome'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = {k: [v] for k, v in request.form.items()}
    features_df = pd.DataFrame(features)
    features_df = features_df[FEATURE_ORDER]  # Ensure correct feature order
    features_df_preprocessed = preprocessor.transform(features_df)
    prediction = model.predict(features_df_preprocessed)

    if prediction[0] == 1:
        return render_template('index.html', prediction_text='The Customer will subscribe to a term deposit')
    else:
        return render_template('index.html', prediction_text='The Customer will not subscribe to a term deposit')


if __name__ == "__main__":
    app.run()


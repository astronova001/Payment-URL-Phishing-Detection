# importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

# Load all models
def load_models():
    models = {}
    model_names = [
        'logistic_regression', 'knn', 'svm', 'naive_bayes', 
        'decision_tree', 'random_forest', 'gradient_boosting',
        'catboost', 'xgboost', 'mlp'
    ]
    
    for model_name in model_names:
        with open(f"pickle/{model_name}.pkl", "rb") as file:
            models[model_name] = pickle.load(file)
    return models

# Load all models at startup
models = load_models()

# Add this after loading models
def load_accuracies():
    with open("pickle/model_accuracies.pkl", "rb") as file:
        return pickle.load(file)

# Load models and accuracies at startup
models = load_models()
model_accuracies = load_accuracies()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30)

        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            try:
                # Get prediction and probability
                y_pred = model.predict(x)[0]
                
                # Special handling for SVM (GridSearchCV)
                if model_name == 'svm':
                    decision_score = model.decision_function(x)[0]
                    confidence = (1 / (1 + np.exp(-decision_score))) * 100
                    predictions[model_name] = {
                        'prediction': 'Safe' if y_pred == 1 else 'Unsafe',
                        'probability': round(confidence, 2),
                        'accuracy': model_accuracies[model_name]
                    }
                else:
                    probabilities = model.predict_proba(x)[0]
                    safe_probability = probabilities[1] if y_pred == 1 else probabilities[0]
                    predictions[model_name] = {
                        'prediction': 'Safe' if y_pred == 1 else 'Unsafe',
                        'probability': round(safe_probability * 100, 2),
                        'accuracy': model_accuracies[model_name]
                    }
            except Exception as e:
                print(f"Error with {model_name} model:", str(e))
                predictions[model_name] = {
                    'prediction': f'Error: {str(e)}',           
                    'probability': 0,
                    'accuracy': model_accuracies[model_name]
                }

        # Sort predictions by accuracy
        sorted_predictions = dict(sorted(predictions.items(), 
                                       key=lambda x: x[1]['accuracy'], 
                                       reverse=True))

        return render_template('index.html', 
                             predictions=sorted_predictions,
                             url=url,
                             show_results=True)
    
    # For GET request, show sorted accuracies
    sorted_accuracies = dict(sorted(model_accuracies.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
    return render_template("index.html", 
                         show_results=False, 
                         model_accuracies=sorted_accuracies)

if __name__ == "__main__":
    app.run(debug=True)
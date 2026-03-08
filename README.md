# Iris Flower Classification using Machine Learning

This project builds a machine learning model to classify iris flower species based on their physical characteristics.

## Dataset
The project uses the Iris dataset containing four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target variable:
- Setosa
- Versicolor
- Virginica

## Project Workflow

1. Data Loading and Exploration
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing and Feature Scaling
4. Training Multiple Classification Models
5. Model Evaluation and Comparison
6. Saving the Best Model
7. Running Inference with the Saved Model

## Algorithms Used

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

## Model Saving

The best-performing model is saved using Joblib.

Saved file:

iris_best_model.pkl

## Example Prediction

```python
import joblib

model = joblib.load("iris_best_model.pkl")

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Species:", prediction[0])

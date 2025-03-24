# Iris Classification ML Demo with Observability

## What This App Does

This application demonstrates machine learning classification using the classic Iris dataset with a focus on model observability. Key features include:

- Training and evaluation of multiple classification algorithms:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - Neural Network
- Comprehensive model evaluation metrics including accuracy, confusion matrices, and classification reports
- Feature importance visualization
- Information Criteria (AIC/BIC) for model comparison
- Interactive prediction interface for real-time testing
- Detailed observability dashboard tracking model training, data transformations, and system performance

## How to Use

1. **Setup and Installation**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Launch the app with: `streamlit run Iris.py`

2. **ML Classification Demo Tab**:
   - View dataset overview with PCA visualization
   - Select a classification algorithm from the dropdown
   - Click "Train Model" to fit the model to the Iris dataset
   - Examine model performance metrics and visualizations
   - Use the "Make Your Own Prediction" section to interactively test the model with custom feature values

3. **Observability Dashboard Tab**:
   - Navigate through different observability views using the sub-tabs:
     - Execution Logs: Step-by-step record of operations
     - Data Snapshots: Statistical information about data at various transformation stages
     - Model Training: Detailed training metrics and history
     - Performance Metrics: Function execution times and bottlenecks
     - Feature Importance: Analysis of feature contributions to model predictions
     - Information Criteria: AIC/BIC comparison for multiple trained models

4. **Model Comparison**:
   - Train multiple models to enable model comparison visualizations
   - View comparative metrics to identify the best performing model based on different criteria

## What to Look For

- **Model Performance Variations**: Observe how different algorithms perform on the same dataset
- **Feature Importance**: Identify which features (sepal length, sepal width, petal length, petal width) have the strongest influence on classification
- **Complexity vs. Performance**: Use the Information Criteria tab to understand the tradeoff between model complexity and performance
- **Execution Timeline**: Monitor system performance through timing metrics
- **Data Transformations**: Track how data flows through the pipeline with data snapshots
- **Real-time Predictions**: See how your custom inputs affect predictions and where they fall within the feature space

The observability components demonstrate principles of ML monitoring that can be applied to more complex real-world systems.

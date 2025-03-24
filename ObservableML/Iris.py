import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import logging
import time
import uuid
import json
from datetime import datetime
import os
from sklearn.datasets import load_iris
# https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
# Learn from scratch https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/
import seaborn as sns
from scipy import stats
import math

# Algorithm imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression  # Added for enhancement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_observability")

# Create a session ID for traceability
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id

# Initialize state containers for observability
if "parameter_history" not in st.session_state:
    st.session_state.parameter_history = []

if "data_snapshots" not in st.session_state:
    st.session_state.data_snapshots = []

if "model_traces" not in st.session_state:
    st.session_state.model_traces = []

if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []

if "timing_metrics" not in st.session_state:
    st.session_state.timing_metrics = []

if "training_history" not in st.session_state:
    st.session_state.training_history = []

if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = []

# Initialize model comparison container for AIC/BIC
if "model_comparison" not in st.session_state:
    st.session_state.model_comparison = {}

# ========== OBSERVABILITY FUNCTIONS ==========

def log_execution(message, level="INFO"):
    """Record execution events in a structured format"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Log to Python logger
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    
    # Store in session state for UI display
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "session_id": SESSION_ID
    }
    st.session_state.execution_logs.append(log_entry)
    
    return log_entry

def track_parameters(params):
    """Record parameter changes for traceability"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a clean copy with only scalar values
    clean_params = {}
    for key, value in params.items():
        # Skip callable values
        if callable(value):
            continue
        # Convert numpy values to Python scalars
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                clean_params[key] = value.item()
            except (ValueError, AttributeError):
                clean_params[key] = str(value)
        else:
            clean_params[key] = value
    
    param_record = {
        "timestamp": timestamp,
        "session_id": SESSION_ID,
        "parameters": clean_params
    }
    
    st.session_state.parameter_history.append(param_record)
    log_execution(f"Parameters updated: {json.dumps(clean_params, default=str)}")
    return param_record

def record_data_snapshot(step_name, data, metadata=None):
    """Capture key data states at critical transformation points"""
    # Ensure data is not empty
    if data is None or len(data) == 0:
        log_execution(f"Warning: Empty data for snapshot '{step_name}'", level="WARNING")
        return None
        
    try:
        # Calculate statistics safely
        data_array = np.array(data)
        count = len(data_array)
        
        # Determine data type for appropriate analysis
        is_numeric = np.issubdtype(data_array.dtype, np.number)
        
        # Initialize with default values
        mean_val = min_val = max_val = median_val = std_val = 0.0
        value_distribution = None
        
        # Type-specific analysis
        if count > 0:
            if is_numeric:
                # Numerical statistics for numeric arrays
                mean_val = float(np.mean(data_array))
                min_val = float(np.min(data_array))
                max_val = float(np.max(data_array))
                median_val = float(np.median(data_array))
                std_val = float(np.std(data_array))
            else:
                # Categorical analysis for non-numeric arrays
                unique_values, counts = np.unique(data_array, return_counts=True)
                value_distribution = {str(val): int(count) for val, count in zip(unique_values, counts)}
                log_execution(f"Categorical data detected in '{step_name}', computing distribution instead of statistics", level="INFO")
        
        # Prepare clean metadata with only scalar values
        clean_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if hasattr(value, 'item') and callable(getattr(value, 'item')):
                    try:
                        clean_metadata[key] = value.item()
                    except (ValueError, AttributeError):
                        clean_metadata[key] = str(value)
                elif isinstance(value, (int, float, bool, str)):
                    clean_metadata[key] = value
                elif value is None:
                    clean_metadata[key] = None
                else:
                    clean_metadata[key] = str(value)
        
        # Create snapshot with type-appropriate metrics
        snapshot = {
            "step": step_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_type": str(data_array.dtype),
            "statistics": {
                "count": count,
                "mean": mean_val if is_numeric else None,
                "min": min_val if is_numeric else None,
                "max": max_val if is_numeric else None,
                "median": median_val if is_numeric else None,
                "std_dev": std_val if is_numeric else None
            },
            "categorical_distribution": value_distribution,
            "metadata": clean_metadata,
            "data_sample": data_array[:min(5, count)].tolist() if count > 0 else []
        }
        
        # Store snapshot
        st.session_state.data_snapshots.append(snapshot)
        
        # Log appropriate metrics based on data type
        if is_numeric:
            log_execution(f"Data snapshot '{step_name}' recorded: count={count}, mean={mean_val:.2f}, min={min_val}, max={max_val}")
        else:
            category_count = len(value_distribution) if value_distribution else 0
            log_execution(f"Data snapshot '{step_name}' recorded: count={count}, unique categories={category_count}")
        
        return snapshot
    except Exception as e:
        log_execution(f"Error recording data snapshot for '{step_name}': {str(e)}", level="ERROR")
        return None

def record_training_step(model_name, training_stage, metrics, iteration=None):
    """Record incremental training information for observability"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Clean metrics to ensure they're all serializable
        clean_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item') and callable(getattr(value, 'item')):
                try:
                    clean_metrics[key] = value.item()
                except (ValueError, AttributeError):
                    clean_metrics[key] = str(value)
            elif isinstance(value, (int, float, bool, str)):
                clean_metrics[key] = value
            elif value is None:
                clean_metrics[key] = None
            else:
                try:
                    # Try to convert to list if it's an array-like object
                    clean_metrics[key] = list(value)
                except:
                    clean_metrics[key] = str(value)
        
        # Create the training record
        training_record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "training_stage": training_stage,
            "iteration": iteration,
            "metrics": clean_metrics,
            "session_id": SESSION_ID
        }
        
        # Store the record
        st.session_state.training_history.append(training_record)
        
        # Log the training step
        log_execution(f"Training step recorded: {model_name} - {training_stage}")
        
        return training_record
    except Exception as e:
        log_execution(f"Error recording training step for '{model_name}': {str(e)}", level="ERROR")
        return None

def record_feature_importance(model_name, feature_names, importance_values, step="feature_importance"):
    """Record feature importance for a model"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if importance_values is None or len(importance_values) == 0:
            log_execution(f"Warning: Empty importance values for '{model_name}'", level="WARNING")
            return None
            
        if len(feature_names) != len(importance_values):
            log_execution(f"Error: Mismatch between feature names ({len(feature_names)}) and importance values ({len(importance_values)})", level="ERROR")
            return None
        
        # Ensure importance values are serializable
        clean_importance = []
        for name, value in zip(feature_names, importance_values):
            if hasattr(value, 'item') and callable(getattr(value, 'item')):
                try:
                    clean_importance.append({"feature": name, "importance": value.item()})
                except (ValueError, AttributeError):
                    clean_importance.append({"feature": name, "importance": float(0)})
            else:
                try:
                    clean_importance.append({"feature": name, "importance": float(value)})
                except (ValueError, TypeError):
                    clean_importance.append({"feature": name, "importance": float(0)})
        
        # Sort by importance (descending)
        clean_importance = sorted(clean_importance, key=lambda x: x["importance"], reverse=True)
        
        # Create the record
        importance_record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "step": step,
            "importance": clean_importance,
            "session_id": SESSION_ID
        }
        
        # Store the record
        st.session_state.feature_importance.append(importance_record)
        
        # Log the feature importance
        log_execution(f"Feature importance recorded for {model_name}")
        
        return importance_record
    except Exception as e:
        log_execution(f"Error recording feature importance for '{model_name}': {str(e)}", level="ERROR")
        return None

def trace_model_predictions(model, input_data, actual_labels=None, prediction_context=None):
    """Record model outputs and parameters for traceability"""
    # Always ensure we have valid arrays
    if input_data is None or len(input_data) == 0:
        log_execution("Cannot trace model predictions: input data is empty", level="WARNING")
        return None
    
    try:
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_data)
            except Exception as e:
                log_execution(f"Error getting prediction probabilities: {str(e)}", level="WARNING")
    except Exception as e:
        log_execution(f"Error during prediction: {str(e)}", level="ERROR")
        return None
    
    try:
        # Calculate accuracy if actual labels are provided
        accuracy = None
        if predictions is not None and actual_labels is not None:
            if len(predictions) == len(actual_labels):
                accuracy = float(accuracy_score(actual_labels, predictions))
            else:
                log_execution(f"Mismatch between predictions length ({len(predictions)}) and actual labels length ({len(actual_labels)})", level="WARNING")
        
        # Create prediction trace with scalar values only
        trace = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": type(model).__name__,
            "input_shape": str(input_data.shape),
            "session_id": SESSION_ID,
            "context": prediction_context or {},
            "accuracy": accuracy
        }
        
        # Add model parameters if available
        if hasattr(model, "get_params"):
            trace["parameters"] = model.get_params()
        
        # Extract a sample of predictions and probabilities
        if predictions is not None:
            sample_size = min(5, len(predictions))
            trace["prediction_sample"] = predictions[:sample_size].tolist()
            
            if probabilities is not None:
                sample_probs = probabilities[:sample_size].tolist()
                # Ensure values are native Python floats
                clean_probs = []
                for probs in sample_probs:
                    clean_probs.append([float(p) for p in probs])
                trace["probability_sample"] = clean_probs
        
        # Store trace
        st.session_state.model_traces.append(trace)
        log_execution(f"Model prediction traced: {type(model).__name__}, accuracy: {accuracy}")
        
        return trace
    except Exception as e:
        log_execution(f"Error tracing model predictions: {str(e)}", level="ERROR")
        return None

def time_execution(function_name):
    """Decorator to measure execution time of functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            log_execution(f"Starting {function_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record timing
                timing_record = {
                    "function": function_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_time": execution_time,
                    "session_id": SESSION_ID,
                    "status": "success"
                }
                st.session_state.timing_metrics.append(timing_record)
                
                log_execution(f"Completed {function_name} in {execution_time:.4f} seconds")
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record failed execution
                timing_record = {
                    "function": function_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_time": execution_time,
                    "session_id": SESSION_ID,
                    "status": "error",
                    "error": str(e)
                }
                st.session_state.timing_metrics.append(timing_record)
                
                log_execution(f"Error in {function_name}: {str(e)}", level="ERROR")
                raise
        
        return wrapper
    return decorator

# ========== INFORMATION CRITERIA FUNCTIONS (AIC/BIC) ==========

def calculate_model_complexity(model, X_train):
    """
    Determine model complexity (number of parameters) based on model type.
    This function handles different model architectures appropriately.
    
    Returns:
        int: Estimated number of parameters in the model
    """
    model_type = type(model).__name__
    n_features = X_train.shape[1]
    n_classes = len(np.unique(model.classes_)) if hasattr(model, 'classes_') else 0
    
    try:
        if model_type == "LogisticRegression":
            # For multiclass, parameters = n_features * (n_classes - 1) + (n_classes - 1) intercepts
            if model.multi_class == 'multinomial':
                return n_features * (n_classes - 1) + (n_classes - 1)
            # For OvR (one-vs-rest), parameters = n_features * n_classes + n_classes intercepts
            else:
                return n_features * n_classes + n_classes
                
        elif model_type == "SVC":
            # For SVM, complexity depends on number of support vectors
            if hasattr(model, 'support_vectors_'):
                return model.support_vectors_.shape[0]
            else:
                return n_features + 1
                
        elif model_type == "DecisionTreeClassifier":
            # For trees, complexity is related to number of nodes
            if hasattr(model, 'tree_'):
                return model.tree_.node_count
            else:
                return n_features
                
        elif model_type == "RandomForestClassifier":
            # For random forests, sum complexity across all trees
            if hasattr(model, 'estimators_'):
                return sum(estimator.tree_.node_count for estimator in model.estimators_)
            else:
                return n_features * model.n_estimators
                
        elif model_type == "KNeighborsClassifier":
            # KNN has no real parameters in the traditional sense
            # Return 1 to avoid divide-by-zero errors
            return 1
            
        elif model_type == "MLPClassifier":
            # For neural networks, count all weights and biases
            if hasattr(model, 'coefs_') and hasattr(model, 'intercepts_'):
                return sum(np.prod(coef.shape) for coef in model.coefs_) + \
                       sum(np.prod(intercept.shape) for intercept in model.intercepts_)
            else:
                # Estimate based on layer sizes
                return np.sum([n_features * model.hidden_layer_sizes[0]] + 
                             [model.hidden_layer_sizes[i] * model.hidden_layer_sizes[i+1] 
                              for i in range(len(model.hidden_layer_sizes)-1)] + 
                             [model.hidden_layer_sizes[-1] * n_classes])
        else:
            # Default fallback - estimate based on features
            return n_features + 1
            
    except Exception as e:
        log_execution(f"Error estimating complexity for {model_type}: {e}", level="WARNING")
        return n_features + 1  # Default fallback

def calculate_information_criteria(model, X, y):
    """
    Calculate AIC and BIC for the given model and data.
    Handles different model types appropriately.
    
    Returns:
        dict: Dictionary containing AIC, BIC values and metadata
    """
    n_samples = X.shape[0]
    model_type = type(model).__name__
    
    # Initialize results structure
    result = {
        "model_type": model_type,
        "n_samples": n_samples,
        "complexity": None,
        "log_likelihood": None,
        "aic": None,
        "bic": None,
        "approach": None,
        "is_applicable": False
    }
    
    # Calculate model complexity (number of parameters)
    complexity = calculate_model_complexity(model, X)
    result["complexity"] = complexity
    
    try:
        # Different calculation approaches based on model type
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            
            # Calculate log-likelihood for classification models
            log_likelihood = 0
            for i, true_class in enumerate(y):
                # Get probability for the true class, with small epsilon to avoid log(0)
                epsilon = 1e-15
                prob = max(min(y_pred_proba[i, true_class], 1 - epsilon), epsilon)
                log_likelihood += np.log(prob)
            
            result["log_likelihood"] = log_likelihood
            result["is_applicable"] = True
            result["approach"] = "predict_proba"
            
            # Calculate AIC and BIC
            result["aic"] = -2 * log_likelihood + 2 * complexity
            result["bic"] = -2 * log_likelihood + np.log(n_samples) * complexity
            
        elif model_type == "KNeighborsClassifier":
            # KNN doesn't have a probabilistic interpretation in scikit-learn
            # Use a proxy approach based on prediction accuracy
            y_pred = model.predict(X)
            accuracy = np.mean(y_pred == y)
            
            # Convert accuracy to a pseudo-likelihood 
            # (higher accuracy → higher likelihood, max likelihood = n_samples)
            pseudo_log_likelihood = n_samples * np.log(max(accuracy, 1e-15))
            
            result["log_likelihood"] = pseudo_log_likelihood
            result["is_applicable"] = True
            result["approach"] = "knn_proxy"
            
            # Calculate AIC and BIC using the proxy
            result["aic"] = -2 * pseudo_log_likelihood + 2 * complexity
            result["bic"] = -2 * pseudo_log_likelihood + np.log(n_samples) * complexity
            
        else:
            # For models without probabilistic output, mark as not applicable
            result["is_applicable"] = False
            result["approach"] = "not_applicable"
        
    except Exception as e:
        log_execution(f"Error calculating information criteria: {str(e)}", level="WARNING")
        result["is_applicable"] = False
        result["approach"] = f"error: {str(e)}"
    
    return result

# ========== ML APPLICATION CODE WITH OBSERVABILITY ==========

# Set page configuration
st.set_page_config(
    page_title="ATA AI Test Fest 2025 - Iris Classification ML Demo",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("ATA AI Test Fest 2025 - Iris Classification ML Demo")
st.markdown("""
This interactive demonstration is aimed at showing - in a highly simplified manner - what happens under the hood during the training and 
evaluation of machine learning models on the classic Iris dataset. 
Observe how different algorithms learn patterns in the data, 
and gain insights into the model training process.
""")

# Create tabs for main view and observability
main_tab, observability_tab = st.tabs(["ML Classification Demo", "Observability Dashboard"])

# Function to load and prepare the Iris dataset
@time_execution("load_and_prepare_iris_dataset")
def load_and_prepare_iris_dataset():
    """Load and prepare the Iris dataset"""
    try:
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Record initial data snapshot
        record_data_snapshot("raw_features", X, metadata={
            "dataset": "Iris",
            "feature_names": feature_names,
            "target_names": target_names
        })
        record_data_snapshot("raw_targets", y, metadata={
            "target_names": target_names.tolist()
        })
        
        # Create a DataFrame for better visualization
        iris_df = pd.DataFrame(X, columns=feature_names)
        iris_df['species'] = pd.Categorical.from_codes(y, target_names)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=50, stratify=y
        )
        
        # Record training and testing data snapshots
        record_data_snapshot("X_train", X_train, metadata={"shape": X_train.shape})
        record_data_snapshot("X_test", X_test, metadata={"shape": X_test.shape})
        record_data_snapshot("y_train", y_train, metadata={"shape": y_train.shape})
        record_data_snapshot("y_test", y_test, metadata={"shape": y_test.shape})
        
        # Return all the necessary data
        return X, y, X_train, X_test, y_train, y_test, feature_names, target_names, iris_df
    except Exception as e:
        log_execution(f"Error loading Iris dataset: {str(e)}", level="ERROR")
        # Return minimal data to prevent cascading errors
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), [], [], pd.DataFrame()

# Function to train and evaluate a model with incremental observability
@time_execution("train_and_evaluate_model")
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate a model with detailed step-by-step observability"""
    # Check for empty data
    if len(X_train) == 0 or len(y_train) == 0:
        log_execution(f"Cannot train model with empty data", level="ERROR")
        return None, 0.0, np.array([[0]]), {}, np.array([]), {}
    
    try:
        # Pre-training metrics
        class_distribution = {}
        if len(y_train) > 0:
            unique_labels, counts = np.unique(y_train, return_counts=True)
            for label, count in zip(unique_labels, counts):
                class_distribution[str(int(label))] = int(count)
        
        record_training_step(model_name, "pre_training", {
            "data_shape": X_train.shape,
            "class_distribution": class_distribution
        })
        
        # Record parameters before training
        if hasattr(model, "get_params"):
            track_parameters({f"{model_name}_{k}": v 
                            for k, v in model.get_params().items()})
        
        # Fit the model (training)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Post-training metrics
        record_training_step(model_name, "post_training", {
            "training_time": training_time
        })
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, 
                                            target_names=[str(name) for name in load_iris().target_names], 
                                            output_dict=True)
        
        # Calculate information criteria (AIC/BIC)
        info_criteria = calculate_information_criteria(model, X_test, y_test)
        
        # Record evaluation metrics
        record_training_step(model_name, "evaluation", {
            "accuracy": accuracy,
            "class_report": class_report,
            "information_criteria": info_criteria
        })
        
        # Trace model predictions
        trace_model_predictions(model, X_test, y_test, prediction_context={
            "model_name": model_name,
            "accuracy": accuracy
        })
        
        # Record feature importance if available
        if hasattr(model, "feature_importances_"):
            record_feature_importance(model_name, feature_names, model.feature_importances_)
        # For logistic regression, record coefficients as feature importance
        elif model_name == "Logistic Regression" and hasattr(model, "coef_"):
            # Use absolute values of coefficients as importance
            importance_values = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            record_feature_importance(model_name, feature_names, importance_values, step="coefficient_magnitude")
        
        # Return the trained model and evaluation results
        return model, accuracy, conf_matrix, class_report, y_pred, info_criteria
    except Exception as e:
        log_execution(f"Error in model training/evaluation: {str(e)}", level="ERROR")
        # Return minimal data to prevent cascading errors
        return model, 0.0, np.array([[0]]), {}, np.array([]), {}

with main_tab:
    # Load and prepare the dataset when the app starts
    if "iris_data" not in st.session_state:
        X, y, X_train, X_test, y_train, y_test, feature_names, target_names, iris_df = load_and_prepare_iris_dataset()
        
        # Ensure data is valid before proceeding
        if len(X) > 0 and len(y) > 0:
            st.session_state.iris_data = {
                "X": X, "y": y, 
                "X_train": X_train, "X_test": X_test, 
                "y_train": y_train, "y_test": y_test,
                "feature_names": feature_names, "target_names": target_names,
                "iris_df": iris_df
            }
        else:
            st.error("Failed to load Iris dataset. Check logs for details.")
            st.stop()
    else:
        # Retrieve from session state
        X = st.session_state.iris_data["X"]
        y = st.session_state.iris_data["y"]
        X_train = st.session_state.iris_data["X_train"]
        X_test = st.session_state.iris_data["X_test"]
        y_train = st.session_state.iris_data["y_train"]
        y_test = st.session_state.iris_data["y_test"]
        feature_names = st.session_state.iris_data["feature_names"]
        target_names = st.session_state.iris_data["target_names"]
        iris_df = st.session_state.iris_data["iris_df"]
    
    # Show dataset overview
    st.header("Dataset Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            # Create a scatter plot for visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Use PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Plot the points
            for i, target_name in enumerate(target_names):
                ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                         label=target_name, alpha=0.7)
            
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('Iris Dataset PCA Visualization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display PCA explained variance
            st.markdown(f"""
            **PCA Explained Variance**: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}
            (Total: {sum(pca.explained_variance_ratio_):.2f})
            """)
            
            # Show PCA component loadings
            st.subheader("PCA Component Loadings")
            component_df = pd.DataFrame(pca.components_, 
                                       columns=feature_names,
                                       index=['PC1', 'PC2'])
            st.dataframe(component_df)
        except Exception as e:
            log_execution(f"Error creating PCA visualization: {str(e)}", level="ERROR")
            st.error("Could not create PCA visualization. See logs for details.")
    
    with col2:
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"**Number of samples:** {len(X)}")
        st.write(f"**Number of features:** {len(feature_names)}")
        st.write(f"**Number of classes:** {len(target_names)}")
        
        # Display class distribution
        st.subheader("Class Distribution")
        class_counts = pd.Series(y).value_counts().sort_index()
        class_names = [target_names[i] for i in class_counts.index]
        distribution_df = pd.DataFrame({"Species": class_names, "Count": class_counts.values})
        st.dataframe(distribution_df)
        
        # Display feature summary
        st.subheader("Feature Summary")
        summary_df = pd.DataFrame({
            "Feature": feature_names,
            "Min": [X[:, i].min() for i in range(X.shape[1])],
            "Max": [X[:, i].max() for i in range(X.shape[1])],
            "Mean": [X[:, i].mean() for i in range(X.shape[1])],
            "Std": [X[:, i].std() for i in range(X.shape[1])]
        })
        st.dataframe(summary_df)
    
    # Model training and evaluation section
    st.header("Model Training and Evaluation")
    
    # Model selection
    model_col1, model_col2 = st.columns([1, 1])
    
    with model_col1:
        # Allow user to select algorithm (with Logistic Regression added)
        selected_model = st.selectbox(
            "Select a classification algorithm",
            ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", 
             "Random Forest", "Neural Network"]
        )
        
        # Map selection to actual model (with Logistic Regression added)
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000)
        }
        
        model = model_map[selected_model]
        
        # Train button
        if st.button("Train Model"):
            try:
                with st.spinner(f"Training {selected_model}..."):
                    # Train and evaluate the model (with updated function signature)
                    trained_model, accuracy, conf_matrix, class_report, y_pred, info_criteria = train_and_evaluate_model(
                        model, selected_model, X_train, X_test, y_train, y_test, feature_names
                    )
                    
                    # Store results in session state for display (with info_criteria added)
                    st.session_state.current_model = {
                        "name": selected_model,
                        "model": trained_model,
                        "accuracy": accuracy,
                        "conf_matrix": conf_matrix,
                        "class_report": class_report,
                        "y_pred": y_pred,
                        "info_criteria": info_criteria
                    }
                    
                    # Update model comparison data
                    if info_criteria["is_applicable"]:
                        st.session_state.model_comparison[selected_model] = {
                            "accuracy": accuracy,
                            "aic": info_criteria["aic"],
                            "bic": info_criteria["bic"],
                            "complexity": info_criteria["complexity"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
            except Exception as e:
                log_execution(f"Error training model: {str(e)}", level="ERROR")
                st.error(f"Error training model: {str(e)}")
    
    # Display model results if available
    if "current_model" in st.session_state:
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {st.session_state.current_model['accuracy']:.4f}")
            
            # Display confusion matrix
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                st.session_state.current_model['conf_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax
            )
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
        
        with result_col2:
            # Display classification report
            st.subheader("Classification Report")
            report = st.session_state.current_model['class_report']
            
            # Create a dataframe from the classification report
            report_df = pd.DataFrame({
                "Precision": [report[str(name)]['precision'] for name in target_names],
                "Recall": [report[str(name)]['recall'] for name in target_names],
                "F1-Score": [report[str(name)]['f1-score'] for name in target_names],
                "Support": [report[str(name)]['support'] for name in target_names]
            }, index=target_names)
            
            st.dataframe(report_df)
            
            # Display information criteria if available
            if "info_criteria" in st.session_state.current_model:
                st.subheader("Information Criteria")
                info_criteria = st.session_state.current_model["info_criteria"]
                
                if info_criteria["is_applicable"]:
                    # Display AIC and BIC metrics side by side
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        if info_criteria["aic"] is not None:
                            st.metric(
                                "AIC", 
                                f"{info_criteria['aic']:.2f}", 
                                help="Akaike Information Criterion: Lower values indicate better models considering fit and complexity"
                            )
                        else:
                            st.metric("AIC", "N/A")
                    
                    with metric_col2:
                        if info_criteria["bic"] is not None:
                            st.metric(
                                "BIC", 
                                f"{info_criteria['bic']:.2f}", 
                                help="Bayesian Information Criterion: Lower values indicate better models with stronger complexity penalty"
                            )
                        else:
                            st.metric("BIC", "N/A")
                    
                    # Display model complexity
                    st.metric(
                        "Model Complexity", 
                        f"{info_criteria['complexity']}", 
                        help="Number of effective parameters in the model"
                    )
                    
                    # Add interpretative information
                    st.info(f"Information criteria calculated using: {info_criteria.get('approach', 'unknown')} approach. Lower AIC/BIC values generally indicate better models when comparing between algorithms.")
                else:
                    st.warning("Information criteria not applicable to this model type or could not be calculated.")
                    st.write(f"Model complexity: {info_criteria.get('complexity', 'Unknown')}")
            
            # Display feature importance if available
            if hasattr(st.session_state.current_model['model'], 'feature_importances_'):
                st.subheader("Feature Importance")
                importances = st.session_state.current_model['model'].feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(range(len(importances)), importances[indices])
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
            
            # For Logistic Regression, display coefficients
            elif st.session_state.current_model['name'] == "Logistic Regression" and hasattr(st.session_state.current_model['model'], 'coef_'):
                st.subheader("Feature Coefficients")
                
                coef = st.session_state.current_model['model'].coef_
                
                if coef.ndim > 1:
                    # Multiclass case - show coefficients for each class
                    coef_df = pd.DataFrame(
                        coef,
                        columns=feature_names,
                        index=[target_names[i] for i in range(coef.shape[0])]
                    )
                    st.dataframe(coef_df)
                    
                    # Plot coefficient magnitudes
                    st.subheader("Coefficient Magnitudes")
                    coef_mag = np.abs(coef).mean(axis=0)
                    indices = np.argsort(coef_mag)[::-1]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(len(coef_mag)), coef_mag[indices])
                    ax.set_xticks(range(len(coef_mag)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Coefficient Magnitude')
                    ax.set_title('Feature Coefficient Magnitudes')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # Binary case
                    coef_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Coefficient": coef[0]
                    })
                    st.dataframe(coef_df.sort_values("Coefficient", ascending=False))
                    
                    # Plot coefficients
                    indices = np.argsort(np.abs(coef[0]))[::-1]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(len(coef[0])), coef[0][indices])
                    ax.set_xticks(range(len(coef[0])))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Coefficient Value')
                    ax.set_title('Logistic Regression Coefficients')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Model Comparison section (if multiple models have been trained)
    if len(st.session_state.model_comparison) > 1:
        st.header("Model Comparison")
        
        # Convert comparison data to DataFrame
        comparison_data = []
        for model_name, metrics in st.session_state.model_comparison.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "AIC": metrics["aic"] if "aic" in metrics and metrics["aic"] is not None else float('nan'),
                "BIC": metrics["bic"] if "bic" in metrics and metrics["bic"] is not None else float('nan'),
                "Complexity": metrics["complexity"] if "complexity" in metrics else "N/A",
                "Timestamp": metrics["timestamp"] if "timestamp" in metrics else "N/A"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight the best model based on different criteria
        if len(comparison_df) > 1:
            # Find best models
            try:
                best_acc_idx = comparison_df["Accuracy"].astype(float).idxmax()
                best_aic_idx = comparison_df["AIC"].astype(float).idxmin() if not comparison_df["AIC"].isna().all() else None
                best_bic_idx = comparison_df["BIC"].astype(float).idxmin() if not comparison_df["BIC"].isna().all() else None
                
                # Add indicator columns
                comparison_df["Best Accuracy"] = ""
                comparison_df.loc[best_acc_idx, "Best Accuracy"] = "✓"
                
                if best_aic_idx is not None:
                    comparison_df["Best AIC"] = ""
                    comparison_df.loc[best_aic_idx, "Best AIC"] = "✓"
                
                if best_bic_idx is not None:
                    comparison_df["Best BIC"] = ""
                    comparison_df.loc[best_bic_idx, "Best BIC"] = "✓"
            except Exception as e:
                log_execution(f"Error highlighting best models: {str(e)}", level="WARNING")
        
        # Display the comparison table
        st.dataframe(comparison_df)
        
        # Provide interpretation guidance
        st.markdown("""
        **Interpreting the comparison metrics:**
        - **Accuracy**: Higher is better, represents prediction correctness
        - **AIC/BIC**: Lower is better, balances model fit with complexity
        - **Complexity**: Number of effective parameters in the model
        
        The best model often balances these metrics rather than excelling at just one.
        AIC and BIC penalize complex models to prevent overfitting, with BIC generally imposing a stronger penalty.
        """)
        
        # Plot comparison visualization
        st.subheader("Model Metrics Comparison")
        
        if not comparison_df["AIC"].isna().all():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Accuracy plot (higher is better)
            axes[0].bar(comparison_df["Model"], comparison_df["Accuracy"])
            axes[0].set_title("Accuracy (higher is better)")
            axes[0].set_ylim(0, 1)
            plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
            
            # AIC plot (lower is better)
            valid_aic = comparison_df[~comparison_df["AIC"].isna()]
            if not valid_aic.empty:
                axes[1].bar(valid_aic["Model"], valid_aic["AIC"])
                axes[1].set_title("AIC (lower is better)")
                plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
            else:
                axes[1].set_title("AIC not available")
            
            # BIC plot (lower is better)
            valid_bic = comparison_df[~comparison_df["BIC"].isna()]
            if not valid_bic.empty:
                axes[2].bar(valid_bic["Model"], valid_bic["BIC"])
                axes[2].set_title("BIC (lower is better)")
                plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
            else:
                axes[2].set_title("BIC not available")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Option to reset comparison
        if st.button("Reset Comparison"):
            st.session_state.model_comparison = {}
            st.experimental_rerun()
    
    # User prediction section
    st.header("Make Your Own Prediction")
    
    # Create sliders for each feature
    st.subheader("Set Feature Values")
    user_input = {}
    
    # Create two columns for feature inputs
    pred_col1, pred_col2 = st.columns(2)
    
    # Find min and max values for each feature for slider ranges
    feature_mins = [float(X[:, i].min()) for i in range(X.shape[1])]
    feature_maxs = [float(X[:, i].max()) for i in range(X.shape[1])]
    
    # Distribute feature sliders across columns
    for i, feature in enumerate(feature_names):
        col = pred_col1 if i < len(feature_names) // 2 else pred_col2
        with col:
            user_input[feature] = st.slider(
                f"{feature}",
                min_value=feature_mins[i],
                max_value=feature_maxs[i],
                value=(feature_mins[i] + feature_maxs[i]) / 2,  # Default: middle value
                step=0.1
            )
    
    # Predict button
    if st.button("Predict"):
        try:
            # Create input array from user inputs
            input_array = np.array([[user_input[feature] for feature in feature_names]])
            
            # Check if we have a trained model
            if "current_model" in st.session_state and st.session_state.current_model["model"] is not None:
                model = st.session_state.current_model["model"]
                model_name = st.session_state.current_model["name"]
                
                # Make prediction
                prediction = model.predict(input_array)[0]
                predicted_species = target_names[prediction]
                
                # Get probabilities if available
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_array)[0]
                
                # Display prediction
                st.subheader("Prediction Result")
                st.write(f"**Predicted species:** {predicted_species}")
                
                # Display probabilities if available
                if probabilities is not None:
                    st.write("**Probabilities:**")
                    prob_df = pd.DataFrame({
                        "Species": target_names,
                        "Probability": probabilities
                    })
                    
                    # Plot probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(target_names, probabilities)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    st.pyplot(fig)
                
                # Show the user's sample in PCA space
                try:
                    st.subheader("Your Sample in the Iris Dataset")
                    
                    # Create PCA transformation again
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    
                    # Transform the user's input
                    user_pca = pca.transform(input_array)
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot the original data
                    for i, target_name in enumerate(target_names):
                        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                                 label=target_name, alpha=0.3)
                    
                    # Plot the user's input
                    ax.scatter(user_pca[0, 0], user_pca[0, 1], 
                             marker='*', s=200, c='red', label='Your Input')
                    
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    ax.set_title('Your Sample in the Iris Dataset (PCA)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                except Exception as e:
                    log_execution(f"Error creating PCA plot for user sample: {str(e)}", level="ERROR")
                    st.error("Could not create PCA visualization for your sample")
            else:
                st.warning("Please train a model first")
        except Exception as e:
            log_execution(f"Error making prediction: {str(e)}", level="ERROR")
            st.error("Could not make prediction. Please check logs for details.")

# Observability Dashboard
with observability_tab:
    st.header("ML Observability Dashboard")
    st.write("Explore the inner workings of the ML training and inference process.")
    
    # Create tabs for different observability views
    obs_tabs = st.tabs([
        "Execution Logs", 
        "Data Snapshots", 
        "Model Training", 
        "Performance Metrics",
        "Feature Importance",
        "Information Criteria"  # New tab for information criteria observability
    ])
    
    # Execution Logs tab
    with obs_tabs[0]:
        st.subheader("Execution Logs")
        
        # Filter by log level
        log_level = st.selectbox(
            "Filter by log level",
            ["All", "INFO", "WARNING", "ERROR", "DEBUG"]
        )
        
        # Get logs
        logs = st.session_state.execution_logs
        
        # Filter logs by level if needed
        if log_level != "All":
            logs = [log for log in logs if log["level"] == log_level]
        
        # Display logs in a table
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df[["timestamp", "level", "message"]], use_container_width=True)
        else:
            st.info("No logs available")
    
    # Data Snapshots tab
    with obs_tabs[1]:
        st.subheader("Data Snapshots")
        
        # Get snapshots
        snapshots = st.session_state.data_snapshots
        
        if snapshots:
            # Allow user to select a snapshot
            snapshot_names = [f"{s['step']} ({s['timestamp']})" for s in snapshots]
            selected_snapshot = st.selectbox("Select a data snapshot", snapshot_names)
            
            # Get the selected snapshot
            selected_idx = snapshot_names.index(selected_snapshot)
            snapshot = snapshots[selected_idx]
            
            # Display snapshot information
            st.write(f"**Step:** {snapshot['step']}")
            st.write(f"**Timestamp:** {snapshot['timestamp']}")
            st.write(f"**Data Type:** {snapshot['data_type']}")
            
            # Display statistics or distribution based on data type
            if snapshot['statistics']['mean'] is not None:
                # Numeric data - show statistics
                st.subheader("Statistics")
                stats_df = pd.DataFrame({
                    "Statistic": ["Count", "Mean", "Median", "Min", "Max", "Std Dev"],
                    "Value": [
                        snapshot['statistics']['count'],
                        snapshot['statistics']['mean'],
                        snapshot['statistics']['median'],
                        snapshot['statistics']['min'],
                        snapshot['statistics']['max'],
                        snapshot['statistics']['std_dev']
                    ]
                })
                st.dataframe(stats_df)
            else:
                # Categorical data - show distribution
                st.subheader("Categorical Distribution")
                if snapshot['categorical_distribution']:
                    distribution = snapshot['categorical_distribution']
                    
                    # Convert to DataFrame for display
                    dist_df = pd.DataFrame({
                        "Category": list(distribution.keys()),
                        "Count": list(distribution.values())
                    })
                    
                    # Display as table and chart
                    st.dataframe(dist_df)
                    
                    fig, ax = plt.subplots()
                    ax.bar(dist_df["Category"], dist_df["Count"])
                    ax.set_xlabel("Category")
                    ax.set_ylabel("Count")
                    ax.set_title("Category Distribution")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Display data sample
            st.subheader("Data Sample")
            st.write(f"First {len(snapshot['data_sample'])} records:")
            st.write(snapshot['data_sample'])
            
            # Display metadata if any
            if snapshot['metadata']:
                st.subheader("Metadata")
                for key, value in snapshot['metadata'].items():
                    st.write(f"**{key}:** {value}")
        else:
            st.info("No data snapshots available")
    
    # Model Training tab
    with obs_tabs[2]:
        st.subheader("Model Training History")
        
        # Get training records
        training_records = st.session_state.training_history
        
        if training_records:
            # Allow user to filter by model
            model_names = list(set([record["model_name"] for record in training_records]))
            selected_model = st.selectbox("Select a model", ["All"] + model_names)
            
            # Filter records by model if needed
            if selected_model != "All":
                filtered_records = [record for record in training_records if record["model_name"] == selected_model]
            else:
                filtered_records = training_records
            
            # Create a simplified view of training records
            records_df = pd.DataFrame([
                {
                    "Timestamp": record["timestamp"],
                    "Model": record["model_name"],
                    "Stage": record["training_stage"],
                    "Iteration": record["iteration"] if record["iteration"] else "-"
                }
                for record in filtered_records
            ])
            
            # Display records
            st.dataframe(records_df, use_container_width=True)
            
            # Allow user to select a record for detailed view
            record_names = [f"{r['model_name']} - {r['training_stage']} ({r['timestamp']})" for r in filtered_records]
            if record_names:
                selected_record = st.selectbox("Select a record for details", record_names)
                
                # Get the selected record
                selected_idx = record_names.index(selected_record)
                record = filtered_records[selected_idx]
                
                # Display record details
                st.subheader("Record Details")
                st.write(f"**Model:** {record['model_name']}")
                st.write(f"**Stage:** {record['training_stage']}")
                st.write(f"**Timestamp:** {record['timestamp']}")
                
                # Display metrics
                st.subheader("Metrics")
                for key, value in record["metrics"].items():
                    # Special handling for information criteria
                    if key == "information_criteria" and isinstance(value, dict):
                        st.write("**Information Criteria:**")
                        st.json({k: v for k, v in value.items() if k not in ["approach", "log_likelihood"]})
                    # Check if it's a simple value or a complex object
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        st.write(f"**{key}:** {value}")
                    elif isinstance(value, dict):
                        # For dictionaries like class reports
                        st.write(f"**{key}:**")
                        st.json(value)
                    elif isinstance(value, list):
                        # For lists
                        st.write(f"**{key}:** {value}")
        else:
            st.info("No training records available")
    
    # Performance Metrics tab
    with obs_tabs[3]:
        st.subheader("Performance Metrics")
        
        # Get timing metrics
        timing_metrics = st.session_state.timing_metrics
        
        if timing_metrics:
            # Create DataFrame for visualization
            timing_df = pd.DataFrame([
                {
                    "Function": record["function"],
                    "Timestamp": record["timestamp"],
                    "Execution Time (s)": record["execution_time"],
                    "Status": record["status"]
                }
                for record in timing_metrics
            ])
            
            # Display summary statistics
            st.subheader("Function Performance Summary")
            
            # Group by function and calculate stats
            summary = timing_df.groupby("Function").agg({
                "Execution Time (s)": ["count", "mean", "min", "max"]
            }).reset_index()
            
            # Flatten multi-level columns
            summary.columns = ["Function", "Count", "Mean Time (s)", "Min Time (s)", "Max Time (s)"]
            
            # Display summary
            st.dataframe(summary, use_container_width=True)
            
            # Plot execution times
            st.subheader("Execution Time by Function")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by function for the plot
            for function in timing_df["Function"].unique():
                function_df = timing_df[timing_df["Function"] == function]
                ax.plot(range(len(function_df)), function_df["Execution Time (s)"], marker='o', label=function)
            
            ax.set_xlabel("Execution Order")
            ax.set_ylabel("Execution Time (s)")
            ax.set_title("Function Execution Times")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display all timing records
            st.subheader("All Timing Records")
            st.dataframe(timing_df, use_container_width=True)
        else:
            st.info("No performance metrics available")
    
    # Feature Importance tab
    with obs_tabs[4]:
        st.subheader("Feature Importance Analysis")
        
        # Get feature importance records
        importance_records = st.session_state.feature_importance
        
        if importance_records:
            # Allow user to select a model
            model_names = list(set([record["model_name"] for record in importance_records]))
            selected_model = st.selectbox("Select a model for feature importance", model_names)
            
            # Filter records by model
            filtered_records = [record for record in importance_records if record["model_name"] == selected_model]
            
            # If we have records for this model
            if filtered_records:
                # Get the most recent record
                record = filtered_records[-1]
                
                # Extract feature importance data
                features = [item["feature"] for item in record["importance"]]
                importance = [item["importance"] for item in record["importance"]]
                
                # Display as a table
                st.subheader("Feature Importance Ranking")
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                })
                st.dataframe(importance_df, use_container_width=True)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(features, importance)
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature Importance for {selected_model}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Special note for Logistic Regression
                if selected_model == "Logistic Regression":
                    st.info("For Logistic Regression, feature importance is derived from the absolute values of the model coefficients, representing the strength of feature influence on the classification outcome.")
            else:
                st.info(f"No feature importance data available for {selected_model}")
        else:
            st.info("No feature importance data available")
    
    # Information Criteria tab (new)
    with obs_tabs[5]:
        st.subheader("Information Criteria Analysis")
        
        if st.session_state.model_comparison:
            # Display model comparison table with information criteria
            st.subheader("Model Comparison by Information Criteria")
            
            # Convert comparison data to DataFrame
            comparison_data = []
            for model_name, metrics in st.session_state.model_comparison.items():
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "AIC": metrics.get("aic", float('nan')),
                    "BIC": metrics.get("bic", float('nan')),
                    "Complexity": metrics.get("complexity", "N/A"),
                    "Timestamp": metrics.get("timestamp", "N/A")
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Highlight the best model based on different criteria
            if len(comparison_df) > 1:
                # Find best models
                try:
                    best_acc_idx = comparison_df["Accuracy"].astype(float).idxmax()
                    best_aic_idx = comparison_df["AIC"].astype(float).idxmin() if not comparison_df["AIC"].isna().all() else None
                    best_bic_idx = comparison_df["BIC"].astype(float).idxmin() if not comparison_df["BIC"].isna().all() else None
                    
                    # Add indicator columns
                    comparison_df["Best Accuracy"] = ""
                    comparison_df.loc[best_acc_idx, "Best Accuracy"] = "✓"
                    
                    if best_aic_idx is not None:
                        comparison_df["Best AIC"] = ""
                        comparison_df.loc[best_aic_idx, "Best AIC"] = "✓"
                    
                    if best_bic_idx is not None:
                        comparison_df["Best BIC"] = ""
                        comparison_df.loc[best_bic_idx, "Best BIC"] = "✓"
                except Exception as e:
                    log_execution(f"Error highlighting best models: {str(e)}", level="WARNING")
            
            # Display the comparison table
            st.dataframe(comparison_df)
            
            # Visualize information criteria comparison
            if len(comparison_df) > 1 and not comparison_df["AIC"].isna().all():
                st.subheader("Information Criteria Visualization")
                
                # Create a plot with two subplots for AIC and BIC
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Filter out models without AIC/BIC values
                valid_models = comparison_df[~comparison_df["AIC"].isna()]
                
                if not valid_models.empty:
                    # Sort by AIC for better visualization
                    valid_models = valid_models.sort_values("AIC")
                    
                    # AIC plot
                    axes[0].bar(valid_models["Model"], valid_models["AIC"])
                    axes[0].set_title("AIC by Model (lower is better)")
                    axes[0].set_ylabel("AIC Value")
                    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
                    
                    # BIC plot
                    axes[1].bar(valid_models["Model"], valid_models["BIC"])
                    axes[1].set_title("BIC by Model (lower is better)")
                    axes[1].set_ylabel("BIC Value")
                    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Create a scatterplot of complexity vs. AIC/BIC
                    st.subheader("Model Complexity Analysis")
                    
                    # Convert complexity to numeric if possible
                    try:
                        valid_models["Complexity"] = pd.to_numeric(valid_models["Complexity"])
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(
                            valid_models["Complexity"], 
                            valid_models["AIC"],
                            s=100,
                            alpha=0.7,
                            c=valid_models["BIC"],
                            cmap="viridis"
                        )
                        
                        # Add model labels to points
                        for i, model in enumerate(valid_models["Model"]):
                            ax.annotate(model, 
                                      (valid_models["Complexity"].iloc[i], valid_models["AIC"].iloc[i]),
                                      xytext=(5, 5), textcoords="offset points")
                        
                        ax.set_xlabel("Model Complexity (number of parameters)")
                        ax.set_ylabel("AIC Value (lower is better)")
                        ax.set_title("Model Complexity vs. Information Criteria")
                        ax.grid(True, alpha=0.3)
                        
                        # Add colorbar for BIC
                        cbar = plt.colorbar(scatter)
                        cbar.set_label("BIC Value (lower is better)")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    except:
                        st.info("Could not create complexity analysis due to non-numeric complexity values.")
            
            # Explanation of information criteria
            st.subheader("Understanding Information Criteria")
            st.markdown("""
            ### Akaike Information Criterion (AIC)
            AIC = -2 * log-likelihood + 2 * complexity
            
            AIC estimates the relative quality of statistical models for a given dataset. It rewards goodness of fit (via the likelihood term) while penalizing model complexity to prevent overfitting. Lower AIC values indicate better models.
            
            ### Bayesian Information Criterion (BIC)
            BIC = -2 * log-likelihood + complexity * log(n)
            
            BIC is similar to AIC but applies a stronger penalty for complexity, especially with larger sample sizes (n). BIC tends to prefer simpler models compared to AIC. Lower BIC values indicate better models.
            
            ### Interpretation Guidelines
            - **Comparing AIC/BIC Values**: When comparing models, differences of 2-7 points are considered positive evidence, while differences >10 points represent strong evidence.
            - **Complexity and Overfitting**: Models with many parameters may achieve high accuracy on training data but generalize poorly to new data.
            - **Different Modeling Goals**: Choose AIC when prediction accuracy is paramount; prefer BIC when explanatory power and parsimony are priorities.
            """)
        else:
            st.info("Train multiple models to see information criteria comparison.")

    # Session information
    st.subheader("Session Information")
    st.write(f"**Session ID:** {SESSION_ID}")
    st.write(f"**Start Time:** {st.session_state.execution_logs[0]['timestamp'] if st.session_state.execution_logs else 'N/A'}")
    
    # Export data button
    if st.button("Export Observability Data"):
        try:
            # Create a dictionary with all observability data
            export_data = {
                "session_id": SESSION_ID,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "execution_logs": st.session_state.execution_logs,
                "parameter_history": st.session_state.parameter_history,
                "data_snapshots": st.session_state.data_snapshots,
                "model_traces": st.session_state.model_traces,
                "timing_metrics": st.session_state.timing_metrics,
                "training_history": st.session_state.training_history,
                "feature_importance": st.session_state.feature_importance,
                "model_comparison": st.session_state.model_comparison  # Added model comparison export
            }
            
            # Convert to JSON
            export_json = json.dumps(export_data, indent=2)
            
            # Offer for download
            st.download_button(
                label="Download JSON",
                data=export_json,
                file_name=f"ml_observability_export_{SESSION_ID}.json",
                mime="application/json"
            )
        except Exception as e:
            log_execution(f"Error exporting observability data: {str(e)}", level="ERROR")
            st.error(f"Error exporting data: {str(e)}")

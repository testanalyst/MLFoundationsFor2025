"""
ML Testing Dashboard - Attempt by a Tester for Testers
-------------------
A comprehensive yet evolving (and certainly buggy as well) testing suite for machine learning models
focused on KNN classification with the Iris dataset. 
I extensively used LLMs (Anthropic Sonnet 3.7), Books, Perplexity to reach here.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import warnings
from PIL import Image
import io
import base64

# Load necessary components from your programs
# Using try/except to handle potential import issues
try:
    from program1 import IrisClassifier
    PROGRAM1_AVAILABLE = True
except ImportError:
    PROGRAM1_AVAILABLE = False
    st.warning("Program1 module not available. Some functionality will be limited.")

try:
    from program2 import ModelTester
    PROGRAM2_AVAILABLE = True
except ImportError:
    PROGRAM2_AVAILABLE = False
    st.warning("Program2 module not available. Some functionality will be limited.")

try:
    from program3 import RobustnessTester
    PROGRAM3_AVAILABLE = True
except ImportError:
    PROGRAM3_AVAILABLE = False
    st.warning("Program3 module not available. Some functionality will be limited.")

try:
    from program4 import AdvancedTester
    PROGRAM4_AVAILABLE = True
except ImportError:
    PROGRAM4_AVAILABLE = False
    st.warning("Program4 module not available. Some functionality will be limited.")

# Utility functions for handling images and visualization
def get_image_as_base64(image_path):
    """Convert an image to base64 for inline display."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None

def display_image_if_exists(image_path, caption=None, width=None):
    """Display an image if it exists, otherwise show a message."""
    if os.path.exists(image_path):
        img = Image.open(image_path)
        if width:
            st.image(img, caption=caption, width=width)
        else:
            st.image(img, caption=caption)
    else:
        st.info(f"Image not found: {image_path}")

def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="ML Testing Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for layout and styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4257b2;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5c7cfa;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f1f3f9;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4257b2;
    }
    .metric-label {
        font-size: 1rem;
        color: #718096;
    }
    .alert-box {
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-box.success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .alert-box.warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .alert-box.error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">ML Testing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Comprehensive Testing Suite for KNN Classification</h2>', 
                unsafe_allow_html=True)
    
    # Display a brief description of the application
    st.markdown("""
    This dashboard provides a comprehensive testing framework for machine learning models,
    focusing on KNN classification with the Iris dataset. It includes:
    
    - **Basic Testing**: Model training, evaluation, and visualization
    - **Advanced Testing**: Cross-validation and model explainability
    - **Robustness Testing**: Performance with outliers and edge cases
    - **Hyperparameter Optimization**: Finding optimal model parameters
    - **Adversarial Testing**: Testing model vulnerability to adversarial examples
    - **Data Drift Simulation**: Assessing model performance under different data distributions
    """)

def run_basic_testing():
    """Run basic testing (Program 1 functionality)."""
    st.markdown('## Basic Model Testing')
    
    if not PROGRAM1_AVAILABLE:
        st.error("Program1 module is not available. Cannot perform basic testing.")
        return
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 3)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    
    with col2:
        st.subheader("Execution Options")
        load_model = st.checkbox("Load pre-trained model if available", True)
        explore_data = st.checkbox("Perform exploratory data analysis", True)
    
    if st.button("Run Basic Testing"):
        with st.spinner("Running basic model testing..."):
            try:
                # Initialize classifier
                iris_clf = IrisClassifier()
                
                # Execute the testing pipeline based on user selections
                if load_model and os.path.exists("iris_classifier_model.pkl"):
                    iris_clf.load_model()
                    st.success("Pre-trained model loaded successfully.")
                else:
                    progress_bar = st.progress(0)
                    
                    # Data loading
                    iris_clf.load_data()
                    progress_bar.progress(20)
                    
                    # Exploratory data analysis
                    if explore_data:
                        iris_clf.explore_data()
                    progress_bar.progress(40)
                    
                    # Data preparation
                    iris_clf.prepare_data(test_size=test_size)
                    progress_bar.progress(60)
                    
                    # Model training
                    iris_clf.train_model(n_neighbors=n_neighbors)
                    progress_bar.progress(80)
                    
                    # Model evaluation
                    iris_clf.evaluate_model()
                    progress_bar.progress(90)
                    
                    # Save model
                    iris_clf.save_model()
                    progress_bar.progress(100)
                    
                    st.success("Model training and evaluation completed successfully!")
                
                # Display results in an expander
                with st.expander("View Model Performance Metrics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{getattr(iris_clf, 'accuracy', 0):.4f}")
                    with col2:
                        st.metric("Recall (Macro)", f"{getattr(iris_clf, 'recall', 0):.4f}")
                    with col3:
                        st.metric("F1 Score (Macro)", f"{getattr(iris_clf, 'f1', 0):.4f}")
                    
                    # Display classification report if available
                    if hasattr(iris_clf, 'class_report'):
                        st.text("Classification Report:")
                        st.text(iris_clf.class_report)
                
                # Display visualizations
                st.subheader("Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    display_image_if_exists("program1_confusion_matrix.png", 
                                          "Confusion Matrix")
                
                with viz_col2:
                    display_image_if_exists("program1_decision_boundaries.png", 
                                          "Decision Boundaries")
                
                if explore_data:
                    eda_col1, eda_col2 = st.columns(2)
                    
                    with eda_col1:
                        display_image_if_exists("program1_pairplot.png", 
                                              "Feature Pairplot")
                    
                    with eda_col2:
                        display_image_if_exists("program1_boxplots.png", 
                                              "Feature Distributions")
            
            except Exception as e:
                st.error(f"Error during basic testing: {str(e)}")
                st.exception(e)

def run_advanced_testing():
    """Run advanced testing (Program 2 functionality)."""
    st.markdown('## Advanced Model Testing')
    
    if not PROGRAM2_AVAILABLE or not PROGRAM1_AVAILABLE:
        st.error("Program1 or Program2 module is not available. Cannot perform advanced testing.")
        return
    
    st.subheader("Cross-Validation and Explainability Testing")
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        load_model = st.checkbox("Load pre-trained model", True)
    
    with col2:
        load_results = st.checkbox("Load previous test results if available", False)
        n_iterations = st.slider("Performance Test Iterations", 10, 200, 100)
    
    perform_shap = st.checkbox("Attempt SHAP analysis (may have visualization errors)", False)
    
    if perform_shap:
        st.warning("Note: SHAP visualization has known errors in this implementation. " +
                  "The app will try to handle gracefully, but some visualizations may not appear.")
    
    if st.button("Run Advanced Testing"):
        with st.spinner("Running advanced model testing..."):
            try:
                # Step 1: Make sure we have a trained model
                iris_clf = IrisClassifier()
                
                if load_model:
                    if os.path.exists("iris_classifier_model.pkl"):
                        iris_clf.load_model()
                        st.info("Pre-trained model loaded successfully.")
                    else:
                        st.warning("No pre-trained model found. Training a new model...")
                        iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                else:
                    iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                
                # Step 2: Initialize and run model tester
                tester = ModelTester(iris_clf)
                
                if load_results and os.path.exists("model_tester_results.pkl"):
                    tester.load_results()
                    st.info("Previous test results loaded successfully.")
                
                progress_bar = st.progress(0)
                
                # Cross-validation
                if not hasattr(tester, 'cv_results') or tester.cv_results is None:
                    tester.perform_cross_validation(cv=cv_folds)
                progress_bar.progress(33)
                
                # SHAP analysis (with error handling)
                if perform_shap and not hasattr(tester, 'shap_values'):
                    try:
                        tester.analyze_model_explainability()
                    except Exception as e:
                        st.warning(f"SHAP analysis failed: {str(e)}. Continuing with other tests.")
                progress_bar.progress(66)
                
                # Performance analysis
                if not hasattr(tester, 'performance_metrics') or not tester.performance_metrics:
                    tester.analyze_performance_and_resources(n_iterations=n_iterations)
                progress_bar.progress(90)
                
                # Final assessment
                tester.assess_model_limitations()
                tester.summarize_findings()
                tester.save_results()
                progress_bar.progress(100)
                
                st.success("Advanced testing completed successfully!")
                
                # Display cross-validation results
                if hasattr(tester, 'cv_results') and tester.cv_results:
                    with st.expander("Cross-Validation Results", expanded=True):
                        cv_col1, cv_col2, cv_col3 = st.columns(3)
                        
                        with cv_col1:
                            if 'accuracy' in tester.cv_results:
                                st.metric("CV Accuracy", 
                                        f"{tester.cv_results['accuracy']['mean']:.4f} ± {tester.cv_results['accuracy']['std']:.4f}")
                        
                        with cv_col2:
                            if 'recall_macro' in tester.cv_results:
                                st.metric("CV Recall", 
                                        f"{tester.cv_results['recall_macro']['mean']:.4f} ± {tester.cv_results['recall_macro']['std']:.4f}")
                        
                        with cv_col3:
                            if 'f1_macro' in tester.cv_results:
                                st.metric("CV F1 Score", 
                                        f"{tester.cv_results['f1_macro']['mean']:.4f} ± {tester.cv_results['f1_macro']['std']:.4f}")
                        
                        # Show cross-validation visualizations
                        display_image_if_exists("program2_cross_validation_results.png", 
                                              "Cross-Validation Results")
                        display_image_if_exists("program2_learning_curve.png", 
                                              "Learning Curve")
                
                # Display SHAP results (if available)
                if perform_shap:
                    with st.expander("Explainability Analysis (SHAP)", expanded=False):
                        st.info("Note: SHAP visualization may have errors. Missing images are likely due to those errors.")
                        
                        shap_col1, shap_col2 = st.columns(2)
                        
                        with shap_col1:
                            display_image_if_exists("program2_shap_feature_importance.png", 
                                                  "SHAP Feature Importance")
                        
                        with shap_col2:
                            display_image_if_exists("program2_shap_summary_plot.png", 
                                                  "SHAP Summary Plot")
                        
                        display_image_if_exists("program2_shap_decision_plot.png", 
                                              "SHAP Decision Plot")
                
                # Display performance metrics
                if hasattr(tester, 'performance_metrics') and tester.performance_metrics:
                    with st.expander("Performance Metrics", expanded=True):
                        perf_col1, perf_col2 = st.columns(2)
                        
                        with perf_col1:
                            st.metric("Avg Prediction Time (ms)", 
                                    f"{tester.performance_metrics['avg_prediction_time']*1000:.2f}")
                            st.metric("Memory Usage (MB)", 
                                    f"{tester.performance_metrics['memory_usage_mb']:.2f}")
                        
                        with perf_col2:
                            display_image_if_exists("program2_prediction_time_distribution.png", 
                                                  "Prediction Time Distribution")
                            display_image_if_exists("program2_memory_usage.png", 
                                                  "Memory Usage")
                
                # Display findings
                if hasattr(tester, 'findings'):
                    with st.expander("Model Assessment", expanded=True):
                        st.subheader("Model Strengths")
                        if tester.findings['strengths']:
                            for strength in tester.findings['strengths']:
                                st.markdown(f"✅ {strength}")
                        else:
                            st.info("No specific strengths identified.")
                        
                        st.subheader("Model Limitations")
                        if tester.findings['limitations']:
                            for limitation in tester.findings['limitations']:
                                st.markdown(f"⚠️ {limitation}")
                        else:
                            st.info("No specific limitations identified.")
            
            except Exception as e:
                st.error(f"Error during advanced testing: {str(e)}")
                st.exception(e)

def run_robustness_testing():
    """Run robustness testing (Program 3 functionality)."""
    st.markdown('## Robustness Testing')
    
    if not PROGRAM3_AVAILABLE or not PROGRAM1_AVAILABLE:
        st.error("Program1 or Program3 module is not available. Cannot perform robustness testing.")
        return
    
    st.subheader("Testing Model Performance with Outliers")
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_outliers = st.slider("Number of Artificial Outliers", 5, 20, 10)
        outlier_scale = st.slider("Outlier Intensity Scale", 1.0, 5.0, 3.0, 0.5)
    
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        contamination = st.slider("Outlier Detection Contamination", 0.01, 0.1, 0.05, 0.01)
    
    load_model = st.checkbox("Load pre-trained model", True)
    load_results = st.checkbox("Load previous test results if available", False)
    
    if st.button("Run Robustness Testing"):
        with st.spinner("Running robustness testing..."):
            try:
                # Step 1: Make sure we have a trained model
                iris_clf = IrisClassifier()
                
                if load_model:
                    if os.path.exists("iris_classifier_model.pkl"):
                        iris_clf.load_model()
                        st.info("Pre-trained model loaded successfully.")
                    else:
                        st.warning("No pre-trained model found. Training a new model...")
                        iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                else:
                    iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                
                # Step 2: Initialize and run robustness tester
                # Try to load the tester from Program 2 (not essential for Program 3)
                tester = None
                if PROGRAM2_AVAILABLE:
                    try:
                        tester = ModelTester(iris_clf)
                        tester.load_results()
                    except Exception:
                        pass
                
                robustness_tester = RobustnessTester(iris_clf, tester)
                
                if load_results and os.path.exists("robustness_tester_results.pkl"):
                    # Note: Program 3 doesn't have a direct load_results method,
                    # so we're not loading previous results yet
                    st.info("(Note: Loading previous robustness results is not yet implemented.)")
                
                progress_bar = st.progress(0)
                
                # Identify existing outliers
                robustness_tester.identify_outliers(contamination=contamination)
                progress_bar.progress(20)
                
                # Generate artificial outliers
                robustness_tester.generate_artificial_outliers(
                    n_outliers=n_outliers, outlier_scale=outlier_scale)
                progress_bar.progress(40)
                
                # Evaluate model robustness
                robustness_tester.evaluate_model_robustness()
                progress_bar.progress(60)
                
                # Cross-validation with outliers
                robustness_tester.perform_cross_validation(cv=cv_folds)
                progress_bar.progress(80)
                
                # Detect inadequacies and summarize
                robustness_tester.detect_test_inadequacies()
                robustness_tester.summarize_findings()
                robustness_tester.save_results()
                progress_bar.progress(100)
                
                st.success("Robustness testing completed successfully!")
                
                # Display existing outliers results
                if hasattr(robustness_tester, 'existing_outliers') and robustness_tester.existing_outliers:
                    with st.expander("Existing Outliers Analysis", expanded=True):
                        st.metric("Number of Outliers", f"{robustness_tester.existing_outliers['count']}")
                        st.metric("Percentage of Dataset", 
                                f"{robustness_tester.existing_outliers['percentage']:.2f}%")
                        
                        out_col1, out_col2 = st.columns(2)
                        
                        with out_col1:
                            display_image_if_exists("program3_existing_outliers.png", 
                                                  "Existing Outliers Visualization")
                        
                        with out_col2:
                            display_image_if_exists("program3_outliers_boxplot.png", 
                                                  "Outliers Boxplot")
                
                # Display artificial outliers results
                if hasattr(robustness_tester, 'X_with_outliers') and robustness_tester.X_with_outliers is not None:
                    with st.expander("Artificial Outliers", expanded=True):
                        display_image_if_exists("program3_artificial_outliers.png", 
                                              "Artificial Outliers Visualization")
                
                # Display robustness evaluation results
                if (hasattr(robustness_tester, 'original_performance') and 
                    hasattr(robustness_tester, 'outlier_performance')):
                    with st.expander("Robustness Evaluation", expanded=True):
                        # Performance metrics comparison
                        st.subheader("Performance Impact of Outliers")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            if 'accuracy' in robustness_tester.original_performance:
                                st.metric("Accuracy", 
                                        f"{robustness_tester.original_performance['accuracy']:.4f}",
                                        f"{-robustness_tester.outlier_performance.get('accuracy_impact', 0):.2f}%")
                        
                        with metric_col2:
                            if 'recall' in robustness_tester.original_performance:
                                st.metric("Recall", 
                                        f"{robustness_tester.original_performance['recall']:.4f}",
                                        f"{-robustness_tester.outlier_performance.get('recall_impact', 0):.2f}%")
                        
                        with metric_col3:
                            if 'f1' in robustness_tester.original_performance:
                                st.metric("F1 Score", 
                                        f"{robustness_tester.original_performance['f1']:.4f}",
                                        f"{-robustness_tester.outlier_performance.get('f1_impact', 0):.2f}%")
                        
                        # Visualization of robustness results
                        rob_col1, rob_col2 = st.columns(2)
                        
                        with rob_col1:
                            display_image_if_exists("program3_robustness_metrics.png", 
                                                  "Robustness Metrics")
                        
                        with rob_col2:
                            display_image_if_exists("program3_robustness_confusion.png", 
                                                  "Confusion Matrix Comparison")
                        
                        display_image_if_exists("program3_outlier_impact.png", 
                                              "Impact of Outliers on Classification")
                
                # Display cross-validation results
                if hasattr(robustness_tester, 'cv_results') and robustness_tester.cv_results:
                    with st.expander("Cross-Validation with Outliers", expanded=True):
                        cv_col1, cv_col2 = st.columns(2)
                        
                        with cv_col1:
                            st.metric("CV Score (clean)", 
                                    f"{robustness_tester.cv_results['clean_mean']:.4f} ± {robustness_tester.cv_results['clean_std']:.4f}")
                            st.metric("CV Score (with outliers)", 
                                    f"{robustness_tester.cv_results['outlier_mean']:.4f} ± {robustness_tester.cv_results['outlier_std']:.4f}")
                            st.metric("Average Impact", 
                                    f"{robustness_tester.cv_results['avg_impact']:.2f}%")
                        
                        with cv_col2:
                            display_image_if_exists("program3_cv_outlier_comparison.png", 
                                                  "Cross-Validation with Outliers")
                
                # Display test inadequacies
                if hasattr(robustness_tester, 'inadequacy_findings') and robustness_tester.inadequacy_findings:
                    with st.expander("Test Inadequacies", expanded=True):
                        st.subheader("Missing Test Components")
                        
                        for i, finding in enumerate(robustness_tester.inadequacy_findings):
                            with st.container():
                                st.markdown(f"**{i+1}. {finding.get('issue', 'Issue')} in {finding.get('program', 'Unknown')}**")
                                st.markdown(f"Description: {finding.get('description', 'No description')}")
                                st.markdown(f"Recommendation: {finding.get('recommendation', 'No recommendation')}")
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"Error during robustness testing: {str(e)}")
                st.exception(e)

def run_hyperparameter_optimization():
    """Run hyperparameter optimization (from Program 4)."""
    st.markdown('## Hyperparameter Optimization')
    
    if not PROGRAM4_AVAILABLE or not PROGRAM1_AVAILABLE:
        st.error("Program1 or Program4 module is not available. Cannot perform hyperparameter optimization.")
        return
    
    st.subheader("Finding Optimal Model Parameters")
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        min_neighbors = st.slider("Min Neighbors (k)", 1, 10, 3)
        max_neighbors = st.slider("Max Neighbors (k)", min_neighbors, 20, 15)
        
        # Generate neighbors range based on user selection
        neighbors_range = list(range(min_neighbors, max_neighbors + 1, 2))
        if len(neighbors_range) < 3:  # Ensure at least 3 values for visualization
            neighbors_range = list(range(min_neighbors, max_neighbors + 1))
        
        st.write(f"Neighbors to test: {neighbors_range}")
    
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        weights_options = st.multiselect("Weight Options", 
                                       ["uniform", "distance"], 
                                       ["uniform", "distance"])
        metrics_options = st.multiselect("Distance Metrics", 
                                      ["euclidean", "manhattan", "minkowski"], 
                                      ["euclidean", "manhattan", "minkowski"])
    
    load_model = st.checkbox("Load pre-trained model", True)
    use_custom_grid = st.checkbox("Use custom parameter grid", True)
    
    if st.button("Run Hyperparameter Optimization"):
        with st.spinner("Optimizing hyperparameters... This may take a while."):
            try:
                # Step 1: Make sure we have a trained model
                iris_clf = IrisClassifier()
                
                if load_model:
                    if os.path.exists("iris_classifier_model.pkl"):
                        iris_clf.load_model()
                        st.info("Pre-trained model loaded successfully.")
                    else:
                        st.warning("No pre-trained model found. Training a new model...")
                        iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                else:
                    iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                
                # Step 2: Initialize advanced tester
                advanced_tester = AdvancedTester(iris_clf)
                
                # Create custom parameter grid if selected
                if use_custom_grid:
                    param_grid = {
                        'n_neighbors': neighbors_range,
                        'weights': weights_options,
                        'metric': metrics_options,
                        'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
                    }
                    
                    if len(metrics_options) == 1 and metrics_options[0] != 'minkowski':
                        # If only one metric selected and it's not minkowski, don't include p
                        del param_grid['p']
                    
                    # Run hyperparameter optimization with custom grid
                    advanced_tester.optimize_hyperparameters(param_grid=param_grid, cv=cv_folds)
                else:
                    # Run hyperparameter optimization with default grid
                    advanced_tester.optimize_hyperparameters(cv=cv_folds)
                
                # Save results
                advanced_tester.save_results()
                
                st.success("Hyperparameter optimization completed successfully!")
                
                # Display optimization results
                if (hasattr(advanced_tester, 'optimization_results') and 
                    advanced_tester.optimization_results['best_params']):
                    with st.expander("Optimization Results", expanded=True):
                        st.subheader("Best Hyperparameters")
                        
                        # Display best parameters
                        best_params = advanced_tester.optimization_results['best_params']
                        for param, value in best_params.items():
                            st.metric(param, f"{value}")
                        
                        st.metric("Best Cross-Validation Score", 
                                f"{advanced_tester.optimization_results['best_score']:.4f}")
                        
                        # Compare with original model
                        if hasattr(iris_clf, 'accuracy'):
                            improvement = ((advanced_tester.optimization_results['best_score'] - iris_clf.accuracy) / 
                                         iris_clf.accuracy * 100)
                            st.metric("Improvement over Original Model", 
                                    f"{improvement:.2f}%")
                        
                        # Show parameter importance
                        st.subheader("Parameter Importance")
                        
                        importance_df = pd.DataFrame({
                            'Parameter': list(advanced_tester.optimization_results['param_importance'].keys()),
                            'Importance': list(advanced_tester.optimization_results['param_importance'].values())
                        })
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        st.bar_chart(importance_df.set_index('Parameter'))
                        
                        # Show visualizations
                        st.subheader("Optimization Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            display_image_if_exists("program4_hyperparameter_heatmap.png", 
                                                  "Hyperparameter Heatmap")
                        
                        with viz_col2:
                            display_image_if_exists("program4_hyperparameter_plots.png", 
                                                  "Hyperparameter Impact Plots")
                        
                        # Display key findings
                        if hasattr(advanced_tester, 'advanced_findings'):
                            st.subheader("Key Findings")
                            
                            for finding in advanced_tester.advanced_findings['hyperparameter_opt']:
                                st.markdown(f"📊 {finding}")
            
            except Exception as e:
                st.error(f"Error during hyperparameter optimization: {str(e)}")
                st.exception(e)

def run_adversarial_testing():
    """Run adversarial testing (from Program 4)."""
    st.markdown('## Adversarial Testing')
    
    if not PROGRAM4_AVAILABLE or not PROGRAM1_AVAILABLE:
        st.error("Program1 or Program4 module is not available. Cannot perform adversarial testing.")
        return
    
    st.subheader("Testing Model Vulnerability to Adversarial Examples")
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_examples = st.slider("Number of Adversarial Examples", 5, 30, 20)
        epsilon = st.slider("Perturbation Step Size (Epsilon)", 0.05, 0.5, 0.2, 0.05)
    
    with col2:
        use_optimized = st.checkbox("Use optimized model if available", True)
        load_model = st.checkbox("Load pre-trained model", True)
    
    if st.button("Run Adversarial Testing"):
        with st.spinner("Generating and testing adversarial examples..."):
            try:
                # Step 1: Make sure we have a trained model
                iris_clf = IrisClassifier()
                
                if load_model:
                    if os.path.exists("iris_classifier_model.pkl"):
                        iris_clf.load_model()
                        st.info("Pre-trained model loaded successfully.")
                    else:
                        st.warning("No pre-trained model found. Training a new model...")
                        iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                else:
                    iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                
                # Step 2: Initialize advanced tester
                advanced_tester = AdvancedTester(iris_clf)
                
                # Step 3: Load optimized model if requested and available
                if use_optimized:
                    if os.path.exists("advanced_tester_results.pkl"):
                        try:
                            with open("advanced_tester_results.pkl", "rb") as f:
                                results = pickle.load(f)
                            
                            if ('hyperparameter_optimization' in results and 
                                'best_params' in results['hyperparameter_optimization']):
                                # Run hyperparameter optimization to get the optimized model
                                st.info("Optimized model found. Running hyperparameter optimization to recreate it...")
                                advanced_tester.optimize_hyperparameters()
                                st.success("Optimized model loaded successfully.")
                            else:
                                st.warning("No optimized model parameters found in previous results. Using original model.")
                        except Exception as e:
                            st.warning(f"Error loading optimized model: {str(e)}. Using original model.")
                    else:
                        st.warning("No previous results found. Using original model.")
                
                # Step 4: Generate and test adversarial examples
                advanced_tester.generate_adversarial_examples(n_examples=n_examples, epsilon=epsilon)
                
                # Save results
                advanced_tester.save_results()
                
                st.success("Adversarial testing completed successfully!")
                
                # Display adversarial testing results
                if (hasattr(advanced_tester, 'adversarial_examples') and 
                    'success_rate' in advanced_tester.adversarial_examples):
                    with st.expander("Adversarial Testing Results", expanded=True):
                        # Display success metrics
                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        
                        with adv_col1:
                            st.metric("Success Rate", 
                                    f"{advanced_tester.adversarial_examples['success_rate']*100:.2f}%")
                        
                        with adv_col2:
                            if 'avg_perturbation' in advanced_tester.adversarial_examples:
                                st.metric("Avg Perturbation Magnitude", 
                                        f"{advanced_tester.adversarial_examples['avg_perturbation']:.4f}")
                        
                        with adv_col3:
                            st.metric("Examples Generated", 
                                    f"{len(advanced_tester.adversarial_examples['X_original'])}")
                        
                        # Vulnerability assessment
                        success_rate = advanced_tester.adversarial_examples['success_rate']
                        
                        if success_rate > 0.5:
                            st.error("⚠️ HIGH VULNERABILITY: Model is easily fooled by adversarial examples")
                        elif success_rate > 0.1:
                            st.warning("⚠️ MODERATE VULNERABILITY: Model can be fooled by carefully crafted examples")
                        else:
                            st.success("✅ LOW VULNERABILITY: Model is relatively robust against adversarial examples")
                        
                        # Show visualizations
                        st.subheader("Adversarial Example Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            display_image_if_exists("program4_adversarial_features.png", 
                                                  "Adversarial Feature Perturbations")
                        
                        with viz_col2:
                            display_image_if_exists("program4_adversarial_boundaries.png", 
                                                  "Decision Boundaries with Adversarial Examples")
                        
                        # Display key findings
                        if hasattr(advanced_tester, 'advanced_findings'):
                            st.subheader("Key Findings")
                            
                            for finding in advanced_tester.advanced_findings['adversarial']:
                                st.markdown(f"🔍 {finding}")
            
            except Exception as e:
                st.error(f"Error during adversarial testing: {str(e)}")
                st.exception(e)

def run_data_drift_simulation():
    """Run data drift simulation (from Program 4)."""
    st.markdown('## Data Drift Simulation')
    
    if not PROGRAM4_AVAILABLE or not PROGRAM1_AVAILABLE:
        st.error("Program1 or Program4 module is not available. Cannot perform data drift simulation.")
        return
    
    st.subheader("Testing Model Performance Under Data Drift")
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        drift_types = st.multiselect("Drift Types to Simulate", 
                                   ["covariate", "concept", "feature"], 
                                   ["covariate", "concept", "feature"])
        n_simulations = st.slider("Simulations per Drift Type", 1, 5, 3)
    
    with col2:
        use_optimized = st.checkbox("Use optimized model if available", True)
        load_model = st.checkbox("Load pre-trained model", True)
    
    if st.button("Run Data Drift Simulation"):
        with st.spinner("Simulating data drift scenarios..."):
            try:
                # Step 1: Make sure we have a trained model
                iris_clf = IrisClassifier()
                
                if load_model:
                    if os.path.exists("iris_classifier_model.pkl"):
                        iris_clf.load_model()
                        st.info("Pre-trained model loaded successfully.")
                    else:
                        st.warning("No pre-trained model found. Training a new model...")
                        iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                else:
                    iris_clf.load_data().prepare_data().train_model().evaluate_model().save_model()
                
                # Step 2: Initialize advanced tester
                advanced_tester = AdvancedTester(iris_clf)
                
                # Step 3: Load optimized model if requested and available
                if use_optimized:
                    if os.path.exists("advanced_tester_results.pkl"):
                        try:
                            with open("advanced_tester_results.pkl", "rb") as f:
                                results = pickle.load(f)
                            
                            if ('hyperparameter_optimization' in results and 
                                'best_params' in results['hyperparameter_optimization']):
                                # Run hyperparameter optimization to get the optimized model
                                st.info("Optimized model found. Running hyperparameter optimization to recreate it...")
                                advanced_tester.optimize_hyperparameters()
                                st.success("Optimized model loaded successfully.")
                            else:
                                st.warning("No optimized model parameters found in previous results. Using original model.")
                        except Exception as e:
                            st.warning(f"Error loading optimized model: {str(e)}. Using original model.")
                    else:
                        st.warning("No previous results found. Using original model.")
                
                # Step 4: Simulate data drift
                advanced_tester.simulate_data_drift(
                    drift_types=drift_types, n_simulations=n_simulations)
                
                # Save results
                advanced_tester.save_results()
                
                st.success("Data drift simulation completed successfully!")
                
                # Display drift simulation results
                if (hasattr(advanced_tester, 'drift_simulation_results') and 
                    'performance_impact' in advanced_tester.drift_simulation_results):
                    with st.expander("Data Drift Simulation Results", expanded=True):
                        performance_impact = advanced_tester.drift_simulation_results['performance_impact']
                        
                        st.subheader("Performance Impact by Drift Type")
                        
                        # Create tabs for each drift type
                        if drift_types:
                            tabs = st.tabs(drift_types)
                            
                            for i, drift_type in enumerate(drift_types):
                                with tabs[i]:
                                    if drift_type in performance_impact:
                                        impact = performance_impact[drift_type]
                                        
                                        # Metrics
                                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                                        
                                        with metric_col1:
                                            if 'accuracy' in impact:
                                                st.metric("Accuracy Impact", 
                                                        f"{impact['accuracy']:.2f}% degradation")
                                        
                                        with metric_col2:
                                            if 'recall' in impact:
                                                st.metric("Recall Impact", 
                                                        f"{impact['recall']:.2f}% degradation")
                                        
                                        with metric_col3:
                                            if 'f1' in impact:
                                                st.metric("F1 Score Impact", 
                                                        f"{impact['f1']:.2f}% degradation")
                                        
                                        # Description of drift type
                                        if drift_type == 'covariate':
                                            st.info("Covariate drift occurs when the distribution of input features " +
                                                  "changes while the relationship between features and target remains the same.")
                                        elif drift_type == 'concept':
                                            st.info("Concept drift occurs when the relationship between features and " +
                                                  "target changes, even if the feature distribution remains the same.")
                                        elif drift_type == 'feature':
                                            st.info("Feature drift occurs when the relationships between features change " +
                                                  "(e.g., correlation structure), affecting the model's performance.")
                                    else:
                                        st.warning(f"No impact data available for {drift_type} drift.")
                        
                        # Overall vulnerability assessment
                        st.subheader("Overall Vulnerability Assessment")
                        
                        max_impact = max(
                            impact.get('accuracy', 0) 
                            for impact in performance_impact.values()
                        ) if performance_impact else 0
                        
                        if max_impact > 20:
                            st.error("⚠️ HIGH VULNERABILITY: Model performance degrades significantly under data drift")
                        elif max_impact > 5:
                            st.warning("⚠️ MODERATE VULNERABILITY: Model shows some sensitivity to data drift")
                        else:
                            st.success("✅ LOW VULNERABILITY: Model is relatively robust against data drift")
                        
                        # Show visualizations
                        st.subheader("Data Drift Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            display_image_if_exists("program4_drift_impact.png", 
                                                  "Drift Impact on Performance")
                        
                        with viz_col2:
                            display_image_if_exists("program4_drift_distributions.png", 
                                                  "Feature Distribution Shifts")
                        
                        display_image_if_exists("program4_drift_boundaries.png", 
                                              "Decision Boundary Changes Under Drift")
                        
                        # Display key findings
                        if hasattr(advanced_tester, 'advanced_findings'):
                            st.subheader("Key Findings")
                            
                            for finding in advanced_tester.advanced_findings['data_drift']:
                                st.markdown(f"📉 {finding}")
            
            except Exception as e:
                st.error(f"Error during data drift simulation: {str(e)}")
                st.exception(e)

def run_comprehensive_testing():
    """Run a comprehensive testing suite combining all tests."""
    st.markdown('## Comprehensive ML Testing')
    
    if (not PROGRAM1_AVAILABLE or not PROGRAM2_AVAILABLE or 
        not PROGRAM3_AVAILABLE or not PROGRAM4_AVAILABLE):
        st.error("One or more program modules are not available. Cannot perform comprehensive testing.")
        return
    
    st.subheader("Running All Tests in Sequence")
    
    # Create columns for key parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_neighbors = st.slider("KNN Neighbors (k)", 1, 15, 3)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col2:
        n_outliers = st.slider("Outliers to Generate", 5, 15, 10)
        n_adversarial = st.slider("Adversarial Examples", 5, 20, 10)
    
    with col3:
        optimize_params = st.checkbox("Optimize Hyperparameters", True)
        run_adv_tests = st.checkbox("Run Adversarial Tests", True)
        run_drift_tests = st.checkbox("Run Drift Tests", True)
    
    if st.button("Run Comprehensive Testing"):
        with st.spinner("Running comprehensive testing suite... This may take several minutes."):
            try:
                # Create containers to organize the output
                basic_container = st.container()
                advanced_container = st.container()
                robustness_container = st.container()
                hyperopt_container = st.container()
                adv_container = st.container()
                drift_container = st.container()
                summary_container = st.container()
                
                with basic_container:
                    st.markdown("### 1. Basic Testing")
                    progress_bar = st.progress(0)
                    
                    # Basic testing (Program 1)
                    iris_clf = IrisClassifier()
                    iris_clf.load_data()
                    progress_bar.progress(20)
                    
                    iris_clf.prepare_data()
                    progress_bar.progress(40)
                    
                    iris_clf.train_model(n_neighbors=n_neighbors)
                    progress_bar.progress(60)
                    
                    iris_clf.evaluate_model()
                    progress_bar.progress(80)
                    
                    iris_clf.save_model()
                    progress_bar.progress(100)
                    
                    st.success("Basic testing complete!")
                    
                    # Show key metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Accuracy", f"{iris_clf.accuracy:.4f}")
                    
                    with metrics_col2:
                        st.metric("Recall", f"{iris_clf.recall:.4f}")
                    
                    with metrics_col3:
                        st.metric("F1 Score", f"{iris_clf.f1:.4f}")
                
                with advanced_container:
                    st.markdown("### 2. Advanced Testing")
                    progress_bar = st.progress(0)
                    
                    # Advanced testing (Program 2)
                    tester = ModelTester(iris_clf)
                    
                    tester.perform_cross_validation(cv=cv_folds)
                    progress_bar.progress(33)
                    
                    # Skip SHAP analysis since it has known issues
                    progress_bar.progress(66)
                    
                    tester.analyze_performance_and_resources()
                    tester.assess_model_limitations()
                    tester.save_results()
                    progress_bar.progress(100)
                    
                    st.success("Advanced testing complete!")
                    
                    # Show key metrics
                    if tester.cv_results and 'accuracy' in tester.cv_results:
                        st.metric("Cross-Val Accuracy", 
                                f"{tester.cv_results['accuracy']['mean']:.4f} ± {tester.cv_results['accuracy']['std']:.4f}")
                
                with robustness_container:
                    st.markdown("### 3. Robustness Testing")
                    progress_bar = st.progress(0)
                    
                    # Robustness testing (Program 3)
                    robustness_tester = RobustnessTester(iris_clf, tester)
                    
                    robustness_tester.identify_outliers()
                    progress_bar.progress(25)
                    
                    robustness_tester.generate_artificial_outliers(n_outliers=n_outliers)
                    progress_bar.progress(50)
                    
                    robustness_tester.evaluate_model_robustness()
                    progress_bar.progress(75)
                    
                    robustness_tester.perform_cross_validation(cv=cv_folds)
                    robustness_tester.detect_test_inadequacies()
                    robustness_tester.save_results()
                    progress_bar.progress(100)
                    
                    st.success("Robustness testing complete!")
                    
                    # Show key metrics
                    if (hasattr(robustness_tester, 'outlier_performance') and 
                        'accuracy_impact' in robustness_tester.outlier_performance):
                        st.metric("Outlier Impact on Accuracy", 
                                f"{robustness_tester.outlier_performance['accuracy_impact']:.2f}%")
                
                # Initialize advanced tester
                advanced_tester = AdvancedTester(iris_clf)
                
                if optimize_params:
                    with hyperopt_container:
                        st.markdown("### 4. Hyperparameter Optimization")
                        
                        with st.spinner("Running hyperparameter optimization..."):
                            advanced_tester.optimize_hyperparameters(cv=cv_folds)
                        
                        st.success("Hyperparameter optimization complete!")
                        
                        # Show best parameters
                        if (hasattr(advanced_tester, 'optimization_results') and 
                            'best_params' in advanced_tester.optimization_results):
                            best_params = advanced_tester.optimization_results['best_params']
                            best_score = advanced_tester.optimization_results['best_score']
                            
                            st.metric("Best CV Score", f"{best_score:.4f}")
                            st.json(best_params)
                
                if run_adv_tests:
                    with adv_container:
                        st.markdown("### 5. Adversarial Testing")
                        
                        with st.spinner("Running adversarial testing..."):
                            advanced_tester.generate_adversarial_examples(n_examples=n_adversarial)
                        
                        st.success("Adversarial testing complete!")
                        
                        # Show success rate
                        if (hasattr(advanced_tester, 'adversarial_examples') and 
                            'success_rate' in advanced_tester.adversarial_examples):
                            st.metric("Adversarial Success Rate", 
                                    f"{advanced_tester.adversarial_examples['success_rate']*100:.2f}%")
                
                if run_drift_tests:
                    with drift_container:
                        st.markdown("### 6. Data Drift Simulation")
                        
                        with st.spinner("Running data drift simulation..."):
                            advanced_tester.simulate_data_drift()
                        
                        st.success("Data drift simulation complete!")
                        
                        # Show max impact
                        if (hasattr(advanced_tester, 'drift_simulation_results') and 
                            'performance_impact' in advanced_tester.drift_simulation_results):
                            
                            performance_impact = advanced_tester.drift_simulation_results['performance_impact']
                            
                            if performance_impact:
                                max_impact = max(
                                    impact.get('accuracy', 0) 
                                    for impact in performance_impact.values()
                                )
                                
                                st.metric("Max Accuracy Impact from Drift", f"{max_impact:.2f}%")
                
                # Save advanced tester results
                advanced_tester.save_results()
                
                with summary_container:
                    st.markdown("### Comprehensive Testing Summary")
                    
                    # Create summary metrics
                    st.subheader("Key Performance Metrics")
                    
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("Base Model Accuracy", f"{iris_clf.accuracy:.4f}")
                    
                    with summary_col2:
                        if (hasattr(advanced_tester, 'optimization_results') and 
                            'best_score' in advanced_tester.optimization_results):
                            st.metric("Optimized Model Accuracy", 
                                    f"{advanced_tester.optimization_results['best_score']:.4f}",
                                    f"{(advanced_tester.optimization_results['best_score'] - iris_clf.accuracy) / iris_clf.accuracy * 100:.2f}%")
                        else:
                            st.metric("Optimized Model Accuracy", "N/A")
                    
                    with summary_col3:
                        if (hasattr(robustness_tester, 'outlier_performance') and 
                            'accuracy_impact' in robustness_tester.outlier_performance):
                            st.metric("Outlier Vulnerability", 
                                    f"{robustness_tester.outlier_performance['accuracy_impact']:.2f}%")
                        else:
                            st.metric("Outlier Vulnerability", "N/A")
                    
                    with summary_col4:
                        if run_adv_tests and hasattr(advanced_tester, 'adversarial_examples'):
                            st.metric("Adversarial Vulnerability", 
                                    f"{advanced_tester.adversarial_examples['success_rate']*100:.2f}%")
                        else:
                            st.metric("Adversarial Vulnerability", "N/A")
                    
                    # Overall assessment
                    st.subheader("Overall Assessment")
                    
                    # Collect strengths and weaknesses
                    strengths = []
                    weaknesses = []
                    
                    # From Program 2
                    if hasattr(tester, 'findings'):
                        strengths.extend(tester.findings['strengths'])
                        weaknesses.extend(tester.findings['limitations'])
                    
                    # From Program 4
                    if hasattr(advanced_tester, 'advanced_findings'):
                        for findings in advanced_tester.advanced_findings.values():
                            for finding in findings:
                                if "robust" in finding.lower() or "low vulnerability" in finding.lower():
                                    strengths.append(finding)
                                elif "vulnerability" in finding.lower() or "drop" in finding.lower():
                                    weaknesses.append(finding)
                    
                    # Display strengths and weaknesses
                    st.markdown("**Model Strengths:**")
                    for strength in strengths:
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("**Model Weaknesses:**")
                    for weakness in weaknesses:
                        st.markdown(f"⚠️ {weakness}")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    recommendations = []
                    
                    # Add recommendations based on test results
                    if (hasattr(advanced_tester, 'optimization_results') and 
                        'best_score' in advanced_tester.optimization_results and
                        advanced_tester.optimization_results['best_score'] > iris_clf.accuracy):
                        recommendations.append("Implement optimized hyperparameters to improve model performance.")
                    
                    if (hasattr(robustness_tester, 'outlier_performance') and 
                        'accuracy_impact' in robustness_tester.outlier_performance and
                        robustness_tester.outlier_performance['accuracy_impact'] > 5):
                        recommendations.append("Add outlier detection and handling to improve robustness.")
                    
                    if (run_adv_tests and hasattr(advanced_tester, 'adversarial_examples') and
                        advanced_tester.adversarial_examples['success_rate'] > 0.1):
                        recommendations.append("Consider adversarial training to improve model security.")
                    
                    if (run_drift_tests and hasattr(advanced_tester, 'drift_simulation_results') and
                        'performance_impact' in advanced_tester.drift_simulation_results):
                        performance_impact = advanced_tester.drift_simulation_results['performance_impact']
                        if performance_impact:
                            max_impact = max(
                                impact.get('accuracy', 0) 
                                for impact in performance_impact.values()
                            )
                            if max_impact > 10:
                                recommendations.append("Implement drift detection and monitoring for production deployments.")
                    
                    # Display recommendations
                    for recommendation in recommendations:
                        st.markdown(f"📋 {recommendation}")
                    
                    if not recommendations:
                        st.markdown("📋 No critical issues found. The model appears suitable for use as is.")
                
                st.success("Comprehensive testing completed successfully!")
            
            except Exception as e:
                st.error(f"Error during comprehensive testing: {str(e)}")
                st.exception(e)


def main():
    """Main function to run the Streamlit application."""
    configure_page()
    display_header()

    # Ensure directories exist
    from utils.data_handler import ensure_directories
    ensure_directories()

    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Testing Module",
        [
            "Overview", 
            "Basic Testing",
            "Advanced Testing",
            "Robustness Testing",
            "Hyperparameter Optimization",
            "Adversarial Testing",
            "Data Drift Simulation",
            "Comprehensive Testing"
        ]
    )
    
    # Display about information in the sidebar
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        **ML Testing Dashboard** is a comprehensive tool for testing machine learning models,
        with a focus on KNN classification using the Iris dataset.
        
        This dashboard provides different testing approaches:
        - Basic model evaluation
        - Cross-validation and explainability
        - Robustness to outliers
        - Hyperparameter optimization
        - Adversarial testing
        - Data drift simulation
        
        Use the navigation menu to explore different testing modules.
        
        Version: 1.0.0
        """)
    
    # Display page based on selection
    if page == "Overview":
        st.markdown("## ML Testing Overview")
        
        st.markdown("""
        Machine learning testing is a critical part of the ML development lifecycle. 
        This dashboard provides a suite of tools to comprehensively test your models.
        
        ### Testing Categories in this Dashboard:
        
        1. **Basic Testing (Program 1)**
           - Model training and evaluation
           - Basic performance metrics (accuracy, recall, F1)
           - Decision boundary visualization
           
        2. **Advanced Testing (Program 2)**
           - Cross-validation to assess generalizability
           - Model explainability with SHAP (Shapley Additive Explanations)
           - Performance and resource usage analysis
           
        3. **Robustness Testing (Program 3)**
           - Detection of existing outliers
           - Generation of artificial outliers
           - Assessment of model performance with outliers
           
        4. **Hyperparameter Optimization (Program 4, Part 1)**
           - Systematic search for optimal parameters
           - Parameter importance analysis
           - Performance improvement quantification
           
        5. **Adversarial Testing (Program 4, Part 2)**
           - Generation of adversarial examples
           - Assessment of model vulnerability
           - Visualization of decision boundary modifications
           
        6. **Data Drift Simulation (Program 4, Part 3)**
           - Simulation of various types of data drift:
             - Covariate drift (feature distribution changes)
             - Concept drift (relationship between features and labels changes)
             - Feature drift (new correlations between features)
           - Quantification of performance degradation under drift
        
        7. **Comprehensive Testing**
           - Integration of all test types
           - Overall assessment of model quality
           - Recommendations for improvement
        
        ### Getting Started:
        
        Use the navigation menu on the left to select a testing module and follow the steps to
        evaluate your model. Each module provides visualization and metrics to help you understand
        your model's strengths and weaknesses.
        """)
        
        # Display icons for each testing category
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 📊 Basic Testing")
            st.markdown("Train and evaluate a model on the Iris dataset using basic metrics.")
            
        with col2:
            st.markdown("### 🔍 Advanced Testing")
            st.markdown("Assess model generalizability and peek inside the black box.")
            
        with col3:
            st.markdown("### 💪 Robustness Testing")
            st.markdown("Test how well your model handles outliers and edge cases.")
            
        with col4:
            st.markdown("### ⚙️ Optimization")
            st.markdown("Fine-tune hyperparameters for optimal performance.")
            
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown("### 🛡️ Adversarial Testing")
            st.markdown("Assess model vulnerability to adversarial examples.")
            
        with col6:
            st.markdown("### 📉 Drift Simulation")
            st.markdown("Evaluate model robustness to changing data distributions.")
            
        with col7:
            st.markdown("### 🧠 Comprehensive Testing")
            st.markdown("Run all tests and get an overall assessment of your model.")
            
        with col8:
            st.markdown("### 📈 Visualization")
            st.markdown("All tests include interactive visualizations to understand results.")
    
    elif page == "Basic Testing":
        run_basic_testing()
    
    elif page == "Advanced Testing":
        run_advanced_testing()
    
    elif page == "Robustness Testing":
        run_robustness_testing()
    
    elif page == "Hyperparameter Optimization":
        run_hyperparameter_optimization()
    
    elif page == "Adversarial Testing":
        run_adversarial_testing()
    
    elif page == "Data Drift Simulation":
        run_data_drift_simulation()
    
    elif page == "Comprehensive Testing":
        run_comprehensive_testing()


if __name__ == "__main__":
    main()

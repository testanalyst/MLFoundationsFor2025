"""
Advanced Testing of Iris Classification Model
--------------------------------------------
Purpose: Evaluate the basic KNN model using explainability tools and cross-validation
Author: ML Testing Team
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score
import time
import psutil
import os
import pickle
import warnings

# Try to import shap, but handle the case where it's not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Explainability analysis will be limited.")

# Try to import from Program 1, but provide fallbacks if not available
try:
    from program1 import IrisClassifier
except ImportError:
    warnings.warn("Could not import IrisClassifier from program1. Using minimal compatibility class.")
    
    class IrisClassifier:
        """Minimal compatibility class for IrisClassifier when program1 isn't available."""
        def __init__(self):
            self.X = None
            self.y = None
            self.model = None
            self.feature_names = None
            self.target_names = None
            self.scaler = None
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.X_train_scaled = None
            self.X_test_scaled = None
            self.accuracy = None
            
        def load_model(self, filename='iris_classifier_model.pkl'):
            """Load a trained model from a file."""
            print(f"\nLoading model from {filename}...")
            
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Model file {filename} not found.")
                
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load model and essential attributes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            
            # Load optional attributes if available
            for attr, value in model_data.items():
                if value is not None and attr not in ['model', 'scaler', 'feature_names', 'target_names']:
                    setattr(self, attr, value)
            
            print(f"  ✓ Model loaded from {filename}")
            
            return self

# Set random seed for reproducibility
np.random.seed(42)


class ModelTester:
    """A class to thoroughly test the Iris classification model."""
    
    def __init__(self, classifier):
        """Initialize with a trained classifier from Program 1."""
        self.classifier = classifier
        
        

        self.X = getattr(classifier, 'X', None)
        self.y = getattr(classifier, 'y', None)

        # ADD THIS CODE BLOCK - Ensures data is available when loaded from pickle
        # If X/y are missing but we have training data, use that instead
        if self.X is None and hasattr(classifier, 'X_train') and classifier.X_train is not None:
            self.X = classifier.X_train
        
        if self.y is None and hasattr(classifier, 'y_train') and classifier.y_train is not None:
            self.y = classifier.y_train



        self.X_test = getattr(classifier, 'X_test', None)
        self.X_test_scaled = getattr(classifier, 'X_test_scaled', None)
        self.y_test = getattr(classifier, 'y_test', None)
        self.model = getattr(classifier, 'model', None)
        self.feature_names = getattr(classifier, 'feature_names', None)
        self.target_names = getattr(classifier, 'target_names', None)
        
        # Initialize other attributes
        self.cv_results = None
        self.shap_values = None
        self.explainer = None
        self.sample_indices = None
        self.prediction_times = None
        self.performance_metrics = {}
        self.findings = {'strengths': [], 'limitations': []}
        
        # Check if we have all the required attributes
        self._check_attributes()
    
    def _check_attributes(self):
        """Check if all required attributes are available."""
        required_attrs = ['X', 'y', 'X_test', 'X_test_scaled', 'y_test', 'model', 
                         'feature_names', 'target_names']
        missing = [attr for attr in required_attrs if getattr(self, attr) is None]
        
        if missing:
            warnings.warn(f"The following required attributes are missing: {missing}. "
                         f"Some functionality may be limited.")
        
    def perform_cross_validation(self, cv=5):
        """Perform k-fold cross-validation on the model."""
        print("\nStep 1: Performing Cross-Validation...")
        
        if self.model is None or self.X is None or self.y is None:
            print("  ! Cannot perform cross-validation: model or data not available.")
            self.cv_results = {
                'accuracy': {'mean': 0, 'std': 0, 'scores': np.array([0]), 'time': 0},
                'recall_macro': {'mean': 0, 'std': 0, 'scores': np.array([0]), 'time': 0},
                'f1_macro': {'mean': 0, 'std': 0, 'scores': np.array([0]), 'time': 0}
            }
            return self
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'recall_macro': make_scorer(recall_score, average='macro'),
            'f1_macro': make_scorer(f1_score, average='macro')
        }
        
        # Initialize k-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Store cross-validation results
        self.cv_results = {}
        
        # Perform cross-validation for each metric
        for metric_name, scorer in scoring.items():
            start_time = time.time()
            try:
                scores = cross_val_score(
                    self.model, self.classifier.X_train_scaled, self.classifier.y_train, 
                    cv=kf, scoring=scorer
                )
            except Exception as e:
                print(f"  ! Error in cross-validation for {metric_name}: {str(e)}")
                scores = np.array([0])
                
            cv_time = time.time() - start_time
            
            self.cv_results[metric_name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'time': cv_time
            }
            
            print(f"  ✓ {metric_name}: {scores.mean():.4f} ± {scores.std():.4f} (took {cv_time:.4f}s)")
        
        # Visualize cross-validation results
        self._visualize_cv_results()
        
        return self
    
    def _visualize_cv_results(self):
        """Create visualizations of cross-validation results."""
        print("\n  Creating cross-validation visualizations...")
        
        if self.cv_results is None:
            print("  ! No cross-validation results available for visualization.")
            return self
        
        # Bar plot of cross-validation results
        plt.figure(figsize=(10, 6))
        metrics = list(self.cv_results.keys())
        means = [self.cv_results[m]['mean'] for m in metrics]
        stds = [self.cv_results[m]['std'] for m in metrics]
        
        # Plot bars with error bars
        bars = plt.bar(metrics, means, yerr=stds, alpha=0.8, capsize=10)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('Cross-Validation Results')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('program2_cross_validation_results.png')
        plt.close()
        
        # Learning curve visualization
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, self.X, self.y, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5, scoring='accuracy', n_jobs=-1
            )
            
            # Calculate mean and std for training and test scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                            alpha=0.1, color='blue')
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                            alpha=0.1, color='orange')
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
            plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
            
            plt.title('Learning Curve')
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('program2_learning_curve.png')
            plt.close()
            
            print("  ✓ Cross-validation visualizations created: program2_cross_validation_results.png, program2_learning_curve.png")
        except Exception as e:
            print(f"  ! Error creating learning curve: {str(e)}")
            print("  ✓ Cross-validation bar chart created: program2_cross_validation_results.png")
        
        return self
    
    def analyze_model_explainability(self):
        """Analyze model explainability using SHAP values."""
        print("\nStep 2: Analyzing Model Explainability with SHAP...")
        
        if not SHAP_AVAILABLE:
            print("  ! SHAP library not available. Skipping explainability analysis.")
            self.shap_values = None
            self.explainer = None
            return self
        
        if self.model is None or self.X_test_scaled is None:
            print("  ! Cannot perform SHAP analysis: model or test data not available.")
            return self
        
        try:
            # Create a background dataset for SHAP
            # For KNN, we'll use a subset of the training data
            background_data = shap.kmeans(self.classifier.X_train_scaled, 10)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background_data
            )
            
            # Calculate SHAP values for test set (using a smaller subset for speed)
            sample_size = min(20, len(self.X_test_scaled))  # Use at most 20 samples for visualization
            self.sample_indices = np.random.choice(len(self.X_test_scaled), sample_size, replace=False)
            
            start_time = time.time()
            self.shap_values = self.explainer.shap_values(
                self.X_test_scaled[self.sample_indices], nsamples=100
            )
            shap_time = time.time() - start_time
            
            print(f"  ✓ SHAP values calculated (took {shap_time:.2f}s)")
            
            # Create SHAP visualizations
            self._visualize_shap_values()
        except Exception as e:
            print(f"  ! Error in SHAP analysis: {str(e)}")
            self.shap_values = None
            self.explainer = None
        
        return self
    
    def _visualize_shap_values(self):
        """Create visualizations of SHAP values with robust error protection."""
        if not SHAP_AVAILABLE or self.shap_values is None or self.explainer is None:
            print("  ! SHAP values or explainer not available. Skipping SHAP visualizations.")
            return self
            
        print("\n  Creating SHAP visualizations...")
    
        try:
            # CRITICAL DIAGNOSTIC: Verify SHAP values structure before proceeding
            # For KernelExplainer with classification, shap_values should be a list with one array per class
            if not isinstance(self.shap_values, list):
                print(f"  ! Unexpected SHAP values structure: {type(self.shap_values)}. Expected list of arrays.")
                return self
            
            # STRUCTURAL VALIDATION: Create guaranteed-safe indices no matter what
            safe_sample_count = min(20, len(self.X_test_scaled))
            safe_indices = np.arange(safe_sample_count, dtype=int)  # Explicitly create integer indices
        
            # Extract needed data with protective validation
            sample_data = self.X_test_scaled[safe_indices]
        
            # ISOLATED VISUALIZATION BLOCK 1: Feature Importance Summary
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    self.shap_values,  # Use full structure for summary plot
                    sample_data,
                    feature_names=self.feature_names,
                    class_names=self.target_names,
                    plot_type="bar",
                    show=False
                )
                plt.tight_layout()
                plt.savefig('program2_shap_feature_importance.png')
                plt.close()
                print("  ✓ Generated feature importance visualization")
            except Exception as e:
                print(f"  ! Error creating SHAP feature importance plot: {str(e)}")
        
            # ISOLATED VISUALIZATION BLOCK 2: Class-specific distributions 
            try:
                plt.figure(figsize=(10, 8))
                for i, class_name in enumerate(self.target_names):
                    if i >= len(self.shap_values):  # Defensive bound check
                        print(f"  ! Class index {i} exceeds SHAP values length {len(self.shap_values)}")
                        continue
                    
                    plt.subplot(len(self.target_names), 1, i+1)
                    shap.summary_plot(
                        self.shap_values[i],  # Class-specific SHAP values
                        sample_data,
                        feature_names=self.feature_names,
                        plot_type="dot",
                        show=False
                    )
                    plt.title(f'SHAP Values for Class: {class_name}')
                
                plt.tight_layout()
                plt.savefig('program2_shap_summary_plot.png')
                plt.close()
                print("  ✓ Generated class distribution visualization")
            except Exception as e:
                print(f"  ! Error creating SHAP distribution plot: {str(e)}")
        
            # ISOLATED VISUALIZATION BLOCK 3: Decision plots with additional protection
            try:
                plt.figure(figsize=(10, 8))
                # Use even smaller sample for decision plots (more complex)
                valid_sample_count = min(5, safe_sample_count)
            
                # CRITICAL: Verify explainer expected_value structure
                if not hasattr(self.explainer, 'expected_value') or self.explainer.expected_value is None:
                    print("  ! Explainer missing expected_value attribute")
                    raise ValueError("Explainer missing expected_value")
                
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    expected_values = self.explainer.expected_value
                else:
                    # Handle case where expected_value is a scalar
                    expected_values = [self.explainer.expected_value] * len(self.target_names)
            
                for i, class_idx in enumerate(range(min(len(self.target_names), len(self.shap_values)))):
                    if class_idx >= len(expected_values):
                        print(f"  ! Class index {class_idx} exceeds expected_value length {len(expected_values)}")
                        continue
                    
                    plt.subplot(min(len(self.target_names), len(self.shap_values)), 1, i+1)
                
                    # Use safe slicing with explicit validation
                    class_shap_values = self.shap_values[class_idx][:valid_sample_count]
                    class_sample_data = sample_data[:valid_sample_count]
                
                    shap.decision_plot(
                        expected_values[class_idx],
                        class_shap_values, 
                        class_sample_data,
                        feature_names=self.feature_names,
                        show=False
                    )
                    plt.title(f'Decision Plot for Class: {self.target_names[class_idx]}')
                
                plt.tight_layout()
                plt.savefig('program2_shap_decision_plot.png')
                plt.close()
                print("  ✓ Generated decision plot visualization")
            except Exception as e:
                print(f"  ! Error creating SHAP decision plot: {str(e)}")
        
            print("  ✓ SHAP visualizations completed with available data")
            return self
        except Exception as e:
            print(f"  ! Error creating SHAP visualizations: {str(e)}")
            return self
    
    def analyze_performance_and_resources(self, n_iterations=100):
        """Analyze model performance and resource usage."""
        print("\nStep 3: Analyzing Performance and Resource Usage...")
        
        if self.model is None or self.X_test_scaled is None:
            print("  ! Cannot analyze performance: model or test data not available.")
            self.prediction_times = []
            self.performance_metrics = {
                'avg_prediction_time': 0,
                'min_prediction_time': 0,
                'max_prediction_time': 0,
                'std_prediction_time': 0,
                'memory_before_mb': 0,
                'memory_after_mb': 0,
                'memory_usage_mb': 0
            }
            return self
        
        try:
            # Record memory usage before prediction
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure prediction time over multiple iterations
            self.prediction_times = []
            for _ in range(n_iterations):
                start_time = time.time()
                self.model.predict(self.X_test_scaled)
                end_time = time.time()
                self.prediction_times.append(end_time - start_time)
            
            # Record memory usage after prediction
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate and store metrics
            self.performance_metrics = {
                'avg_prediction_time': np.mean(self.prediction_times),
                'min_prediction_time': np.min(self.prediction_times),
                'max_prediction_time': np.max(self.prediction_times),
                'std_prediction_time': np.std(self.prediction_times),
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_usage_mb': memory_after - memory_before
            }
            
            # Print performance metrics
            print(f"  ✓ Average prediction time: {self.performance_metrics['avg_prediction_time']*1000:.2f} ms")
            print(f"  ✓ Min/Max prediction time: {self.performance_metrics['min_prediction_time']*1000:.2f} / "
                f"{self.performance_metrics['max_prediction_time']*1000:.2f} ms")
            print(f"  ✓ Memory usage: {self.performance_metrics['memory_usage_mb']:.2f} MB")
            
            # Visualize performance metrics
            self._visualize_performance()
        except Exception as e:
            print(f"  ! Error in performance analysis: {str(e)}")
            self.prediction_times = []
            self.performance_metrics = {
                'avg_prediction_time': 0,
                'min_prediction_time': 0,
                'max_prediction_time': 0,
                'std_prediction_time': 0,
                'memory_before_mb': 0,
                'memory_after_mb': 0,
                'memory_usage_mb': 0
            }
        
        return self
    
    def _visualize_performance(self):
        """Create visualizations of performance metrics."""
        if not self.prediction_times:
            print("  ! No prediction time data available for visualization.")
            return self
            
        print("\n  Creating performance visualizations...")
        
        try:
            # 1. Prediction time distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.prediction_times, bins=20, alpha=0.7, color='blue')
            plt.axvline(self.performance_metrics['avg_prediction_time'], color='red', 
                       linestyle='dashed', linewidth=2, label='Mean')
            
            plt.title('Distribution of Prediction Times')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('program2_prediction_time_distribution.png')
            plt.close()
            
            # 2. Memory usage
            plt.figure(figsize=(8, 6))
            labels = ['Before Prediction', 'After Prediction']
            sizes = [self.performance_metrics['memory_before_mb'], 
                    self.performance_metrics['memory_usage_mb']]
            colors = ['#66b3ff', '#ff9999']
            
            plt.bar(labels, sizes, color=colors)
            for i, v in enumerate(sizes):
                plt.text(i, v + 0.1, f"{v:.2f} MB", ha='center')
                
            plt.title('Memory Usage')
            plt.ylabel('Memory (MB)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('program2_memory_usage.png')
            plt.close()
            
            print("  ✓ Performance visualizations created: program2_prediction_time_distribution.png, program2_memory_usage.png")
        except Exception as e:
            print(f"  ! Error creating performance visualizations: {str(e)}")
        
        return self
    
    def assess_model_limitations(self):
        """Assess model limitations and potential issues."""
        print("\nStep 4: Assessing Model Limitations and Findings...")
        
        self.findings = {'strengths': [], 'limitations': []}
        
        # Check if we have the necessary data
        if self.cv_results is None or self.model is None:
            print("  ! Cannot assess model limitations: cross-validation results or model not available.")
            self.findings['limitations'].append("Insufficient data to perform thorough model assessment.")
            return self
        
        try:
            # Check cross-validation consistency
            cv_std = self.cv_results['accuracy']['std']
            if cv_std > 0.05:
                self.findings['limitations'].append(
                    f"Model performance varies across folds (std={cv_std:.4f}), " +
                    "suggesting potential instability."
                )
            else:
                self.findings['strengths'].append(
                    f"Model performance is consistent across folds (std={cv_std:.4f})."
                )
            
            # Check for class imbalance effects
            if 'recall_macro' in self.cv_results and 'accuracy' in self.cv_results:
                recall = self.cv_results['recall_macro']['mean']
                accuracy = self.cv_results['accuracy']['mean']
                if recall < accuracy - 0.05:
                    self.findings['limitations'].append(
                        f"Recall ({recall:.4f}) is significantly lower than " +
                        f"accuracy ({accuracy:.4f}), suggesting class imbalance issues."
                    )
                else:
                    self.findings['strengths'].append(
                        f"Recall ({recall:.4f}) is close to accuracy ({accuracy:.4f}), " +
                        "suggesting good performance across classes."
                    )
            
            # Check performance speed
            if self.performance_metrics:
                avg_time = self.performance_metrics['avg_prediction_time'] * 1000  # Convert to ms
                if avg_time > 10:  # Arbitrary threshold for this small dataset
                    self.findings['limitations'].append(
                        f"Average prediction time ({avg_time:.2f} ms) is relatively high " +
                        "for this small dataset."
                    )
                else:
                    self.findings['strengths'].append(
                        f"Good prediction speed ({avg_time:.2f} ms)."
                    )
                
                # Check memory usage
                memory_usage = self.performance_metrics['memory_usage_mb']
                if memory_usage > 10:  # Arbitrary threshold for this small dataset
                    self.findings['limitations'].append(
                        f"Memory usage ({memory_usage:.2f} MB) is relatively high " +
                        "for this small dataset."
                    )
                else:
                    self.findings['strengths'].append(
                        f"Efficient memory usage ({memory_usage:.2f} MB)."
                    )
            
            # Check for explainability concerns from SHAP
            if self.shap_values is not None:
                # Check if any feature has overwhelmingly high importance
                feature_importance = np.abs(self.shap_values[0]).mean(0)
                max_importance = feature_importance.max()
                total_importance = feature_importance.sum()
                if max_importance / total_importance > 0.5:
                    self.findings['limitations'].append(
                        "Model heavily relies on a single feature, " +
                        "which might indicate overfitting or fragility."
                    )
                else:
                    self.findings['strengths'].append(
                        "Model uses a balanced set of features for predictions."
                    )
            
            # Print findings
            print("\nModel Strengths:")
            for strength in self.findings['strengths']:
                print(f"  ✓ {strength}")
            
            print("\nPotential Limitations:")
            for limitation in self.findings['limitations']:
                print(f"  ! {limitation}")
        except Exception as e:
            print(f"  ! Error assessing model limitations: {str(e)}")
            self.findings['limitations'].append(f"Error during assessment: {str(e)}")
        
        return self
    
    def summarize_findings(self):
        """Summarize all findings in a comprehensive report."""
        print("\nSummary of Findings:")
        print("===================")
        
        try:
            # Print model performance summary
            print("\n1. Model Performance:")
            original_accuracy = getattr(self.classifier, 'accuracy', None)
            if original_accuracy is not None:
                print(f"   - Original accuracy: {original_accuracy:.4f}")
            
            if self.cv_results and 'accuracy' in self.cv_results:
                print(f"   - Cross-validation accuracy: {self.cv_results['accuracy']['mean']:.4f} " +
                    f"± {self.cv_results['accuracy']['std']:.4f}")
                
                if original_accuracy is not None and self.cv_results['accuracy']['mean'] < original_accuracy - 0.05:
                    print("   ⚠ Cross-validation accuracy is lower than the original test accuracy, " +
                        "suggesting potential overfitting.")
                else:
                    print("   ✓ Cross-validation confirms the model's performance.")
            
            # Print explainability insights
            print("\n2. Model Explainability (SHAP Analysis):")
            if self.shap_values is not None:
                # Identify most important features
                feature_importance = np.abs(self.shap_values[0]).mean(0)
                sorted_idx = np.argsort(feature_importance)[::-1]
                print("   Top important features:")
                for i in range(min(len(sorted_idx), 4)):
                    idx = sorted_idx[i]
                    print(f"   - {self.feature_names[idx]}: {feature_importance[idx]:.4f}")
            else:
                print("   - SHAP analysis not available.")
            
            # Print performance and resource usage
            print("\n3. Performance and Resource Usage:")
            if self.performance_metrics:
                print(f"   - Average prediction time: {self.performance_metrics['avg_prediction_time']*1000:.2f} ms")
                print(f"   - Memory usage: {self.performance_metrics['memory_usage_mb']:.2f} MB")
            else:
                print("   - Performance metrics not available.")
            
            # Print strengths and limitations
            print("\n4. Model Strengths:")
            for strength in self.findings['strengths']:
                print(f"   ✓ {strength}")
            
            print("\n5. Model Limitations and Improvement Areas:")
            for limitation in self.findings['limitations']:
                print(f"   ! {limitation}")
            
            # Final assessment
            print("\nFinal Assessment:")
            if len(self.findings['limitations']) > len(self.findings['strengths']):
                print("The model has several limitations that should be addressed before deployment.")
            elif len(self.findings['limitations']) > 0:
                print("The model performs reasonably well but has some limitations to consider.")
            else:
                print("The model performs well with no significant limitations identified.")
        except Exception as e:
            print(f"Error summarizing findings: {str(e)}")
            print("The model assessment is incomplete due to errors during analysis.")
        
        return self
    
    def save_results(self, filename='model_tester_results.pkl'):
        """Save the model tester results to a file."""
        print(f"\nSaving test results to {filename}...")
        
        try:
            # Store essential results
            results = {
                'cv_results': self.cv_results,
                'findings': self.findings,
                'performance_metrics': self.performance_metrics,
            }
            
            # Store SHAP values only if they're not too large
            if self.shap_values is not None:
                try:
                    total_size = sum(array.nbytes for array in self.shap_values)
                    if total_size < 100 * 1024 * 1024:  # Less than 100MB
                        results['shap_values'] = self.shap_values
                    else:
                        print("  ! SHAP values are too large to save, skipping.")
                except Exception:
                    print("  ! Error measuring SHAP values size, skipping.")
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"  ✓ Test results saved to {filename}")
        except Exception as e:
            print(f"  ! Error saving test results: {str(e)}")
        
        return self
    
    def load_results(self, filename='model_tester_results.pkl'):
        """Load test results from a file."""
        print(f"\nLoading test results from {filename}...")
        
        if not os.path.exists(filename):
            print(f"  ! Results file {filename} not found.")
            return self
        
        try:
            with open(filename, 'rb') as f:
                results = pickle.load(f)
            
            # Load results into the object
            self.cv_results = results.get('cv_results')
            self.findings = results.get('findings', {'strengths': [], 'limitations': []})
            self.performance_metrics = results.get('performance_metrics', {})
            self.shap_values = results.get('shap_values')
            
            print(f"  ✓ Test results loaded from {filename}")
        except Exception as e:
            print(f"  ! Error loading test results: {str(e)}")
        
        return self


def main(load_model=True, load_results=False):
    """Main function to run the model testing process."""
    # Create an instance of the classifier
    iris_clf = IrisClassifier()
    
    if load_model:
        # Try to load a pre-trained model
        try:
            iris_clf.load_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}. Training a new model.")
            # If loading fails, create and train a new classifier
            try:
                iris_clf.load_data() \
                       .prepare_data() \
                       .train_model() \
                       .evaluate_model() \
                       .save_model()
            except Exception as e:
                print(f"Error training model: {str(e)}. Proceeding with limited functionality.")
    else:
        # Create and train a new classifier
        try:
            iris_clf.load_data() \
                   .prepare_data() \
                   .train_model() \
                   .evaluate_model() \
                   .save_model()
        except Exception as e:
            print(f"Error training model: {str(e)}. Proceeding with limited functionality.")
    
    # Create the model tester
    tester = ModelTester(iris_clf)
    
    if load_results:
        # Try to load previous test results
        try:
            tester.load_results()
            # Run any missing analyses
            if tester.cv_results is None:
                tester.perform_cross_validation()
            if tester.shap_values is None and SHAP_AVAILABLE:
                tester.analyze_model_explainability()
            if not tester.performance_metrics:
                tester.analyze_performance_and_resources()
            if not tester.findings['strengths'] and not tester.findings['limitations']:
                tester.assess_model_limitations()
        except Exception as e:
            print(f"Error loading test results: {str(e)}. Running full analysis.")
            load_results = False
    
    if not load_results:
        # Run the full testing pipeline
        tester.perform_cross_validation() \
              .analyze_model_explainability() \
              .analyze_performance_and_resources() \
              .assess_model_limitations()
    
    # Always summarize findings and save results
    tester.summarize_findings().save_results()
    
    print("\nModel testing completed successfully!")
    
    # Return the tester for further use
    return tester


if __name__ == "__main__":
    # Set load_model=True to load a pre-trained model, load_results=True to load previous test results
    model_tester = main(load_model=True, load_results=False)

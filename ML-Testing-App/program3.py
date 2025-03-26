"""
Robust Testing of Iris Classification Model with Outliers
--------------------------------------------------------
Purpose: Extend testing by introducing outliers and evaluating model robustness
Author: ML Testing Team
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import time
import psutil
import os
import warnings
import copy
import pickle
from matplotlib.patches import Patch

# Try to import from previous programs, with fallbacks
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

try:
    from program2 import ModelTester
except ImportError:
    warnings.warn("Could not import ModelTester from program2. Using minimal compatibility class.")
    
    class ModelTester:
        """Minimal compatibility class for ModelTester when program2 isn't available."""
        def __init__(self, classifier):
            self.classifier = classifier
            self.X = getattr(classifier, 'X', None)
            self.y = getattr(classifier, 'y', None)
            self.model = getattr(classifier, 'model', None)
            self.feature_names = getattr(classifier, 'feature_names', None)
            self.target_names = getattr(classifier, 'target_names', None)
            
            # Initialize minimal attributes needed by Program 3
            self.cv_results = {'accuracy': {'mean': 0, 'std': 0}}
            self.findings = {'limitations': [], 'strengths': []}
            self.shap_values = None
            
        def load_results(self, filename='model_tester_results.pkl'):
            """Load tester results from a file."""
            print(f"\nLoading test results from {filename}...")
            
            if not os.path.exists(filename):
                print(f"  ! Results file {filename} not found.")
                return self
            
            try:
                with open(filename, 'rb') as f:
                    results = pickle.load(f)
                
                # Load results into the object
                self.cv_results = results.get('cv_results', {'accuracy': {'mean': 0, 'std': 0}})
                self.findings = results.get('findings', {'limitations': [], 'strengths': []})
                self.shap_values = results.get('shap_values')
                
                print(f"  ✓ Test results loaded from {filename}")
            except Exception as e:
                print(f"  ! Error loading test results: {str(e)}")
            
            return self
        
        def perform_cross_validation(self, cv=5):
            """Stub method for backward compatibility."""
            print("Warning: Using stubbed cross-validation in compatibility ModelTester")
            return self

# Set random seed for reproducibility
np.random.seed(42)


class RobustnessTester:
    """A class to test model robustness against outliers and edge cases."""
    
    def __init__(self, classifier, tester=None):
        """Initialize with a trained classifier and optional tester from Program 2."""
        self.classifier = classifier
        self.tester = tester

            
        # Get attributes from classifier, with fallbacks
        self.X = getattr(classifier, 'X', None)
        self.y = getattr(classifier, 'y', None)
        self.X_test = getattr(classifier, 'X_test', None)
        self.X_test_scaled = getattr(classifier, 'X_test_scaled', None)
        self.y_test = getattr(classifier, 'y_test', None)
        self.model = getattr(classifier, 'model', None)
        self.feature_names = getattr(classifier, 'feature_names', None)
        self.target_names = getattr(classifier, 'target_names', None)
        self.scaler = getattr(classifier, 'scaler', None)

        # ADD THIS CODE BLOCK after attribute assignment
        # Ensure training data is available even if main dataset isn't
        if self.X is None and hasattr(classifier, 'X_train') and classifier.X_train is not None:
            self.X = classifier.X_train
        
        if self.y is None and hasattr(classifier, 'y_train') and classifier.y_train is not None:
            self.y = classifier.y_train
    
        # Initialize with defaults for critical attributes
        self.X_with_outliers_scaled = None
        
        # Check for missing essential attributes
        missing_attrs = []
        for attr in ['X', 'y', 'X_test', 'y_test', 'model', 'feature_names', 'target_names', 'scaler']:
            if getattr(self, attr) is None:
                missing_attrs.append(attr)
        
        if missing_attrs:
            warnings.warn(f"Missing essential attributes: {', '.join(missing_attrs)}. " + 
                        "Some functionality may be limited.")
        
        # Initialize attributes for outlier testing
        self.X_with_outliers = None
        self.y_with_outliers = None
        self.outlier_indices = None
        self.existing_outliers = None
        self.original_performance = None
        self.outlier_performance = None
        self.outlier_analysis = None
        self.cv_results = None
        self.inadequacy_findings = []
        
    def identify_outliers(self, contamination=0.05):
        """Identify existing outliers in the dataset using LOF."""
        print("\nStep 1: Identifying Existing Outliers...")
        
        if self.X is None or self.y is None:
            print("  ! Cannot identify outliers: dataset not available.")
            self.existing_outliers = {
                'indices': np.array([]),
                'count': 0,
                'percentage': 0
            }
            return self
        
        try:
            # Use Local Outlier Factor to find outliers
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            y_pred = lof.fit_predict(self.X)
            outlier_mask = y_pred == -1
            
            # Store outlier information
            self.existing_outliers = {
                'indices': np.where(outlier_mask)[0],
                'count': sum(outlier_mask),
                'percentage': sum(outlier_mask) / len(self.X) * 100
            }
            
            print(f"  ✓ Found {self.existing_outliers['count']} outliers " + 
                f"({self.existing_outliers['percentage']:.2f}% of the dataset)")
            
            # Visualize existing outliers
            self._visualize_existing_outliers(outlier_mask)
        except Exception as e:
            print(f"  ! Error identifying outliers: {str(e)}")
            self.existing_outliers = {
                'indices': np.array([]),
                'count': 0,
                'percentage': 0
            }
        
        return self
    
    def _visualize_existing_outliers(self, outlier_mask):
        """Visualize the existing outliers in the dataset."""
        if self.X is None or self.feature_names is None or self.target_names is None:
            print("  ! Cannot visualize outliers: required attributes missing.")
            return self
            
        print("\n  Creating outlier visualizations...")
        
        try:
            # Create a DataFrame for plotting
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['species'] = [self.target_names[i] for i in self.y]
            df['is_outlier'] = outlier_mask
            
            # Feature pairs to visualize
            feature_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
            
            # Create pair plots with outliers highlighted
            plt.figure(figsize=(16, 12))
            for i, (f1, f2) in enumerate(feature_pairs):
                plt.subplot(2, 2, i+1)
                
                # Plot normal points
                for species in self.target_names:
                    species_mask = (df['species'] == species) & (~df['is_outlier'])
                    plt.scatter(
                        df.loc[species_mask, self.feature_names[f1]],
                        df.loc[species_mask, self.feature_names[f2]],
                        label=f"{species} (normal)",
                        alpha=0.7
                    )
                
                # Plot outliers with distinct markers
                for species in self.target_names:
                    species_mask = (df['species'] == species) & (df['is_outlier'])
                    if species_mask.any():
                        plt.scatter(
                            df.loc[species_mask, self.feature_names[f1]],
                            df.loc[species_mask, self.feature_names[f2]],
                            marker='x', s=100,
                            label=f"{species} (outlier)",
                            alpha=0.9
                        )
                
                plt.xlabel(self.feature_names[f1])
                plt.ylabel(self.feature_names[f2])
                plt.title(f'Outliers in {self.feature_names[f1]} vs {self.feature_names[f2]}')
                
                # Add legend for the first subplot only
                if i == 0:
                    plt.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig('program3_existing_outliers.png')
            plt.close()
            
            # Create boxplots to identify outliers
            plt.figure(figsize=(12, 8))
            for i, feature in enumerate(self.feature_names):
                plt.subplot(2, 2, i+1)
                
                # Create boxplot
                sns.boxplot(x='species', y=feature, data=df)
                
                # Overlay outliers
                outlier_data = df[df['is_outlier']]
                if not outlier_data.empty:
                    plt.scatter(
                        x=outlier_data['species'].map(lambda x: list(self.target_names).index(x)),
                        y=outlier_data[feature],
                        color='red', marker='x', s=100
                    )
                    
                plt.title(f'Outliers in {feature}')
                
            plt.tight_layout()
            plt.savefig('program3_outliers_boxplot.png')
            plt.close()
            
            print("  ✓ Outlier visualizations created: program3_existing_outliers.png, program3_outliers_boxplot.png")
        except Exception as e:
            print(f"  ! Error creating outlier visualizations: {str(e)}")
        
        return self
    
    def generate_artificial_outliers(self, n_outliers=10, outlier_scale=3.0):
        """Generate artificial outliers by modifying existing data points."""
        print("\nStep 2: Generating Artificial Outliers...")
        
        if self.X_test is None or self.y_test is None or self.feature_names is None:
            print("  ! Cannot generate outliers: test data not available.")
            self.X_with_outliers = np.copy(self.X_test) if self.X_test is not None else None
            self.y_with_outliers = np.copy(self.y_test) if self.y_test is not None else None
            self.outlier_indices = np.array([])
            return self
        
        try:
            # Create copies of test data
            self.X_with_outliers = np.copy(self.X_test)
            self.y_with_outliers = np.copy(self.y_test)
            
            # Adjust n_outliers if there's not enough test data
            n_outliers = min(n_outliers, len(self.X_test) // 3)
            
            # Calculate feature means and standard deviations per class
            feature_stats = {}
            for class_idx in range(len(self.target_names)):
                class_mask = self.y == class_idx
                if sum(class_mask) > 0:  # Only if we have samples for this class
                    feature_stats[class_idx] = {
                        'mean': np.mean(self.X[class_mask], axis=0),
                        'std': np.std(self.X[class_mask], axis=0)
                    }
                else:
                    # If no samples, use overall stats
                    feature_stats[class_idx] = {
                        'mean': np.mean(self.X, axis=0),
                        'std': np.std(self.X, axis=0)
                    }
            
            # Select random samples to convert to outliers
            outlier_indices = np.random.choice(
                len(self.X_with_outliers), size=n_outliers, replace=False
            )
            self.outlier_indices = outlier_indices
            
            # Create three types of outliers
            outlier_types = ['boundary', 'extreme', 'mixed']
            type_assignments = np.random.choice(outlier_types, size=n_outliers)
            
            # Transform the selected samples into outliers
            for i, idx in enumerate(outlier_indices):
                class_idx = self.y_with_outliers[idx]
                outlier_type = type_assignments[i]
                
                if outlier_type == 'boundary':
                    # Create boundary outliers (between classes)
                    target_class = (class_idx + 1) % len(self.target_names)
                    mix_ratio = np.random.uniform(0.3, 0.7)
                    self.X_with_outliers[idx] = (
                        mix_ratio * feature_stats[class_idx]['mean'] + 
                        (1 - mix_ratio) * feature_stats[target_class]['mean']
                    )
                    
                elif outlier_type == 'extreme':
                    # Create extreme outliers (far from all classes)
                    direction = np.random.choice([-1, 1], size=len(self.feature_names))
                    self.X_with_outliers[idx] = (
                        feature_stats[class_idx]['mean'] + 
                        direction * outlier_scale * feature_stats[class_idx]['std']
                    )
                    
                else:  # 'mixed'
                    # Create noise in specific dimensions
                    noise_dims = np.random.choice([True, False], size=len(self.feature_names))
                    if sum(noise_dims) > 0:  # Ensure at least one dimension has noise
                        self.X_with_outliers[idx, noise_dims] = (
                            feature_stats[class_idx]['mean'][noise_dims] + 
                            np.random.randn(sum(noise_dims)) * outlier_scale * 
                            feature_stats[class_idx]['std'][noise_dims]
                        )
            
            # Scale the data
            if self.scaler is not None and self.X_with_outliers is not None:
                self.X_with_outliers_scaled = self.scaler.transform(self.X_with_outliers)
            else:
                # If no scaler available, use StandardScaler
                #temp_scaler = StandardScaler()
                self.X_with_outliers_scaled = np.array([]) if self.X_with_outliers is None else np.copy(self.X_with_outliers)
            
            print(f"  ✓ Generated {n_outliers} artificial outliers of types: {', '.join(set(type_assignments))}")
            
            # Visualize the artificial outliers
            self._visualize_artificial_outliers()
        except Exception as e:
            print(f"  ! Error generating outliers: {str(e)}")
            self.X_with_outliers_scaled = np.array([])
            #if self.X_test is not None and self.y_test is not None:
                #self.X_with_outliers = np.copy(self.X_test)
                #self.y_with_outliers = np.copy(self.y_test)
                #self.outlier_indices = np.array([])
        
        return self
    
    def _visualize_artificial_outliers(self):
        """Visualize the artificial outliers."""
        if (self.X_test is None or self.X_with_outliers is None or 
            self.feature_names is None or self.outlier_indices is None or
            len(self.outlier_indices) == 0):
            print("  ! Cannot visualize artificial outliers: required data missing.")
            return self
            
        print("\n  Creating artificial outlier visualizations...")
        
        try:
            # Create DataFrames for plotting
            df_orig = pd.DataFrame(self.X_test, columns=self.feature_names)
            df_orig['species'] = [self.target_names[i] for i in self.y_test]
            df_orig['is_outlier'] = False
            
            df_out = pd.DataFrame(self.X_with_outliers, columns=self.feature_names)
            df_out['species'] = [self.target_names[i] for i in self.y_with_outliers]
            df_out['is_outlier'] = False
            df_out.loc[self.outlier_indices, 'is_outlier'] = True
            
            # Feature pairs to visualize
            feature_pairs = [(0, 1), (2, 3)]
            
            # Create plot comparing original data with outlier-injected data
            plt.figure(figsize=(14, 10))
            for i, (f1, f2) in enumerate(feature_pairs):
                # Original data subplot
                plt.subplot(2, 2, i*2+1)
                for species in self.target_names:
                    species_mask = df_orig['species'] == species
                    plt.scatter(
                        df_orig.loc[species_mask, self.feature_names[f1]],
                        df_orig.loc[species_mask, self.feature_names[f2]],
                        label=species,
                        alpha=0.7
                    )
                    
                plt.xlabel(self.feature_names[f1])
                plt.ylabel(self.feature_names[f2])
                plt.title(f'Original Test Data')
                if i == 0:
                    plt.legend(loc='best')
                
                # Outlier-injected data subplot
                plt.subplot(2, 2, i*2+2)
                # Plot normal points
                for species in self.target_names:
                    species_mask = (df_out['species'] == species) & (~df_out['is_outlier'])
                    plt.scatter(
                        df_out.loc[species_mask, self.feature_names[f1]],
                        df_out.loc[species_mask, self.feature_names[f2]],
                        label=f"{species} (normal)",
                        alpha=0.7
                    )
                
                # Highlight outliers
                plt.scatter(
                    df_out.loc[df_out['is_outlier'], self.feature_names[f1]],
                    df_out.loc[df_out['is_outlier'], self.feature_names[f2]],
                    marker='x', s=100, color='red',
                    label='Artificial outliers',
                    alpha=0.9
                )
                    
                plt.xlabel(self.feature_names[f1])
                plt.ylabel(self.feature_names[f2])
                plt.title(f'Test Data with Artificial Outliers')
                if i == 0:
                    plt.legend(loc='best')
            
            plt.tight_layout()
            plt.savefig('program3_artificial_outliers.png')
            plt.close()
            
            print("  ✓ Artificial outlier visualization created: program3_artificial_outliers.png")
        except Exception as e:
            print(f"  ! Error creating artificial outlier visualizations: {str(e)}")
        
        return self
    
    def evaluate_model_robustness(self):
        """Evaluate how the model performs with and without outliers."""
        print("\nStep 3: Evaluating Model Robustness to Outliers...")

        # Ensure X_with_outliers_scaled exists before checking it
        if not hasattr(self, 'X_with_outliers_scaled'):
            self.X_with_outliers_scaled = None
        
        if (self.model is None or self.X_test_scaled is None or 
            self.X_with_outliers_scaled is None or self.y_test is None or 
            self.y_with_outliers is None):
            print("  ! Cannot evaluate robustness: model or test data not available.")
            self.original_performance = {}
            self.outlier_performance = {}
            self.outlier_analysis = []
            return self
        
        try:
            # Evaluate performance on original test data
            start_time = time.time()
            y_pred_orig = self.model.predict(self.X_test_scaled)
            orig_time = time.time() - start_time
            
            # Calculate original metrics
            orig_accuracy = accuracy_score(self.y_test, y_pred_orig)
            orig_recall = recall_score(self.y_test, y_pred_orig, average='macro')
            orig_f1 = f1_score(self.y_test, y_pred_orig, average='macro')
            orig_conf_matrix = confusion_matrix(self.y_test, y_pred_orig)
            
            # Evaluate performance on test data with outliers
            start_time = time.time()
            y_pred_out = self.model.predict(self.X_with_outliers_scaled)
            out_time = time.time() - start_time
            
            # Calculate metrics with outliers
            out_accuracy = accuracy_score(self.y_with_outliers, y_pred_out)
            out_recall = recall_score(self.y_with_outliers, y_pred_out, average='macro')
            out_f1 = f1_score(self.y_with_outliers, y_pred_out, average='macro')
            out_conf_matrix = confusion_matrix(self.y_with_outliers, y_pred_out)
            
            # Calculate impact of outliers
            accuracy_impact = (orig_accuracy - out_accuracy) / orig_accuracy * 100
            recall_impact = (orig_recall - out_recall) / orig_recall * 100
            f1_impact = (orig_f1 - out_f1) / orig_f1 * 100
            time_impact = (out_time - orig_time) / orig_time * 100
            
            # Store the results
            self.original_performance = {
                'accuracy': orig_accuracy,
                'recall': orig_recall,
                'f1': orig_f1,
                'conf_matrix': orig_conf_matrix,
                'prediction_time': orig_time
            }
            
            self.outlier_performance = {
                'accuracy': out_accuracy,
                'recall': out_recall,
                'f1': out_f1,
                'conf_matrix': out_conf_matrix,
                'prediction_time': out_time,
                'accuracy_impact': accuracy_impact,
                'recall_impact': recall_impact,
                'f1_impact': f1_impact,
                'time_impact': time_impact
            }
            
            # Analyze how specific outliers affect predictions
            self.outlier_analysis = self._analyze_outlier_impact(y_pred_out)
            
            # Print results
            print("\n  Performance comparison:")
            print(f"  Original accuracy: {orig_accuracy:.4f}, With outliers: {out_accuracy:.4f} " + 
                f"(Impact: {accuracy_impact:.2f}%)")
            print(f"  Original recall: {orig_recall:.4f}, With outliers: {out_recall:.4f} " + 
                f"(Impact: {recall_impact:.2f}%)")
            print(f"  Original F1: {orig_f1:.4f}, With outliers: {out_f1:.4f} " + 
                f"(Impact: {f1_impact:.2f}%)")
            print(f"  Original prediction time: {orig_time*1000:.2f} ms, " + 
                f"With outliers: {out_time*1000:.2f} ms (Impact: {time_impact:.2f}%)")
            
            # Visualize robustness results
            self._visualize_robustness_results()
        except Exception as e:
            print(f"  ! Error evaluating robustness: {str(e)}")
            self.original_performance = {}
            self.outlier_performance = {}
            self.outlier_analysis = []
        
        return self
    
    def _analyze_outlier_impact(self, y_pred_out):
        """Analyze how specific outliers affect predictions."""
        if self.outlier_indices is None or len(self.outlier_indices) == 0:
            return []
        
        outlier_results = []
        
        try:
            for i, idx in enumerate(self.outlier_indices):
                true_class = self.y_with_outliers[idx]
                pred_class = y_pred_out[idx]
                is_misclassified = true_class != pred_class
                
                # Calculate distance to original class centroid and predicted class centroid
                class_centroids = {}
                for class_idx in range(len(self.target_names)):
                    class_mask = self.y == class_idx
                    if sum(class_mask) > 0:
                        class_centroids[class_idx] = np.mean(self.X[class_mask], axis=0)
                    else:
                        class_centroids[class_idx] = np.mean(self.X, axis=0)
                
                dist_to_true = np.linalg.norm(self.X_with_outliers[idx] - class_centroids[true_class])
                dist_to_pred = np.linalg.norm(self.X_with_outliers[idx] - class_centroids[pred_class]) if is_misclassified else None
                
                outlier_results.append({
                    'index': idx,
                    'true_class': self.target_names[true_class],
                    'pred_class': self.target_names[pred_class],
                    'is_misclassified': is_misclassified,
                    'dist_to_true_centroid': dist_to_true,
                    'dist_to_pred_centroid': dist_to_pred if is_misclassified else None
                })
        except Exception as e:
            print(f"  ! Error analyzing outlier impact: {str(e)}")
        
        return outlier_results
    
    def _visualize_robustness_results(self):
        """Visualize the robustness evaluation results."""
        if not self.original_performance or not self.outlier_performance:
            print("  ! Cannot visualize robustness results: performance data not available.")
            return self
            
        print("\n  Creating robustness visualizations...")
        
        try:
            # 1. Metrics comparison bar chart
            plt.figure(figsize=(10, 6))
            metrics = ['accuracy', 'recall', 'f1']
            orig_scores = [self.original_performance[m] for m in metrics]
            out_scores = [self.outlier_performance[m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, orig_scores, width, label='Original')
            plt.bar(x + width/2, out_scores, width, label='With Outliers')
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Model Performance With and Without Outliers')
            plt.xticks(x, metrics)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add data labels
            for i, (o, w) in enumerate(zip(orig_scores, out_scores)):
                plt.text(i - width/2, o + 0.02, f"{o:.3f}", ha='center')
                plt.text(i + width/2, w + 0.02, f"{w:.3f}", ha='center')
                # Add impact percentage
                impact = (o - w) / o * 100
                plt.text(i, 0.1, f"Impact: {impact:.1f}%", ha='center', 
                        color='red' if impact > 0 else 'green')
            
            plt.tight_layout()
            plt.savefig('program3_robustness_metrics.png')
            plt.close()
            
            # 2. Confusion matrix comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original confusion matrix
            sns.heatmap(self.original_performance['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.target_names, yticklabels=self.target_names, ax=axes[0])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            axes[0].set_title('Original Confusion Matrix')
            
            # Outlier confusion matrix
            sns.heatmap(self.outlier_performance['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.target_names, yticklabels=self.target_names, ax=axes[1])
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            axes[1].set_title('Confusion Matrix with Outliers')
            
            plt.tight_layout()
            plt.savefig('program3_robustness_confusion.png')
            plt.close()
            
            # 3. Outlier impact visualization
            if self.outlier_analysis:
                misclassified = [o for o in self.outlier_analysis if o['is_misclassified']]
                correctly_classified = [o for o in self.outlier_analysis if not o['is_misclassified']]
                
                # Only create this visualization if we have outliers and valid data structures
                if (misclassified or correctly_classified) and self.X_test is not None and self.y_test is not None:
                    plt.figure(figsize=(10, 6))
                    
                    # Create scatter plot with class information
                    for i, class_name in enumerate(self.target_names):
                        # Original test points for this class
                        class_mask = np.where(self.y_test == i)[0] if isinstance(self.y_test, np.ndarray) else []
                        if len(class_mask) > 0:
                            # Select points matching this class
                        
                            plt.scatter(
                                self.X_test[class_mask, 0] if self.X_test.shape[1] > 0 else [],
                                self.X_test[class_mask, 1] if self.X_test.shape[1] > 1 else [],
                                alpha=0.5,
                                label=f"{class_name} (original)"
                        )
                    
                    # Plot misclassified outliers
                    for o in misclassified:
                        idx = o['index']
                        #true_class_id = self.target_names.index(o['true_class'])
                        true_class_id = next((i for i, name in enumerate(self.target_names) 
                     if name == o['true_class']), 0)
                        pred_class_id = self.target_names.index(o['pred_class'])
                        plt.scatter(
                            self.X_with_outliers[idx, 0],
                            self.X_with_outliers[idx, 1],
                            marker='x', s=100, color='red',
                            label=f"Misclassified outlier" if idx == misclassified[0]['index'] else ""
                        )
                        plt.annotate(
                            f"True: {o['true_class']}, Pred: {o['pred_class']}",
                            (self.X_with_outliers[idx, 0], self.X_with_outliers[idx, 1]),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                        )
                    
                    # Plot correctly classified outliers
                    for o in correctly_classified:
                        idx = o['index']
                        plt.scatter(
                            self.X_with_outliers[idx, 0],
                            self.X_with_outliers[idx, 1],
                            marker='o', s=100, edgecolors='green', facecolors='none',
                            label=f"Correctly classified outlier" if idx == correctly_classified[0]['index'] else ""
                        )
                    
                    plt.xlabel(self.feature_names[0])
                    plt.ylabel(self.feature_names[1])
                    plt.title('Impact of Outliers on Classification')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('program3_outlier_impact.png')
                    plt.close()
                    
                    print("  ✓ Robustness visualizations created: program3_robustness_metrics.png, " + 
                        "program3_robustness_confusion.png, program3_outlier_impact.png")
                else:
                    print("  ✓ Robustness visualizations created: program3_robustness_metrics.png, " + 
                        "program3_robustness_confusion.png")
            else:
                print("  ✓ Robustness visualizations created: program3_robustness_metrics.png, " + 
                    "program3_robustness_confusion.png")
        except Exception as e:
            print(f"  ! Error creating robustness visualizations: {str(e)}")
        
        return self
    
    def perform_cross_validation(self, cv=5):
        """Perform cross-validation with and without outliers."""
        print("\nStep 4: Performing Cross-Validation with Outliers...")
        
        if (self.model is None or self.X is None or self.y is None or 
            self.X_with_outliers is None or self.y_with_outliers is None):
            print("  ! Cannot perform cross-validation with outliers: required data missing.")
            self.cv_results = {
                'clean_scores': np.array([]),
                'outlier_scores': np.array([]),
                'clean_mean': 0,
                'clean_std': 0,
                'outlier_mean': 0,
                'outlier_std': 0,
                'impact_by_fold': np.array([]),
                'avg_impact': 0
            }
            return self
        
        try:
            # Create new data with outliers spread throughout
            X_combined = np.vstack([self.X, self.X_with_outliers])
            y_combined = np.concatenate([self.y, self.y_with_outliers])
            
            # Mark the outlier samples
            is_outlier = np.zeros(len(X_combined), dtype=bool)
            is_outlier[len(self.X):] = True
            
            # Create stratified folds that respect class distribution
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Store results for both scenarios
            orig_cv_scores = []
            out_cv_scores = []
            fold_outlier_impact = []
            
            print(f"  Running {cv}-fold cross-validation...")
            
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_combined, y_combined)):
                # Get training and test data for this fold
                X_train_fold, X_test_fold = X_combined[train_idx], X_combined[test_idx]
                y_train_fold, y_test_fold = y_combined[train_idx], y_combined[test_idx]
                
                # Identify outliers in test set
                test_outliers = is_outlier[test_idx]
                
                # Create clean test set (without outliers)
                if any(test_outliers):
                    X_test_clean = X_test_fold[~test_outliers]
                    y_test_clean = y_test_fold[~test_outliers]
                else:
                    X_test_clean = X_test_fold
                    y_test_clean = y_test_fold
                
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_test_scaled = scaler.transform(X_test_fold)
                X_test_clean_scaled = scaler.transform(X_test_clean)
                
                # Train a KNN model
                model = copy.deepcopy(self.model)
                model.fit(X_train_scaled, y_train_fold)
                
                # Evaluate on full test set (with outliers)
                y_pred_full = model.predict(X_test_scaled)
                full_acc = accuracy_score(y_test_fold, y_pred_full)
                out_cv_scores.append(full_acc)
                
                # Evaluate on clean test set (without outliers)
                y_pred_clean = model.predict(X_test_clean_scaled)
                clean_acc = accuracy_score(y_test_clean, y_pred_clean)
                orig_cv_scores.append(clean_acc)
                
                # Calculate impact
                if len(test_outliers) > 0 and any(test_outliers):
                    impact = (clean_acc - full_acc) / clean_acc * 100
                else:
                    impact = 0
                    
                fold_outlier_impact.append(impact)
                
                print(f"  Fold {fold+1}: Clean accuracy: {clean_acc:.4f}, " + 
                    f"With outliers: {full_acc:.4f}, Impact: {impact:.2f}%")
            
            # Calculate average results
            self.cv_results = {
                'clean_scores': np.array(orig_cv_scores),
                'outlier_scores': np.array(out_cv_scores),
                'clean_mean': np.mean(orig_cv_scores),
                'clean_std': np.std(orig_cv_scores),
                'outlier_mean': np.mean(out_cv_scores),
                'outlier_std': np.std(out_cv_scores),
                'impact_by_fold': np.array(fold_outlier_impact),
                'avg_impact': np.mean(fold_outlier_impact)
            }
            
            print(f"\n  Average CV score (clean): {self.cv_results['clean_mean']:.4f} " + 
                f"± {self.cv_results['clean_std']:.4f}")
            print(f"  Average CV score (with outliers): {self.cv_results['outlier_mean']:.4f} " + 
                f"± {self.cv_results['outlier_std']:.4f}")
            print(f"  Average impact: {self.cv_results['avg_impact']:.2f}%")
            
            # Visualize cross-validation results
            self._visualize_cv_results()
        except Exception as e:
            print(f"  ! Error in cross-validation with outliers: {str(e)}")
            self.cv_results = {
                'clean_scores': np.array([]),
                'outlier_scores': np.array([]),
                'clean_mean': 0,
                'clean_std': 0,
                'outlier_mean': 0,
                'outlier_std': 0,
                'impact_by_fold': np.array([]),
                'avg_impact': 0
            }
        
        return self
    
    def _visualize_cv_results(self):
        """Visualize the cross-validation results."""
        if (self.cv_results is None or 
            len(self.cv_results['clean_scores']) == 0 or 
            len(self.cv_results['outlier_scores']) == 0):
            print("  ! Cannot visualize cross-validation results: data not available.")
            return self
            
        print("\n  Creating cross-validation visualizations...")
        
        try:
            # CV scores comparison
            plt.figure(figsize=(10, 6))
            x = np.arange(len(self.cv_results['clean_scores']))
            width = 0.35
            
            plt.bar(x - width/2, self.cv_results['clean_scores'], width, label='Without Outliers')
            plt.bar(x + width/2, self.cv_results['outlier_scores'], width, label='With Outliers')
            
            plt.xlabel('Fold')
            plt.ylabel('Accuracy')
            plt.title('Cross-Validation Results With and Without Outliers')
            plt.xticks(x, [f'Fold {i+1}' for i in range(len(x))])
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add mean line
            plt.axhline(self.cv_results['clean_mean'], linestyle='--', color='blue', alpha=0.7)
            plt.axhline(self.cv_results['outlier_mean'], linestyle='--', color='orange', alpha=0.7)
            
            # Add data labels
            for i, (c, o) in enumerate(zip(self.cv_results['clean_scores'], 
                                           self.cv_results['outlier_scores'])):
                plt.text(i - width/2, c + 0.02, f"{c:.3f}", ha='center')
                plt.text(i + width/2, o + 0.02, f"{o:.3f}", ha='center')
                # Add impact
                plt.text(i, 0.1, f"{self.cv_results['impact_by_fold'][i]:.1f}%", ha='center', 
                        color='red' if self.cv_results['impact_by_fold'][i] > 0 else 'green')
            
            plt.tight_layout()
            plt.savefig('program3_cv_outlier_comparison.png')
            plt.close()
            
            print("  ✓ Cross-validation visualization created: program3_cv_outlier_comparison.png")
        except Exception as e:
            print(f"  ! Error creating cross-validation visualizations: {str(e)}")
        
        return self
    
    def detect_test_inadequacies(self):
        """Detect inadequacies in the previous testing approaches."""
        print("\nStep 5: Detecting Test Inadequacies...")
        
        # Initialize a list to store inadequacy findings
        self.inadequacy_findings = []
        
        # 1. Check if original tests used cross-validation
        has_cv = False
        if self.tester is not None:
            has_cv = hasattr(self.tester, 'cv_results') and self.tester.cv_results is not None
        
        if not has_cv:
            self.inadequacy_findings.append({
                'program': 'Program 1',
                'issue': 'No cross-validation',
                'description': 'The initial model evaluation did not use cross-validation, ' +
                             'which can lead to overfitting and overly optimistic performance estimates.',
                'recommendation': 'Always use cross-validation for more reliable model evaluation.'
            })
        
        # 2. Check for outlier handling in original tests
        has_outlier_analysis = False
        if self.tester is not None and hasattr(self.tester, 'findings'):
            has_outlier_analysis = any('outlier' in str(l).lower() 
                                      for l in self.tester.findings.get('limitations', []))
        
        if not has_outlier_analysis:
            self.inadequacy_findings.append({
                'program': 'Program 1 & 2',
                'issue': 'No outlier analysis',
                'description': 'The previous testing did not specifically examine how outliers ' +
                             'affect model performance.',
                'recommendation': 'Always include outlier detection and robustness testing, ' +
                                'as we did in this program.'
            })
        
        # 3. Check if model parameters were optimized
        has_hyperparameter_opt = hasattr(self.classifier, 'best_params')
        
        if not has_hyperparameter_opt:
            self.inadequacy_findings.append({
                'program': 'Programs 1 & 2',
                'issue': 'No hyperparameter optimization',
                'description': 'The model was trained with default parameters without ' +
                             'proper hyperparameter tuning.',
                'recommendation': 'Use GridSearchCV or RandomizedSearchCV to find optimal ' +
                                'hyperparameters for the model.'
            })
        
        # 4. Check for adequate test coverage of edge cases
        if self.outlier_indices is not None and self.X_test is not None:
            has_adequate_edge_cases = len(self.outlier_indices) >= 0.1 * len(self.X_test)
        else:
            has_adequate_edge_cases = False
            
        if not has_adequate_edge_cases:
            self.inadequacy_findings.append({
                'program': 'Programs 1 & 2',
                'issue': 'Limited edge case testing',
                'description': 'The testing did not adequately cover edge cases and ' +
                             'potential outliers in the data.',
                'recommendation': 'Include more extensive edge case testing with various ' +
                                'types of synthetic outliers.'
            })
        
        # 5. Check if feature importance was analyzed
        has_feature_importance = False
        if self.tester is not None:
            has_feature_importance = hasattr(self.tester, 'shap_values') and self.tester.shap_values is not None
        
        if not has_feature_importance:
            self.inadequacy_findings.append({
                'program': 'Program 1',
                'issue': 'Missing feature importance analysis',
                'description': 'The initial testing did not analyze which features ' +
                             'contribute most to the model\'s predictions.',
                'recommendation': 'Always include feature importance analysis using ' +
                                'SHAP or similar tools.'
            })
        
        # Always check for these advanced techniques
        self.inadequacy_findings.append({
            'program': 'Programs 1 & 2',
            'issue': 'No adversarial testing',
            'description': 'The previous tests did not include adversarial examples ' +
                         'specifically designed to fool the model.',
            'recommendation': 'Include adversarial testing to identify vulnerabilities ' +
                            'in the model.'
        })
        
        self.inadequacy_findings.append({
            'program': 'Programs 1 & 2',
            'issue': 'No data drift simulation',
            'description': 'The tests did not simulate data drift scenarios where ' +
                         'the distribution of input data changes over time.',
            'recommendation': 'Simulate data drift to evaluate model robustness to ' +
                            'changing data distributions.'
        })
        
        # Print findings
        print(f"  ✓ Found {len(self.inadequacy_findings)} test inadequacies in previous programs")
        for i, finding in enumerate(self.inadequacy_findings):
            print(f"\n  {i+1}. {finding['issue']} in {finding['program']}:")
            print(f"     Description: {finding['description']}")
            print(f"     Recommendation: {finding['recommendation']}")
        
        return self
    
    def summarize_findings(self):
        """Summarize all findings in a comprehensive report."""
        print("\nSummary of Robustness Testing Findings:")
        print("====================================")
        
        try:
            # 1. Existing outliers
            print("\n1. Existing Outliers in Dataset:")
            if self.existing_outliers is not None:
                print(f"   - {self.existing_outliers['count']} outliers detected " + 
                    f"({self.existing_outliers['percentage']:.2f}% of the dataset)")
                if self.existing_outliers['count'] > 0:
                    print("   ⚠ The presence of outliers may affect model training and evaluation.")
                else:
                    print("   ✓ No significant outliers found in the original dataset.")
            else:
                print("   - Outlier detection not performed.")
            
            # 2. Robustness to artificial outliers
            print("\n2. Model Robustness to Outliers:")
            if self.outlier_performance and 'accuracy_impact' in self.outlier_performance:
                print(f"   - Accuracy impact: {self.outlier_performance['accuracy_impact']:.2f}%")
                print(f"   - Recall impact: {self.outlier_performance['recall_impact']:.2f}%")
                print(f"   - F1 score impact: {self.outlier_performance['f1_impact']:.2f}%")
                
                if self.outlier_performance['accuracy_impact'] > 10:
                    print("   ⚠ The model is significantly affected by outliers (>10% drop in accuracy).")
                elif self.outlier_performance['accuracy_impact'] > 5:
                    print("   ! The model is moderately affected by outliers (5-10% drop in accuracy).")
                else:
                    print("   ✓ The model is robust to outliers (<5% drop in accuracy).")
            else:
                print("   - Robustness evaluation not performed.")
            
            # 3. Misclassified outliers
            print(f"\n3. Outlier Misclassification Analysis:")
            if self.outlier_analysis:
                misclassified = [o for o in self.outlier_analysis if o['is_misclassified']]
                print(f"   - {len(misclassified)} of {len(self.outlier_indices)} " + 
                    f"outliers were misclassified ({len(misclassified)/len(self.outlier_indices)*100:.1f}% if applicable)")
                
                if misclassified:
                    print("   - Most common misclassifications:")
                    misclass_counts = {}
                    for o in misclassified:
                        pair = (o['true_class'], o['pred_class'])
                        misclass_counts[pair] = misclass_counts.get(pair, 0) + 1
                    
                    for (true, pred), count in sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"     * {true} → {pred}: {count} instances")
            else:
                print("   - Outlier misclassification analysis not performed.")
            
            # 4. Cross-validation results
            print("\n4. Cross-Validation with Outliers:")
            if self.cv_results and 'clean_mean' in self.cv_results:
                print(f"   - Average accuracy without outliers: {self.cv_results['clean_mean']:.4f} " + 
                    f"± {self.cv_results['clean_std']:.4f}")
                print(f"   - Average accuracy with outliers: {self.cv_results['outlier_mean']:.4f} " + 
                    f"± {self.cv_results['outlier_std']:.4f}")
                print(f"   - Average impact: {self.cv_results['avg_impact']:.2f}%")
                
                if self.cv_results['avg_impact'] > 10:
                    print("   ⚠ Cross-validation confirms significant vulnerability to outliers.")
                elif self.cv_results['avg_impact'] > 5:
                    print("   ! Cross-validation shows moderate vulnerability to outliers.")
                else:
                    print("   ✓ Cross-validation confirms robustness to outliers.")
            else:
                print("   - Cross-validation with outliers not performed.")
            
            # 5. Test inadequacies
            print("\n5. Test Inadequacies in Previous Programs:")
            if not self.inadequacy_findings:
                print("   ✓ No significant test inadequacies detected.")
            else:
                for i, finding in enumerate(self.inadequacy_findings[:3]):  # Show top 3 findings
                    print(f"   {i+1}. {finding['issue']} in {finding['program']}")
                    print(f"      → {finding['recommendation']}")
                
                if len(self.inadequacy_findings) > 3:
                    print(f"   ... plus {len(self.inadequacy_findings) - 3} more findings.")
            
            # 6. Final assessment
            print("\nFinal Assessment:")
            
            # Assess model robustness
            if (self.outlier_performance and 'accuracy_impact' in self.outlier_performance and
                self.cv_results and 'avg_impact' in self.cv_results):
                if self.outlier_performance['accuracy_impact'] > 10 or self.cv_results['avg_impact'] > 10:
                    print("The model shows significant vulnerability to outliers and edge cases. " + 
                        "It should be improved before deployment to production environments.")
                elif self.outlier_performance['accuracy_impact'] > 5 or self.cv_results['avg_impact'] > 5:
                    print("The model shows moderate vulnerability to outliers. Consider implementing " + 
                        "outlier detection and handling mechanisms before deployment.")
                else:
                    print("The model demonstrates good robustness to outliers and edge cases. " + 
                        "It should perform reliably in production environments with similar data distributions.")
            else:
                print("Insufficient data to provide a comprehensive assessment of model robustness.")
            
            # Assess test coverage
            if len(self.inadequacy_findings) > 4:
                print("\nThe previous testing approaches had significant gaps in coverage. " + 
                    "Future testing should address these inadequacies for more comprehensive evaluation.")
            elif len(self.inadequacy_findings) > 0:
                print("\nThe previous testing approaches had some gaps that were addressed in this program. " + 
                    "Future testing should integrate these additional test techniques.")
            else:
                print("\nThe previous testing approaches were comprehensive and adequate.")
        except Exception as e:
            print(f"Error summarizing findings: {str(e)}")
            print("The robustness assessment is incomplete due to errors during analysis.")
        
        return self
    
    def save_results(self, filename='robustness_tester_results.pkl'):
        """Save the robustness tester results to a file."""
        print(f"\nSaving robustness results to {filename}...")
        
        try:
            # Store essential results
            results = {
                'existing_outliers': self.existing_outliers,
                'original_performance': self.original_performance,
                'outlier_performance': self.outlier_performance,
                'cv_results': self.cv_results,
                'inadequacy_findings': self.inadequacy_findings
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"  ✓ Robustness results saved to {filename}")
        except Exception as e:
            print(f"  ! Error saving robustness results: {str(e)}")
        
        return self


def main(load_model=True, load_results=False):
    """Main function to run the robustness testing process."""
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
    
    # Try to load the tester from Program 2
    tester = None
    try:
        print("\nLoading the tester from Program 2...")
        tester = ModelTester(iris_clf)
        tester.load_results()
    except Exception as e:
        print(f"Warning: Could not load tester from Program 2: {str(e)}")
        # Create a minimal compatibility tester
        tester = type('MinimalTester', (), {
            'cv_results': {'accuracy': {'mean': 0, 'std': 0}},
            'findings': {'limitations': [], 'strengths': []},
            'shap_values': None
        })()
    
    # Create the robustness tester
    robustness_tester = RobustnessTester(iris_clf, tester)
    
    # Run the robustness testing pipeline
    robustness_tester.identify_outliers() \
                    .generate_artificial_outliers() \
                    .evaluate_model_robustness() \
                    .perform_cross_validation() \
                    .detect_test_inadequacies() \
                    .summarize_findings() \
                    .save_results()
    
    print("\nRobustness testing completed successfully!")
    
    # Return the tester for further use
    return robustness_tester


if __name__ == "__main__":
    # Set load_model=True to load a pre-trained model
    robustness_tester = main(load_model=True)

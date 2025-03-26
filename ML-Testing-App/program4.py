"""
Advanced Testing of Iris Classification Model
--------------------------------------------
Purpose: Extend testing with hyperparameter optimization, adversarial testing, and data drift simulation
Author: ML Testing Team
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time
import os
import warnings
import pickle
import copy
from matplotlib.colors import ListedColormap

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


class AdvancedTester:
    """A class to perform advanced testing on the Iris classification model."""
    
    def __init__(self, classifier):
        """Initialize with a trained classifier."""
        self.classifier = classifier
        
        # Get attributes from classifier, with fallbacks
        self.X = getattr(classifier, 'X', None)
        self.y = getattr(classifier, 'y', None)
        self.X_train = getattr(classifier, 'X_train', None)
        self.X_test = getattr(classifier, 'X_test', None)
        self.X_train_scaled = getattr(classifier, 'X_train_scaled', None)
        self.X_test_scaled = getattr(classifier, 'X_test_scaled', None)
        self.y_train = getattr(classifier, 'y_train', None)
        self.y_test = getattr(classifier, 'y_test', None)
        self.model = getattr(classifier, 'model', None)
        self.feature_names = getattr(classifier, 'feature_names', None)
        self.target_names = getattr(classifier, 'target_names', None)
        self.scaler = getattr(classifier, 'scaler', None)
        
        # Ensure training data is available even if main dataset isn't
        if self.X is None and hasattr(classifier, 'X_train') and classifier.X_train is not None:
            self.X = classifier.X_train
        
        if self.y is None and hasattr(classifier, 'y_train') and classifier.y_train is not None:
            self.y = classifier.y_train
        
        # Initialize attributes for testing results
        self.optimized_model = None
        self.best_params = None
        self.optimization_results = None
        self.adversarial_examples = None
        self.adversarial_results = None
        self.drift_simulation_results = None
        self.advanced_findings = {
            'hyperparameter_opt': [], 
            'adversarial': [], 
            'data_drift': []
        }
        
        # Check for missing essential attributes
        missing_attrs = []
        for attr in ['X', 'y', 'model', 'feature_names', 'target_names']:
            if getattr(self, attr) is None:
                missing_attrs.append(attr)
        
        if missing_attrs:
            warnings.warn(f"Missing essential attributes: {', '.join(missing_attrs)}. " + 
                         "Some functionality may be limited.")
    
    def optimize_hyperparameters(self, param_grid=None, cv=5):
        """Perform hyperparameter optimization using GridSearchCV."""
        print("\nStep 1: Performing Hyperparameter Optimization...")
        
        if self.X is None or self.y is None:
            print("  ! Cannot perform hyperparameter optimization: dataset not available.")
            self.optimization_results = {
                'best_score': 0,
                'best_params': {},
                'cv_results': {},
                'param_importance': {}
            }
            return self
        
        try:
            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean (used with minkowski)
                }
            
            # Prepare data with proper scaling
            if self.X_train_scaled is not None and self.y_train is not None:
                # Use existing training data if available
                X_for_cv = self.X_train_scaled
                y_for_cv = self.y_train
            else:
                # Otherwise, scale the full dataset
                scaler = StandardScaler()
                X_for_cv = scaler.fit_transform(self.X)
                y_for_cv = self.y
            
            # Create a KNN model for optimization
            knn = KNeighborsClassifier()
            
            # Configure GridSearchCV
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                knn, param_grid, cv=cv_splitter,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            # Perform grid search
            n_combinations = cv * len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])
            if 'p' in param_grid:
                n_combinations *= len(param_grid['p'])
            print(f"  Trying {n_combinations} combinations...")
            #print(f"  Trying {len(grid_search.cv.split(X_for_cv, y_for_cv)) * len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} combinations...")
            start_time = time.time()
            grid_search.fit(X_for_cv, y_for_cv)
            search_time = time.time() - start_time
            
            # Store results
            self.best_params = grid_search.best_params_
            self.optimized_model = grid_search.best_estimator_
            
            # Analyze hyperparameter importance
            param_importance = self._analyze_param_importance(grid_search.cv_results_)
            
            # Store comprehensive results
            self.optimization_results = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_results': grid_search.cv_results_,
                'param_importance': param_importance,
                'search_time': search_time
            }
            
            # Print results
            print(f"  ✓ Hyperparameter optimization completed in {search_time:.2f} seconds")
            print(f"  ✓ Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
            print("\n  Parameter importance:")
            for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {param}: {importance:.4f}")
            
            # Add findings
            orig_accuracy = getattr(self.classifier, 'accuracy', None)
            if orig_accuracy is not None and grid_search.best_score_ > orig_accuracy:
                improvement = (grid_search.best_score_ - orig_accuracy) / orig_accuracy * 100
                self.advanced_findings['hyperparameter_opt'].append(
                    f"Optimized model improved accuracy by {improvement:.2f}% compared to original model"
                )
            
            if 'n_neighbors' in grid_search.best_params_:
                if grid_search.best_params_['n_neighbors'] != 3:  # Default in Program 1
                    self.advanced_findings['hyperparameter_opt'].append(
                        f"Optimal n_neighbors={grid_search.best_params_['n_neighbors']} differs from default (3)"
                    )
            
            # Visualize results
            self._visualize_hyperparameter_results()
            
        except Exception as e:
            print(f"  ! Error in hyperparameter optimization: {str(e)}")
            self.optimization_results = {
                'best_score': 0,
                'best_params': {},
                'cv_results': {},
                'param_importance': {}
            }
        
        return self
    
    def _analyze_param_importance(self, cv_results):
        """Analyze the importance of different hyperparameters."""
        # Extract parameters and corresponding scores
        params = {}
        for param_name in cv_results['params'][0].keys():
            params[param_name] = []
        
        scores = []
        
        for i, param_vals in enumerate(cv_results['params']):
            for param_name, param_val in param_vals.items():
                params[param_name].append(param_val)
            scores.append(cv_results['mean_test_score'][i])
            
        # Calculate parameter importance
        param_importance = {}
        for param_name in params.keys():
            # Convert to DataFrame for analysis
            df = pd.DataFrame({
                param_name: params[param_name],
                'score': scores
            })
            
            # Calculate importance as the variance of mean scores for each parameter value
            param_importance[param_name] = df.groupby(param_name)['score'].mean().var()
            
            # Normalize importance values
            if sum(param_importance.values()) > 0:
                for p in param_importance:
                    param_importance[p] /= sum(param_importance.values())
        
        return param_importance
    
    def _visualize_hyperparameter_results(self):
        """Visualize the hyperparameter optimization results."""
        if not self.optimization_results or not self.optimization_results['cv_results']:
            print("  ! Cannot visualize hyperparameter results: data not available.")
            return self
            
        print("\n  Creating hyperparameter optimization visualizations...")
        
        try:
            # 1. Parameter heatmaps for the most important combinations
            results = pd.DataFrame(self.optimization_results['cv_results'])
            
            # Get the top 2 most important parameters
            top_params = sorted(self.optimization_results['param_importance'].items(), 
                              key=lambda x: x[1], reverse=True)[:2]
            
            if len(top_params) >= 2:
                param1 = top_params[0][0]
                param2 = top_params[1][0]
                
                # Create pivot table for heatmap
                if f'param_{param1}' in results.columns and f'param_{param2}' in results.columns:
                    pivot_data = results.pivot_table(
                        values='mean_test_score', 
                        index=f'param_{param1}',
                        columns=f'param_{param2}',
                        aggfunc='mean'
                    )
                    
                    plt.figure(figsize=(10, 8))
                    ax = sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.4f')
                    plt.title(f'Mean Accuracy for {param1} vs {param2}')
                    plt.tight_layout()
                    plt.savefig('program4_hyperparameter_heatmap.png')
                    plt.close()
            
            # 2. Line plots for each parameter
            plt.figure(figsize=(15, 10))
            for i, param_name in enumerate(self.optimization_results['param_importance'].keys()):
                if f'param_{param_name}' in results.columns:
                    # Create subplot
                    plt.subplot(2, 2, i+1)
                    
                    # Group by parameter value and calculate mean score
                    param_data = results.groupby(f'param_{param_name}')['mean_test_score'].agg(['mean', 'std'])
                    
                    # Plot
                    plt.errorbar(
                        param_data.index, param_data['mean'], 
                        yerr=param_data['std'], 
                        fmt='o-', capsize=5
                    )
                    
                    # Highlight best value
                    best_value = self.optimization_results['best_params'].get(param_name)
                    if best_value is not None and best_value in param_data.index:
                        best_score = param_data.loc[best_value, 'mean']
                        plt.scatter([best_value], [best_score], c='red', s=100, zorder=10, label='Best')
                    
                    plt.title(f'Impact of {param_name}')
                    plt.xlabel(param_name)
                    plt.ylabel('Mean Accuracy')
                    plt.grid(True, alpha=0.3)
                    if i == 0:
                        plt.legend()
                
                # Break if we've created 4 subplots (2x2 grid)
                if i >= 3:
                    break
            
            plt.tight_layout()
            plt.savefig('program4_hyperparameter_plots.png')
            plt.close()
            
            print("  ✓ Hyperparameter visualizations created: program4_hyperparameter_heatmap.png, program4_hyperparameter_plots.png")
        except Exception as e:
            print(f"  ! Error creating hyperparameter visualizations: {str(e)}")
        
        return self
    
    def generate_adversarial_examples(self, n_examples=20, epsilon=0.2):
        """Generate adversarial examples to test model vulnerability."""
        print("\nStep 2: Generating Adversarial Examples...")
        
        if (self.X_test is None or self.y_test is None or self.model is None or 
            self.X_test_scaled is None or self.feature_names is None):
            print("  ! Cannot generate adversarial examples: required data missing.")
            self.adversarial_examples = {
                'X_original': np.array([]),
                'X_adversarial': np.array([]),
                'y_true': np.array([]),
                'y_pred_original': np.array([]),
                'y_pred_adversarial': np.array([]),
                'success_rate': 0,
                'avg_perturbation': 0
            }
            return self
        
        try:
            # Use the optimized model if available, otherwise use the original model
            model_to_test = self.optimized_model if self.optimized_model is not None else self.model
            
            # Randomly select samples to perturb
            n_examples = min(n_examples, len(self.X_test))
            indices = np.random.choice(len(self.X_test), n_examples, replace=False)
            
            # Extract samples
            X_original = self.X_test_scaled[indices]
            y_true = self.y_test[indices]
            
            # Get original predictions
            y_pred_original = model_to_test.predict(X_original)
            
            # Create adversarial examples
            X_adversarial = np.copy(X_original)
            perturbations = []
            
            # For each sample, create a targeted adversarial example
            for i, idx in enumerate(range(len(X_original))):
                # Only try to perturb correctly classified examples
                if y_pred_original[idx] == y_true[idx]:
                    # Identify the target class (different from true class)
                    true_class = y_true[idx]
                    target_classes = [c for c in range(len(self.target_names)) if c != true_class]
                    target_class = np.random.choice(target_classes)
                    
                    # Create method 1: Boundary-based perturbation
                    # Find decision boundary direction toward target class
                    X_perturbed = self._find_adversarial_perturbation(
                        model_to_test, X_original[idx], true_class, target_class, epsilon
                    )
                    
                    if X_perturbed is not None:
                        X_adversarial[idx] = X_perturbed
                        perturbation = np.linalg.norm(X_perturbed - X_original[idx])
                        perturbations.append(perturbation)
            
            # Get predictions on adversarial examples
            y_pred_adversarial = model_to_test.predict(X_adversarial)
            
            # Calculate success rate (how often did adversarial examples fool the model)
            n_success = sum(y_pred_adversarial != y_pred_original)
            success_rate = n_success / n_examples
            
            # Store results
            self.adversarial_examples = {
                'X_original': X_original,
                'X_adversarial': X_adversarial,
                'y_true': y_true,
                'y_pred_original': y_pred_original,
                'y_pred_adversarial': y_pred_adversarial,
                'success_rate': success_rate,
                'avg_perturbation': np.mean(perturbations) if perturbations else 0,
                'indices': indices
            }
            
            # Print results
            print(f"  ✓ Generated {n_examples} potential adversarial examples")
            print(f"  ✓ Successfully fooled the model in {n_success} cases ({success_rate*100:.2f}% success rate)")
            if perturbations:
                print(f"  ✓ Average perturbation magnitude: {np.mean(perturbations):.4f}")
            
            # Add findings
            if success_rate > 0.5:
                self.advanced_findings['adversarial'].append(
                    f"Model is highly vulnerable to adversarial examples ({success_rate*100:.2f}% success rate)"
                )
            elif success_rate > 0.1:
                self.advanced_findings['adversarial'].append(
                    f"Model shows moderate vulnerability to adversarial examples ({success_rate*100:.2f}% success rate)"
                )
            else:
                self.advanced_findings['adversarial'].append(
                    f"Model is relatively robust against adversarial examples ({success_rate*100:.2f}% success rate)"
                )
            
            # Additional finding about perturbation size
            if perturbations:
                self.advanced_findings['adversarial'].append(
                    f"Average perturbation needed: {np.mean(perturbations):.4f} in feature space"
                )
            
            # Visualize adversarial examples
            self._visualize_adversarial_examples()
            
        except Exception as e:
            print(f"  ! Error generating adversarial examples: {str(e)}")
            self.adversarial_examples = {
                'X_original': np.array([]),
                'X_adversarial': np.array([]),
                'y_true': np.array([]),
                'y_pred_original': np.array([]),
                'y_pred_adversarial': np.array([]),
                'success_rate': 0,
                'avg_perturbation': 0
            }
        
        return self
    
    def _find_adversarial_perturbation(self, model, x, true_class, target_class, epsilon, max_iter=100):
        """Find a minimal perturbation that changes the model's prediction."""
        # Start with a copy of the original input
        x_adv = np.copy(x)
        
        # Determine feature importance for this sample (simplified approach)
        # We'll use a simple gradient-free approach for KNN
        
        # Determine the direction to move for each feature
        for i in range(max_iter):
            # Get current prediction
            curr_pred = model.predict([x_adv])[0]
            
            # If we've already changed the prediction, we're done
            if curr_pred != true_class:
                return x_adv
            
            # Try perturbing each feature and see which has the most impact
            best_feature = 0
            best_direction = 1
            best_impact = -float('inf')
            
            for feature_idx in range(len(x)):
                # Try positive and negative perturbations
                for direction in [-1, 1]:
                    # Create perturbed version
                    x_test = np.copy(x_adv)
                    x_test[feature_idx] += direction * epsilon
                    
                    # Get probability of target class
                    probs = model.predict_proba([x_test])[0]
                    
                    # Calculate impact as increase in target class probability
                    impact = probs[target_class] - model.predict_proba([x_adv])[0][target_class]
                    
                    # Update best perturbation if this is better
                    if impact > best_impact:
                        best_impact = impact
                        best_feature = feature_idx
                        best_direction = direction
            
            # Apply the best perturbation
            x_adv[best_feature] += best_direction * epsilon
            
            # If we're not making progress, give up
            if best_impact <= 0:
                break
        
        # Check if we succeeded in changing the prediction
        if model.predict([x_adv])[0] != true_class:
            return x_adv
        else:
            return None
    
    def _visualize_adversarial_examples(self):
        """Visualize the adversarial examples."""
        if (self.adversarial_examples is None or 
            len(self.adversarial_examples['X_original']) == 0 or 
            len(self.adversarial_examples['X_adversarial']) == 0):
            print("  ! Cannot visualize adversarial examples: data not available.")
            return self
            
        print("\n  Creating adversarial example visualizations...")
        
        try:
            # 1. Feature perturbation visualization
            plt.figure(figsize=(12, 6))
            
            # Select a few successful adversarial examples to visualize
            success_mask = self.adversarial_examples['y_pred_original'] != self.adversarial_examples['y_pred_adversarial']
            if np.any(success_mask):
                success_indices = np.where(success_mask)[0]
                n_to_show = min(5, len(success_indices))
                show_indices = success_indices[:n_to_show]
                
                # Create a radar chart for each example
                for i, idx in enumerate(show_indices):
                    plt.subplot(1, n_to_show, i+1)
                    
                    # Extract data
                    x_orig = self.adversarial_examples['X_original'][idx]
                    x_adv = self.adversarial_examples['X_adversarial'][idx]
                    y_true = self.adversarial_examples['y_true'][idx]
                    y_pred_adv = self.adversarial_examples['y_pred_adversarial'][idx]
                    
                    # Plot original vs adversarial
                    n_features = len(x_orig)
                    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop
                    
                    # Normalize feature values for radar chart
                    max_vals = np.max(self.X_test_scaled, axis=0)
                    min_vals = np.min(self.X_test_scaled, axis=0)
                    range_vals = max_vals - min_vals
                    range_vals = np.where(range_vals > 0, range_vals, 1)  # Avoid division by zero
                    
                    x_orig_norm = (x_orig - min_vals) / range_vals
                    x_adv_norm = (x_adv - min_vals) / range_vals
                    
                    # Close the loop for radar chart
                    x_orig_norm = np.append(x_orig_norm, x_orig_norm[0])
                    x_adv_norm = np.append(x_adv_norm, x_adv_norm[0])
                    
                    # Set up radar chart
                    ax = plt.subplot(1, n_to_show, i+1, polar=True)
                    ax.plot(angles, x_orig_norm, 'b-', linewidth=1, label='Original')
                    ax.plot(angles, x_adv_norm, 'r-', linewidth=1, label='Adversarial')
                    ax.fill(angles, x_adv_norm, 'r', alpha=0.1)
                    
                    # Set labels
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([f"F{i+1}" for i in range(n_features)])
                    
                    # Set title
                    true_class = self.target_names[int(y_true)]
                    adv_class = self.target_names[int(y_pred_adv)]
                    ax.set_title(f"{true_class} → {adv_class}", size=9)
                    
                    # Only show legend on the first subplot
                    if i == 0:
                        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
                plt.tight_layout()
                plt.savefig('program4_adversarial_features.png')
                plt.close()
            
            # 2. Decision boundary visualization with adversarial examples
            if len(self.feature_names) >= 2:
                # Use the first two features for visualization
                plt.figure(figsize=(10, 8))
                
                # Create a meshgrid to visualize decision boundaries
                model_to_use = self.optimized_model if self.optimized_model is not None else self.model
                
                # Get range of feature values
                x_min, x_max = self.X_test_scaled[:, 0].min() - 1, self.X_test_scaled[:, 0].max() + 1
                y_min, y_max = self.X_test_scaled[:, 1].min() - 1, self.X_test_scaled[:, 1].max() + 1
                
                # Create meshgrid
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))
                
                # Flatten and create points
                grid = np.c_[xx.ravel(), yy.ravel()]
                
                # If model expects more than 2 features, pad with zeros
                if self.X_test_scaled.shape[1] > 2:
                    grid_pad = np.zeros((grid.shape[0], self.X_test_scaled.shape[1]))
                    grid_pad[:, 0:2] = grid
                    grid = grid_pad
                
                # Get predictions and reshape
                Z = model_to_use.predict(grid)
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundaries
                cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
                cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                
                plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
                
                # Plot original points
                success_mask = self.adversarial_examples['y_pred_original'] != self.adversarial_examples['y_pred_adversarial']
                if np.any(success_mask):
                    # Get indices where adversarial attack was successful
                    success_indices = np.where(success_mask)[0]
                    
                    # Plot original points
                    scatter = plt.scatter(
                        self.adversarial_examples['X_original'][success_indices, 0],
                        self.adversarial_examples['X_original'][success_indices, 1],
                        c=self.adversarial_examples['y_true'][success_indices],
                        cmap=cmap_bold,
                        edgecolors='k',
                        s=80,
                        marker='o',
                        label='Original'
                    )
                    
                    # Plot arrows to adversarial points
                    for idx in success_indices:
                        plt.arrow(
                            self.adversarial_examples['X_original'][idx, 0],
                            self.adversarial_examples['X_original'][idx, 1],
                            self.adversarial_examples['X_adversarial'][idx, 0] - self.adversarial_examples['X_original'][idx, 0],
                            self.adversarial_examples['X_adversarial'][idx, 1] - self.adversarial_examples['X_original'][idx, 1],
                            color='black',
                            width=0.01,
                            head_width=0.05,
                            head_length=0.05,
                            length_includes_head=True
                        )
                    
                    # Plot adversarial points
                    plt.scatter(
                        self.adversarial_examples['X_adversarial'][success_indices, 0],
                        self.adversarial_examples['X_adversarial'][success_indices, 1],
                        c=self.adversarial_examples['y_pred_adversarial'][success_indices],
                        cmap=cmap_bold,
                        # Remove edgecolors for 'x' marker
                        s=80,
                        marker='x',
                        label='Adversarial'
                    )
                    
                    # Create custom legend
                    plt.legend()
                
                plt.xlabel(self.feature_names[0])
                plt.ylabel(self.feature_names[1])
                plt.title('Decision Boundaries with Adversarial Examples')
                plt.tight_layout()
                plt.savefig('program4_adversarial_boundaries.png')
                plt.close()
                
                print("  ✓ Adversarial visualizations created: program4_adversarial_features.png, program4_adversarial_boundaries.png")
            else:
                print("  ✓ Adversarial visualization created: program4_adversarial_features.png")
        except Exception as e:
            print(f"  ! Error creating adversarial visualizations: {str(e)}")
        
        return self
    
    def simulate_data_drift(self, drift_types=['covariate', 'concept'], n_simulations=3):
        """Simulate different types of data drift to test model robustness."""
        print("\nStep 3: Simulating Data Drift...")
        
        if self.X is None or self.y is None or self.model is None:
            print("  ! Cannot simulate data drift: required data missing.")
            self.drift_simulation_results = {
                'scenarios': [],
                'performance_impact': {}
            }
            return self
        
        try:
            # Use the optimized model if available, otherwise use the original model
            model_to_test = self.optimized_model if self.optimized_model is not None else self.model
            
            # Scale the data if needed
            if self.scaler is not None:
                X_scaled = self.scaler.transform(self.X)
            else:
                # Create new scaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(self.X)
            
            # Evaluate baseline performance
            baseline_scores = self._evaluate_model_performance(model_to_test, X_scaled, self.y)
            
            # Store results for all drift scenarios
            scenarios = []
            all_drift_data = []
            
            # 1. Simulate covariate drift (feature distribution changes)
            if 'covariate' in drift_types:
                for i in range(n_simulations):
                    # Create a copy of the data
                    X_drifted = np.copy(X_scaled)
                    
                    # Apply increasing levels of drift
                    drift_magnitude = 0.5 + 0.5 * i  # Increasing drift from 0.5 to 1.5
                    
                    # Choose random features to drift
                    n_features_to_drift = min(2, X_scaled.shape[1])
                    drift_features = np.random.choice(X_scaled.shape[1], n_features_to_drift, replace=False)
                    
                    # Apply drift to selected features
                    for feature_idx in drift_features:
                        # Shift the feature distribution
                        X_drifted[:, feature_idx] += drift_magnitude * np.random.normal(0, 1, size=len(X_drifted))
                    
                    # Evaluate performance on drifted data
                    drift_scores = self._evaluate_model_performance(model_to_test, X_drifted, self.y)
                    
                    # Calculate impact
                    impact = {
                        metric: (baseline_scores[metric] - score) / baseline_scores[metric] * 100
                        for metric, score in drift_scores.items()
                    }
                    
                    # Store scenario details
                    scenario = {
                        'type': 'covariate',
                        'magnitude': drift_magnitude,
                        'affected_features': [self.feature_names[idx] for idx in drift_features],
                        'baseline_scores': baseline_scores,
                        'drift_scores': drift_scores,
                        'impact': impact,
                        'X_drifted': X_drifted,
                        'y_drifted': self.y  # Same labels for covariate drift
                    }
                    
                    scenarios.append(scenario)
                    all_drift_data.append((X_drifted, self.y, f"Covariate Drift {i+1}"))
            
            # 2. Simulate concept drift (relationship between features and labels changes)
            if 'concept' in drift_types:
                for i in range(n_simulations):
                    # Create a copy of the data
                    X_drifted = np.copy(X_scaled)
                    y_drifted = np.copy(self.y)
                    
                    # Apply increasing levels of drift
                    drift_magnitude = 0.05 + 0.05 * i  # Increasing drift probability
                    
                    # Flip labels for a small percentage of samples
                    n_samples_to_flip = int(len(y_drifted) * drift_magnitude)
                    flip_indices = np.random.choice(len(y_drifted), n_samples_to_flip, replace=False)
                    
                    # Get available classes for each sample
                    unique_classes = np.unique(self.y)
                    
                    # Flip labels to a different class
                    for idx in flip_indices:
                        available_classes = [c for c in unique_classes if c != y_drifted[idx]]
                        y_drifted[idx] = np.random.choice(available_classes)
                    
                    # Evaluate performance on drifted data
                    drift_scores = self._evaluate_model_performance(model_to_test, X_drifted, y_drifted)
                    
                    # Calculate impact
                    impact = {
                        metric: (baseline_scores[metric] - score) / baseline_scores[metric] * 100
                        for metric, score in drift_scores.items()
                    }
                    
                    # Store scenario details
                    scenario = {
                        'type': 'concept',
                        'magnitude': drift_magnitude * 100,  # Convert to percentage
                        'n_flipped_labels': n_samples_to_flip,
                        'baseline_scores': baseline_scores,
                        'drift_scores': drift_scores,
                        'impact': impact,
                        'X_drifted': X_drifted,
                        'y_drifted': y_drifted
                    }
                    
                    scenarios.append(scenario)
                    all_drift_data.append((X_drifted, y_drifted, f"Concept Drift {i+1}"))
            
            # 3. Simulate feature drift (new feature relationships)
            if 'feature' in drift_types:
                for i in range(n_simulations):
                    # Create a copy of the data
                    X_drifted = np.copy(X_scaled)
                    
                    # Apply increasing levels of drift
                    drift_magnitude = 0.3 + 0.3 * i  # Increasing drift from 0.3 to 0.9
                    
                    # Choose random feature pairs to create correlations
                    if X_scaled.shape[1] >= 2:
                        # Choose two random features
                        feature_pair = np.random.choice(X_scaled.shape[1], 2, replace=False)
                        
                        # Create new correlation between features
                        for idx in range(len(X_drifted)):
                            X_drifted[idx, feature_pair[1]] = (
                                X_drifted[idx, feature_pair[1]] * (1 - drift_magnitude) + 
                                X_drifted[idx, feature_pair[0]] * drift_magnitude
                            )
                        
                        # Evaluate performance on drifted data
                        drift_scores = self._evaluate_model_performance(model_to_test, X_drifted, self.y)
                        
                        # Calculate impact
                        impact = {
                            metric: (baseline_scores[metric] - score) / baseline_scores[metric] * 100
                            for metric, score in drift_scores.items()
                        }
                        
                        # Store scenario details
                        scenario = {
                            'type': 'feature',
                            'magnitude': drift_magnitude,
                            'affected_features': [self.feature_names[idx] for idx in feature_pair],
                            'baseline_scores': baseline_scores,
                            'drift_scores': drift_scores,
                            'impact': impact,
                            'X_drifted': X_drifted,
                            'y_drifted': self.y  # Same labels for feature drift
                        }
                        
                        scenarios.append(scenario)
                        all_drift_data.append((X_drifted, self.y, f"Feature Drift {i+1}"))
            
            # Store all simulation results
            self.drift_simulation_results = {
                'scenarios': scenarios,
                'baseline_scores': baseline_scores,
                'all_drift_data': all_drift_data
            }
            
            # Calculate average performance impact per drift type
            performance_impact = {}
            for drift_type in drift_types:
                type_scenarios = [s for s in scenarios if s['type'] == drift_type]
                if type_scenarios:
                    # Calculate average impact across all scenarios of this type
                    type_impact = {}
                    for metric in baseline_scores.keys():
                        impacts = [s['impact'][metric] for s in type_scenarios]
                        type_impact[metric] = np.mean(impacts)
                    
                    performance_impact[drift_type] = type_impact
            
            self.drift_simulation_results['performance_impact'] = performance_impact
            
            # Print summary of results
            print("\n  Data Drift Simulation Results:")
            for drift_type, impact in performance_impact.items():
                print(f"  {drift_type.capitalize()} Drift Impact:")
                for metric, value in impact.items():
                    print(f"    - {metric}: {value:.2f}% degradation")
            
            # Add findings
            for drift_type, impact in performance_impact.items():
                accuracy_impact = impact.get('accuracy', 0)
                
                if accuracy_impact > 20:
                    self.advanced_findings['data_drift'].append(
                        f"Model is highly vulnerable to {drift_type} drift ({accuracy_impact:.2f}% accuracy drop)"
                    )
                elif accuracy_impact > 5:
                    self.advanced_findings['data_drift'].append(
                        f"Model shows moderate vulnerability to {drift_type} drift ({accuracy_impact:.2f}% accuracy drop)"
                    )
                else:
                    self.advanced_findings['data_drift'].append(
                        f"Model is relatively robust against {drift_type} drift ({accuracy_impact:.2f}% accuracy drop)"
                    )
            
            # Visualize drift results
            self._visualize_drift_results()
            
        except Exception as e:
            print(f"  ! Error simulating data drift: {str(e)}")
            self.drift_simulation_results = {
                'scenarios': [],
                'performance_impact': {}
            }
        
        return self
    
    def _evaluate_model_performance(self, model, X, y):
        """Evaluate model performance with multiple metrics."""
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1
        }
    
    def _visualize_drift_results(self):
        """Visualize the data drift simulation results."""
        if not self.drift_simulation_results or not self.drift_simulation_results['scenarios']:
            print("  ! Cannot visualize drift results: data not available.")
            return self
            
        print("\n  Creating data drift visualizations...")
        
        try:
            # 1. Performance impact barchart by drift type and magnitude
            plt.figure(figsize=(12, 8))
            
            # Group scenarios by type
            scenarios_by_type = {}
            for scenario in self.drift_simulation_results['scenarios']:
                drift_type = scenario['type']
                if drift_type not in scenarios_by_type:
                    scenarios_by_type[drift_type] = []
                scenarios_by_type[drift_type].append(scenario)
            
            # Plot performance impact by type
            n_types = len(scenarios_by_type)
            for i, (drift_type, scenarios) in enumerate(scenarios_by_type.items()):
                plt.subplot(n_types, 1, i+1)
                
                # Sort scenarios by magnitude
                scenarios = sorted(scenarios, key=lambda s: s['magnitude'])
                
                # Extract data for plotting
                magnitudes = [s['magnitude'] for s in scenarios]
                accuracy_impacts = [s['impact']['accuracy'] for s in scenarios]
                recall_impacts = [s['impact']['recall'] for s in scenarios]
                f1_impacts = [s['impact']['f1'] for s in scenarios]
                
                # Create bar positions
                x = np.arange(len(magnitudes))
                width = 0.25
                
                # Plot bars
                plt.bar(x - width, accuracy_impacts, width, label='Accuracy Impact')
                plt.bar(x, recall_impacts, width, label='Recall Impact')
                plt.bar(x + width, f1_impacts, width, label='F1 Impact')
                
                # Add labels
                plt.xlabel('Drift Magnitude' if i == n_types-1 else '')
                plt.ylabel('Performance Impact (%)')
                plt.title(f'{drift_type.capitalize()} Drift Impact')
                plt.xticks(x, [f"{m:.2f}" for m in magnitudes])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('program4_drift_impact.png')
            plt.close()
            
            # 2. Feature distribution shifts visualization
            if self.drift_simulation_results['all_drift_data'] and self.feature_names:
                # Select a few features to visualize
                n_features = min(4, len(self.feature_names))
                
                plt.figure(figsize=(15, 10))
                for i in range(n_features):
                    plt.subplot(2, 2, i+1)
                    
                    # Plot original data distribution
                    if self.X is not None and self.scaler is not None:
                        X_scaled = self.scaler.transform(self.X)
                        sns.kdeplot(X_scaled[:, i], label='Original', color='blue')
                    
                    # Plot drifted data distributions
                    for X_drifted, _, label in self.drift_simulation_results['all_drift_data']:
                        if 'Covariate' in label:  # Only show covariate drift for distribution plots
                            sns.kdeplot(X_drifted[:, i], label=label, alpha=0.7)
                    
                    plt.title(f'Feature Distribution: {self.feature_names[i]}')
                    plt.xlabel('Standardized Value')
                    plt.ylabel('Density')
                    if i == 0:
                        plt.legend()
                
                plt.tight_layout()
                plt.savefig('program4_drift_distributions.png')
                plt.close()
            
            # 3. Decision boundary changes visualization (for 2D projection)
            if len(self.feature_names) >= 2 and self.drift_simulation_results['all_drift_data']:
                # Use the first two features for visualization
                from sklearn.decomposition import PCA
                
                # Create PCA for 2D projection
                pca = PCA(n_components=2)
                
                # Project original data
                if self.X is not None and self.scaler is not None:
                    X_scaled = self.scaler.transform(self.X)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create figure for decision boundaries
                    plt.figure(figsize=(15, 10))
                    
                    # Plot original data and decision boundary
                    plt.subplot(2, 2, 1)
                    self._plot_decision_boundary(self.model, X_pca, self.y, 'Original Model')
                    
                    # Plot drifted data and boundaries for up to 3 scenarios
                    for i, (X_drifted, y_drifted, label) in enumerate(self.drift_simulation_results['all_drift_data'][:3]):
                        # Project drifted data
                        X_drifted_pca = pca.transform(X_drifted)
                        
                        # Plot
                        plt.subplot(2, 2, i+2)
                        self._plot_decision_boundary(self.model, X_drifted_pca, y_drifted, label)
                    
                    plt.tight_layout()
                    plt.savefig('program4_drift_boundaries.png')
                    plt.close()
            
            print("  ✓ Data drift visualizations created: program4_drift_impact.png, program4_drift_distributions.png, program4_drift_boundaries.png")
        except Exception as e:
            print(f"  ! Error creating drift visualizations: {str(e)}")
        
        return self
    
    def _plot_decision_boundary(self, model, X, y, title):
        """Modified function to handle PCA-transformed data correctly."""
        # Create meshgrid
        h = 0.02  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Get predictions for grid (create a simple KNN model just for visualization)
        # This is key: We train a new model on the PCA data instead of using original model
        temp_model = KNeighborsClassifier(n_neighbors=3)
        temp_model.fit(X, y)  # Train on PCA-transformed data

        # Use the visualization-specific model for predictions
        Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Get predictions for grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k')
        plt.title(title)
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
    
    def summarize_findings(self):
        """Summarize all findings from the advanced testing."""
        print("\nSummary of Advanced Testing Findings:")
        print("===================================")
        
        try:
            # 1. Hyperparameter Optimization Results
            print("\n1. Hyperparameter Optimization Findings:")
            if self.optimization_results and self.optimization_results['best_params']:
                print(f"   ✓ Best hyperparameters: {self.optimization_results['best_params']}")
                print(f"   ✓ Best cross-validation score: {self.optimization_results['best_score']:.4f}")
                
                # Compare to original model
                orig_accuracy = getattr(self.classifier, 'accuracy', None)
                if orig_accuracy is not None:
                    improvement = (self.optimization_results['best_score'] - orig_accuracy) / orig_accuracy * 100
                    if improvement > 0:
                        print(f"   ✓ Improved accuracy by {improvement:.2f}% compared to original model")
                    else:
                        print(f"   ! No improvement over original model (change: {improvement:.2f}%)")
                
                # Print parameter importance
                print("\n   Parameter importance:")
                for param, importance in sorted(self.optimization_results['param_importance'].items(), 
                                              key=lambda x: x[1], reverse=True):
                    print(f"   - {param}: {importance:.4f}")
                
                # Print findings
                for finding in self.advanced_findings['hyperparameter_opt']:
                    print(f"   • {finding}")
            else:
                print("   ! Hyperparameter optimization not performed or failed.")
            
            # 2. Adversarial Testing Results
            print("\n2. Adversarial Testing Findings:")
            if self.adversarial_examples and 'success_rate' in self.adversarial_examples:
                success_rate = self.adversarial_examples['success_rate']
                print(f"   ✓ Generated {len(self.adversarial_examples['X_original'])} potential adversarial examples")
                print(f"   ✓ Success rate: {success_rate*100:.2f}% of examples fooled the model")
                
                if 'avg_perturbation' in self.adversarial_examples and self.adversarial_examples['avg_perturbation'] > 0:
                    print(f"   ✓ Average perturbation magnitude: {self.adversarial_examples['avg_perturbation']:.4f}")
                
                # Print vulnerability assessment
                if success_rate > 0.5:
                    print("   ! HIGH VULNERABILITY: Model is easily fooled by adversarial examples")
                elif success_rate > 0.1:
                    print("   ! MODERATE VULNERABILITY: Model can be fooled by carefully crafted adversarial examples")
                else:
                    print("   ✓ LOW VULNERABILITY: Model is relatively robust against adversarial examples")
                
                # Print findings
                for finding in self.advanced_findings['adversarial']:
                    print(f"   • {finding}")
            else:
                print("   ! Adversarial testing not performed or failed.")
            
            # 3. Data Drift Simulation Results
            print("\n3. Data Drift Simulation Findings:")
            if self.drift_simulation_results and 'performance_impact' in self.drift_simulation_results:
                performance_impact = self.drift_simulation_results['performance_impact']
                
                if performance_impact:
                    for drift_type, impact in performance_impact.items():
                        print(f"   {drift_type.capitalize()} Drift Impact:")
                        for metric, value in impact.items():
                            print(f"   - {metric}: {value:.2f}% degradation")
                    
                    # Overall vulnerability assessment
                    max_impact = max(
                        impact.get('accuracy', 0) 
                        for impact in performance_impact.values()
                    )
                    
                    if max_impact > 20:
                        print("\n   ! HIGH VULNERABILITY: Model performance degrades significantly under data drift")
                    elif max_impact > 5:
                        print("\n   ! MODERATE VULNERABILITY: Model shows some sensitivity to data drift")
                    else:
                        print("\n   ✓ LOW VULNERABILITY: Model is relatively robust against data drift")
                
                # Print findings
                for finding in self.advanced_findings['data_drift']:
                    print(f"   • {finding}")
            else:
                print("   ! Data drift simulation not performed or failed.")
            
            # 4. Final Assessment and Recommendations
            print("\nFinal Assessment and Recommendations:")
            
            recommendations = []
            
            # Hyperparameter optimization recommendations
            if self.optimization_results and self.optimization_results['best_params']:
                # Check if hyperparameter tuning provided significant improvement
                orig_accuracy = getattr(self.classifier, 'accuracy', None)
                if orig_accuracy is not None:
                    improvement = (self.optimization_results['best_score'] - orig_accuracy) / orig_accuracy * 100
                    if improvement > 5:
                        recommendations.append(
                            "Implement optimized hyperparameters to improve model performance"
                        )
            else:
                recommendations.append(
                    "Perform hyperparameter optimization to potentially improve model performance"
                )
            
            # Adversarial testing recommendations
            if self.adversarial_examples and 'success_rate' in self.adversarial_examples:
                success_rate = self.adversarial_examples['success_rate']
                if success_rate > 0.1:
                    recommendations.append(
                        "Implement adversarial training to improve model robustness"
                    )
            
            # Data drift recommendations
            if self.drift_simulation_results and 'performance_impact' in self.drift_simulation_results:
                performance_impact = self.drift_simulation_results['performance_impact']
                
                if performance_impact:
                    max_impact = max(
                        impact.get('accuracy', 0) 
                        for impact in performance_impact.values()
                    )
                    
                    if max_impact > 5:
                        recommendations.append(
                            "Implement drift detection mechanisms to monitor model in production"
                        )
                    
                    if max_impact > 10:
                        recommendations.append(
                            "Consider periodic model retraining using production data"
                        )
            
            # Print recommendations
            for i, recommendation in enumerate(recommendations):
                print(f"  {i+1}. {recommendation}")
            
            if not recommendations:
                print("  The model shows good robustness across all tests. No critical issues identified.")
            
        except Exception as e:
            print(f"Error summarizing findings: {str(e)}")
            print("The advanced testing assessment is incomplete due to errors during analysis.")
        
        return self
    
    def save_results(self, filename='advanced_tester_results.pkl'):
        """Save the advanced testing results to a file."""
        print(f"\nSaving advanced testing results to {filename}...")
        
        try:
            # Store essential results
            results = {
                'hyperparameter_optimization': {
                    'best_params': self.best_params,
                    'optimization_results': self.optimization_results
                },
                'adversarial_testing': {
                    'success_rate': self.adversarial_examples.get('success_rate', 0) if self.adversarial_examples else 0,
                    'avg_perturbation': self.adversarial_examples.get('avg_perturbation', 0) if self.adversarial_examples else 0
                },
                'data_drift_simulation': {
                    'performance_impact': self.drift_simulation_results.get('performance_impact', {}) if self.drift_simulation_results else {}
                },
                'advanced_findings': self.advanced_findings
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"  ✓ Advanced testing results saved to {filename}")
        except Exception as e:
            print(f"  ! Error saving advanced testing results: {str(e)}")
        
        return self


def main(load_model=True):
    """Main function to run the advanced testing process."""
    # Create an instance of the classifier
    iris_clf = IrisClassifier()
    
    if load_model:
        # Try to load a pre-trained model
        try:
            iris_clf.load_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}. Training a new model.")
            # If loading fails, create a minimal compatible classifier
            print("Cannot load or train model. Advanced testing capabilities will be limited.")
    
    # Create the advanced tester
    advanced_tester = AdvancedTester(iris_clf)
    
    # Run the advanced testing pipeline
    advanced_tester.optimize_hyperparameters() \
                  .generate_adversarial_examples() \
                  .simulate_data_drift() \
                  .summarize_findings() \
                  .save_results()
    
    print("\nAdvanced testing completed successfully!")
    
    # Return the tester for further use
    return advanced_tester


if __name__ == "__main__":
    # Run the main function
    advanced_tester = main(load_model=True)

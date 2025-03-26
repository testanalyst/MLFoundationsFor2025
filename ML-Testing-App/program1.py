"""
Iris Dataset Classification with KNN
-----------------------------------
Purpose: Train a simple KNN model on the Iris dataset and evaluate its performance
Author: ML Testing Team
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os
import warnings

# Set random seed for reproducibility
np.random.seed(42)


class IrisClassifier:
    """A class to load, train, and evaluate a KNN model on the Iris dataset."""
    
    def __init__(self):
        """Initialize the classifier with empty attributes."""
        self.X = None  # Feature data
        self.y = None  # Target labels
        self.feature_names = None  # Names of features
        self.target_names = None  # Names of target classes
        self.X_train = None  # Training features
        self.X_test = None  # Testing features
        self.y_train = None  # Training labels
        self.y_test = None  # Testing labels
        self.model = None  # The KNN model
        self.scaler = None  # Feature scaler
        self.y_pred = None  # Model predictions
        self.training_time = None  # Time taken to train the model
        self.prediction_time = None  # Time taken to make predictions
        self.df = None  # DataFrame representation of data
        
    def load_data(self):
        """Load the Iris dataset from scikit-learn."""
        print("Step 1: Loading the Iris dataset...")
        # Load the iris dataset
        iris = load_iris()
        
        # Extract features, target, and metadata
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Create a pandas DataFrame for easier data manipulation
        self.df = pd.DataFrame(data=self.X, columns=self.feature_names)
        self.df['species'] = [self.target_names[i] for i in self.y]
        
        print(f"  ✓ Dataset loaded: {len(self.X)} samples, {len(self.feature_names)} features")
        print(f"  ✓ Classes: {', '.join(self.target_names)}")
        
        return self
    
    def explore_data(self):
        """Perform basic exploratory data analysis on the dataset."""
        print("\nStep 2: Exploring the dataset...")
        
        # Display basic statistics
        print("\n  Basic Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        print(f"\n  ✓ Missing values: {missing_values}")
        
        # Check class distribution
        class_counts = self.df['species'].value_counts()
        print("\n  Class Distribution:")
        print(class_counts)
        
        # Visualize the data
        self._visualize_data()
        
        return self
    
    def _visualize_data(self):
        """Create visualizations to understand the dataset better."""
        print("\n  Creating visualizations...")
        
        # 1. Pairplot to see relationships between features
        plt.figure(figsize=(10, 8))
        sns.pairplot(self.df, hue='species', markers=["o", "s", "D"])
        plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
        plt.tight_layout()
        plt.savefig('program1_pairplot.png')
        plt.close()
        
        # 2. Boxplot to check for outliers and distribution
        plt.figure(figsize=(12, 6))
        for i, feature in enumerate(self.feature_names):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='species', y=feature, data=self.df)
            plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig('program1_boxplots.png')
        plt.close()
        
        print("  ✓ Visualizations created: program1_pairplot.png, program1_boxplots.png")
        
        return self
    
    def prepare_data(self, test_size=0.3):
        """Split the data into training and testing sets and scale features."""
        print("\nStep 3: Preparing the data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale the features (important for KNN)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"  ✓ Data split: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
        print("  ✓ Features standardized")
        
        return self
    
    def train_model(self, n_neighbors=3):
        """Train a KNN model on the training data."""
        print("\nStep 4: Training the KNN model...")
        
        # Create and train the model
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Measure training time
        start_time = time.time()
        self.model.fit(self.X_train_scaled, self.y_train)
        self.training_time = time.time() - start_time
        
        print(f"  ✓ Model trained with {n_neighbors} neighbors")
        print(f"  ✓ Training time: {self.training_time:.4f} seconds")
        
        return self
    
    def evaluate_model(self):
        """Evaluate the model using various metrics."""
        print("\nStep 5: Evaluating the model...")
        
        # Make predictions and measure time
        start_time = time.time()
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.prediction_time = time.time() - start_time
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred, average='macro')
        self.f1 = f1_score(self.y_test, self.y_pred, average='macro')
        self.class_report = classification_report(self.y_test, self.y_pred, 
                                                 target_names=self.target_names)
        
        # Print metrics
        print(f"  ✓ Prediction time: {self.prediction_time:.4f} seconds")
        print(f"  ✓ Accuracy: {self.accuracy:.4f}")
        print(f"  ✓ Recall: {self.recall:.4f}")
        print(f"  ✓ F1 Score: {self.f1:.4f}")
        
        print("\n  Classification Report:")
        print(self.class_report)
        
        print("\n  Confusion Matrix:")
        print(self.conf_matrix)
        
        # Visualize results
        self._visualize_results()
        
        return self
    
    def _visualize_results(self):
        """Create visualizations of the model evaluation results."""
        print("\n  Creating result visualizations...")
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('program1_confusion_matrix.png')
        plt.close()
        
        # 2. Decision Boundary Visualization (for 2 features only)
        plt.figure(figsize=(12, 5))
        
        # Select the first two features for visualization
        for idx, pair in enumerate([(0, 1), (2, 3)]):
            plt.subplot(1, 2, idx+1)
            
            # Plot decision boundaries
            x_min, x_max = self.X_test[:, pair[0]].min() - 0.5, self.X_test[:, pair[0]].max() + 0.5
            y_min, y_max = self.X_test[:, pair[1]].min() - 0.5, self.X_test[:, pair[1]].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            # Scale the features for prediction
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Create full grid with mean values instead of zeros (FIX)
            feature_means = np.mean(self.X_test, axis=0)
            full_grid = np.tile(feature_means, (grid_points.shape[0], 1))
            full_grid[:, pair[0]] = grid_points[:, 0]
            full_grid[:, pair[1]] = grid_points[:, 1]
            
            full_grid_scaled = self.scaler.transform(full_grid)
            
            # Use only the relevant features
            Z = self.model.predict(full_grid_scaled)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            
            # Plot test points
            scatter = plt.scatter(self.X_test[:, pair[0]], self.X_test[:, pair[1]], 
                      c=self.y_test, edgecolors='k', cmap='coolwarm')
            plt.xlabel(self.feature_names[pair[0]])
            plt.ylabel(self.feature_names[pair[1]])
            plt.title(f'Decision Boundaries for Features {pair[0]+1} and {pair[1]+1}')
            
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.tight_layout()
        plt.savefig('program1_decision_boundaries.png')
        plt.close()
        
        print("  ✓ Result visualizations created: program1_confusion_matrix.png, program1_decision_boundaries.png")
        
        return self
    
    def save_model(self, filename='iris_classifier_model.pkl'):
        """Save the trained model and associated data to a file."""
        print("\nStep 6: Saving the model...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'accuracy': self.accuracy if hasattr(self, 'accuracy') else None,
            'conf_matrix': self.conf_matrix if hasattr(self, 'conf_matrix') else None,
            'X_train': self.X_train if hasattr(self, 'X_train') else None,
            'X_test': self.X_test if hasattr(self, 'X_test') else None,
            'y_train': self.y_train if hasattr(self, 'y_train') else None,
            'y_test': self.y_test if hasattr(self, 'y_test') else None,
            'X_train_scaled': self.X_train_scaled if hasattr(self, 'X_train_scaled') else None,
            'X_test_scaled': self.X_test_scaled if hasattr(self, 'X_test_scaled') else None,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  ✓ Model saved to {filename}")
        return self
    
    def load_model(self, filename='iris_classifier_model.pkl'):
        """Load a trained model and associated data from a file."""
        print(f"\nLoading model from {filename}...")
        
        if not os.path.exists(filename):
            print(f"  ! Model file {filename} not found. Please train the model first.")
            return self
            
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
        print(f"  ✓ Model accuracy: {getattr(self, 'accuracy', 'Not evaluated')}")
        
        return self


def main(load_model=False):
    """Main function to run the Iris classification process."""
    # Create an instance of the IrisClassifier
    iris_clf = IrisClassifier()
    
    if load_model:
        # Try to load a pre-trained model
        try:
            iris_clf.load_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}. Training a new model.")
            load_model = False
    
    if not load_model:
        # Run the full pipeline
        iris_clf.load_data() \
               .explore_data() \
               .prepare_data() \
               .train_model() \
               .evaluate_model() \
               .save_model()
    
    print("\nProcess completed successfully!")
    
    # Return the classifier for later use
    return iris_clf


if __name__ == "__main__":
    # Set load_model=True if you want to load a pre-trained model instead of training a new one
    classifier = main(load_model=False)

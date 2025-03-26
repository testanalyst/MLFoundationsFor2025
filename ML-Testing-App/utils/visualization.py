"""
Visualization utilities for ML Testing Dashboard
-----------------------------------------------
Provides advanced visualization capabilities for model testing
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import io
import base64
from matplotlib.colors import ListedColormap
import warnings


def get_plot_as_base64(fig=None, close_after=True):
    """
    Convert a matplotlib figure to base64 for embedding in HTML.
    
    Args:
        fig: Matplotlib figure (if None, uses current figure)
        close_after: Whether to close the figure after converting
        
    Returns:
        str: Base64 encoded string of the figure
    """
    if fig is None:
        fig = plt.gcf()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    if close_after:
        plt.close(fig)
    
    return img_str


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', cmap='Blues'):
    """
    Create a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Title for the plot
        cmap: Colormap for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure containing the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(feature_names, importance_values, title='Feature Importance'):
    """
    Create a feature importance visualization.
    
    Args:
        feature_names: Names of features
        importance_values: Importance values for each feature
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure containing the feature importance plot
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance_values[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_names)), sorted_importance, align='center')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel('Importance')
    plt.tight_layout()
    
    return fig


def plot_decision_boundaries(model, X, y, feature_idx1=0, feature_idx2=1, 
                            feature_names=None, class_names=None, scaler=None):
    """
    Plot decision boundaries for a model based on two selected features.
    
    Args:
        model: Trained classifier model with predict method
        X: Feature data
        y: Target labels
        feature_idx1: Index of first feature to use
        feature_idx2: Index of second feature to use
        feature_names: Names of features
        class_names: Names of classes
        scaler: Fitted StandardScaler if features are scaled
        
    Returns:
        matplotlib.figure.Figure: Figure containing the decision boundary plot
    """
    if feature_names is None:
        feature_names = [f"Feature {feature_idx1+1}", f"Feature {feature_idx2+1}"]
    
    # Extract the two features we are interested in
    X_reduced = X[:, [feature_idx1, feature_idx2]]
    
    # Create a meshgrid to visualize the decision boundaries
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Create full feature vectors by using the mean for other features
    if X.shape[1] > 2:
        feature_means = np.mean(X, axis=0)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        full_grid = np.zeros((mesh_points.shape[0], X.shape[1]))
        
        for i in range(X.shape[1]):
            if i == feature_idx1:
                full_grid[:, i] = mesh_points[:, 0]
            elif i == feature_idx2:
                full_grid[:, i] = mesh_points[:, 1]
            else:
                full_grid[:, i] = feature_means[i]
    else:
        full_grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Apply scaling if provided
    if scaler is not None:
        full_grid = scaler.transform(full_grid)
    
    # Get predictions
    Z = model.predict(full_grid)
    Z = Z.reshape(xx.shape)
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    
    # Plot training points
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold, edgecolors='k')
    
    # Add legend
    if class_names is not None:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, class_names)
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f'Decision Boundaries for {feature_names[0]} vs {feature_names[1]}')
    plt.tight_layout()
    
    return fig


def plot_performance_comparison(metrics_dict, title='Performance Comparison'):
    """
    Create a bar chart comparing multiple performance metrics.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and values as values
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure containing the performance comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, alpha=0.8)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_roc_curve(y_true, y_score, class_names=None):
    """
    Create a ROC curve for multi-class classification.
    
    Args:
        y_true: True labels
        y_score: Predicted probability scores (must be one-hot encoded)
        class_names: Names of classes
        
    Returns:
        matplotlib.figure.Figure: Figure containing the ROC curve
    """
    if not hasattr(y_score, 'shape') or len(y_score.shape) != 2:
        warnings.warn("y_score should be a 2D array with predicted probabilities for each class")
        return None
    
    n_classes = y_score.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert y_true to one-hot format
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i, color, cls in zip(range(n_classes), colors, class_names):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{cls} (area = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    return fig


def plot_outliers(X, outlier_mask, feature_names=None, class_labels=None, class_names=None):
    """
    Create a visualization of outliers in the dataset.
    
    Args:
        X: Feature data
        outlier_mask: Boolean array indicating outliers
        feature_names: Names of features
        class_labels: Class labels for each data point
        class_names: Names of classes
        
    Returns:
        matplotlib.figure.Figure: Figure containing the outlier visualization
    """
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Create pairplot for first 4 features (or fewer if there are less)
    n_features = min(4, X.shape[1])
    feature_pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
    
    fig, axes = plt.subplots(len(feature_pairs), 1, figsize=(10, 5*len(feature_pairs)))
    if len(feature_pairs) == 1:
        axes = [axes]
    
    for ax, (f1, f2) in zip(axes, feature_pairs):
        # If we have class information, use it for coloring
        if class_labels is not None:
            unique_classes = np.unique(class_labels)
            
            # Plot normal points by class
            for cls in unique_classes:
                mask = (class_labels == cls) & (~outlier_mask)
                cls_name = class_names[cls] if class_names is not None else f"Class {cls}"
                ax.scatter(X[mask, f1], X[mask, f2], label=f"{cls_name} (normal)", alpha=0.7)
            
            # Plot outliers with different markers by class
            for cls in unique_classes:
                mask = (class_labels == cls) & outlier_mask
                if np.any(mask):
                    cls_name = class_names[cls] if class_names is not None else f"Class {cls}"
                    ax.scatter(X[mask, f1], X[mask, f2], marker='x', s=100, 
                             label=f"{cls_name} (outlier)", alpha=0.9)
        else:
            # Plot without class information
            ax.scatter(X[~outlier_mask, f1], X[~outlier_mask, f2], label='Normal', alpha=0.7)
            ax.scatter(X[outlier_mask, f1], X[outlier_mask, f2], marker='x', s=100, 
                     label='Outlier', alpha=0.9)
        
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(f'Outliers in {feature_names[f1]} vs {feature_names[f2]}')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_distributions(X, feature_names=None, class_labels=None, class_names=None):
    """
    Create boxplots showing the distribution of features.
    
    Args:
        X: Feature data
        feature_names: Names of features
        class_labels: Class labels for each data point
        class_names: Names of classes
        
    Returns:
        matplotlib.figure.Figure: Figure containing the feature distribution plots
    """
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    n_features = min(4, X.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    if class_labels is not None:
        # Convert to DataFrame for easy plotting
        df = pd.DataFrame(X[:, :n_features], columns=feature_names[:n_features])
        if class_names is not None:
            df['class'] = [class_names[i] for i in class_labels]
        else:
            df['class'] = class_labels
        
        for i in range(n_features):
            sns.boxplot(x='class', y=feature_names[i], data=df, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature_names[i]}')
    else:
        for i in range(n_features):
            sns.boxplot(y=X[:, i], ax=axes[i])
            axes[i].set_title(f'Distribution of {feature_names[i]}')
    
    for i in range(n_features, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_pairplot(X, y=None, feature_names=None, class_names=None):
    """
    Create a pairplot of the dataset features.
    
    Args:
        X: Feature data
        y: Target labels
        feature_names: Names of features
        class_names: Names of classes
        
    Returns:
        seaborn.axisgrid.PairGrid: Pairplot visualization
    """
    # Convert to DataFrame
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    
    if y is not None:
        if class_names is not None:
            df['class'] = [class_names[i] for i in y]
        else:
            df['class'] = y
        
        grid = sns.pairplot(df, hue='class', markers=["o", "s", "D"])
    else:
        grid = sns.pairplot(df)
    
    plt.suptitle('Pairplot of Features', y=1.02)
    
    return grid

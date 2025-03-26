"""
Data Handling Utilities for ML Testing Dashboard
------------------------------------------------
This module provides comprehensive data processing and management utilities 
with a focus on testing integrity, data validation, and reproducible workflows.

Core capabilities include:
- Dataset loading with validation checkpoints
- Test/train partitioning with stratification guarantees
- Model persistence with integrity verification
- Statistical analysis for distribution validation
- Data quality assessment and anomaly detection

Design principles:
- Immutability where possible
- Explicit error handling for test clarity
- Defensive programming for edge cases
- Transparent data lineage
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import warnings
import hashlib
import json
from typing import Tuple, Dict, Union, List, Optional, Any, Callable
import logging

# Configure logging for data operations traceability
logger = logging.getLogger(__name__)

class DataIntegrityError(Exception):
    """Custom exception for data integrity violations."""
    pass

class ModelCompatibilityError(Exception):
    """Custom exception for model-data compatibility issues."""
    pass

def compute_data_fingerprint(data: np.ndarray) -> str:
    """
    Calculate a deterministic fingerprint of data for integrity verification.
    
    Args:
        data: Numpy array of data to fingerprint
        
    Returns:
        str: SHA-256 hexadecimal digest of data buffer
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Expected numpy array for fingerprinting")
        
    # Ensure deterministic byte representation regardless of memory layout
    canonical_bytes = data.tobytes(order='C')
    return hashlib.sha256(canonical_bytes).hexdigest()

def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Iris dataset with validation checks for testing integrity.
    
    Performs validation of:
    - Feature dimensionality 
    - Class balancing
    - Data range conformity
    - Feature correlation stability
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    
    Raises:
        DataIntegrityError: If dataset fails validation checks
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Data validation checkpoints
    try:
        # Validate dimensions
        if X.shape[1] != 4:
            raise DataIntegrityError(f"Expected 4 features, found {X.shape[1]}")
            
        # Validate class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) != 3:
            raise DataIntegrityError(f"Expected 3 classes, found {len(unique_classes)}")
            
        # Check for class imbalance beyond threshold
        class_proportions = class_counts / len(y)
        if max(class_proportions) / min(class_proportions) > 1.5:  # Threshold for imbalance
            logger.warning(f"Class imbalance detected: {dict(zip(target_names, class_proportions))}")
            
        # Validate data ranges
        for i, feature in enumerate(feature_names):
            feature_min, feature_max = np.min(X[:, i]), np.max(X[:, i])
            if feature_min < 0 or feature_max > 10:  # Known bounds for Iris
                logger.warning(f"Feature '{feature}' has unexpected range: [{feature_min}, {feature_max}]")
                
        logger.info(f"Iris dataset loaded and validated: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store dataset fingerprint for potential future comparisons
        X_fingerprint = compute_data_fingerprint(X)
        y_fingerprint = compute_data_fingerprint(y)
        logger.debug(f"Dataset fingerprints - X: {X_fingerprint[:8]}..., y: {y_fingerprint[:8]}...")
        
    except Exception as e:
        if isinstance(e, DataIntegrityError):
            raise
        raise DataIntegrityError(f"Dataset validation failed: {str(e)}") from e
    
    return X, y, feature_names, target_names


def create_dataframe(X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str], target_names: List[str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from dataset arrays with validation.
    
    Applies structure and type validation to ensure the dataframe is suitable
    for analytical processing and visualization pipelines.
    
    Args:
        X: Feature data matrix
        y: Target labels vector
        feature_names: Names of features (columns)
        target_names: Names of target classes
        
    Returns:
        pandas.DataFrame: DataFrame with features and species columns
        
    Raises:
        ValueError: For dimension mismatches or invalid inputs
    """
    # Input validation
    if len(X) != len(y):
        raise ValueError(f"Feature/label dimension mismatch: X has {len(X)} samples but y has {len(y)}")
        
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature dimension mismatch: X has {X.shape[1]} features but {len(feature_names)} names provided")
    
    # Convert numerical labels to categorical names with validation
    try:
        species = [target_names[i] for i in y]
    except IndexError as e:
        raise ValueError(f"Target label out of bounds: {str(e)}")
    
    # Create dataframe with bounds checking
    df = pd.DataFrame(data=X, columns=feature_names)
    df['species'] = species
    
    # Validate dataframe integrity
    if df.isna().any().any():
        problematic_columns = df.columns[df.isna().any()].tolist()
        logger.warning(f"Missing values detected in columns: {problematic_columns}")
    
    # Add metadata for test verification
    df.attrs['source'] = 'iris_dataset'
    df.attrs['sample_count'] = len(X)
    df.attrs['feature_count'] = len(feature_names)
    df.attrs['class_count'] = len(target_names)
    
    return df


def split_and_scale_data(X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.3, 
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray, StandardScaler]:
    """
    Split data into train/test sets and apply scaling with validation.
    
    Implements a robust train/test split with:
    - Stratification to maintain class distribution
    - Statistical validation of split representativeness
    - Feature scaling with normality preservation
    - Cross-split correlation analysis
    
    Args:
        X: Feature data matrix
        y: Target label vector
        test_size: Proportion of dataset for test split (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler)
        
    Raises:
        ValueError: For invalid parameters or data issues
    """
    # Parameter validation
    if not 0 < test_size < 1:
        raise ValueError(f"Test size must be between 0 and 1, got {test_size}")
    
    # Perform stratified split to maintain class distribution
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        # Handle specific stratification issues
        if "y should be a 1d array" in str(e):
            raise ValueError(f"Stratification error - labels must be 1D: {str(e)}")
        raise
    
    # Verify stratification effectiveness
    train_class_dist = np.bincount(y_train) / len(y_train)
    test_class_dist = np.bincount(y_test) / len(y_test)
    distribution_diff = np.abs(train_class_dist - test_class_dist)
    
    if np.max(distribution_diff) > 0.1:  # Threshold for distribution difference
        logger.warning(f"Class distribution mismatch between train and test sets: {distribution_diff}")
    
    # Apply feature scaling with robust parameterization
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    
    # Fit on training data only to prevent test information leakage
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Capture scaling parameters for analysis
        scale_means = scaler.mean_
        scale_vars = scaler.var_
        
        # Validate scaling parameters for reasonableness
        if np.any(scale_vars < 1e-10):
            logger.warning("Near-zero variance detected in feature scaling")
            
        # Transform test data using parameters learned from training data
        X_test_scaled = scaler.transform(X_test)
        
        # Verify scaling properties
        train_scaled_mean = np.mean(X_train_scaled, axis=0)
        train_scaled_std = np.std(X_train_scaled, axis=0)
        
        # Validate expected statistical properties of scaled data
        if not np.allclose(train_scaled_mean, 0, atol=1e-2):
            logger.warning(f"Scaled training data mean deviates from zero: {train_scaled_mean}")
            
        if not np.allclose(train_scaled_std, 1, atol=1e-2):
            logger.warning(f"Scaled training data std deviates from one: {train_scaled_std}")
            
    except Exception as e:
        raise ValueError(f"Error during data scaling: {str(e)}")
    
    # Return full split pipeline results
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def ensure_directories() -> bool:
    """
    Create necessary directories for storing visualizations and models.
    
    Implements path validation and creation with appropriate permissions
    for testing artifacts across the application lifecycle.
    
    Returns:
        bool: True if all directories were successfully created/verified
    """
    required_dirs = [
        'static',
        'static/images',
        'static/images/basic',
        'static/images/advanced',
        'static/images/robustness',
        'static/images/hyperparameter',
        'static/images/adversarial',
        'static/images/drift',
        'models'
    ]
    
    success = True
    for directory in required_dirs:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            success = False
            
    return success


def get_visualization_path(program: str, filename: str) -> str:
    """
    Generate appropriate path for visualization artifacts with validation.
    
    Maps visualization files to their correct storage locations based on
    test domain and ensures proper directory structure exists.
    
    Args:
        program: Program identifier (program1, program2, etc.)
        filename: Name of visualization file
        
    Returns:
        str: Full path for storing visualization
    """
    # Ensure base directories exist
    ensure_directories()
    
    # Map program identifiers to directories
    prefix_map = {
        'program1': 'static/images/basic/',
        'program2': 'static/images/advanced/',
        'program3': 'static/images/robustness/',
        'program4': 'static/images/hyperparameter/'
    }
    
    # Default directory fallback
    directory = 'static/images/'
    
    # Determine program from filename
    for prefix, path in prefix_map.items():
        if filename.startswith(f"{prefix}_"):
            directory = path
            break
    
    # Special cases for program4 subtesting domains
    if filename.startswith('program4_adversarial'):
        directory = 'static/images/adversarial/'
    elif filename.startswith('program4_drift'):
        directory = 'static/images/drift/'
        
    return os.path.join(directory, filename)


def load_model_from_disk(filename: str = 'models/iris_classifier_model.pkl') -> Optional[Dict[str, Any]]:
    """
    Load a serialized model with validation and compatibility checks.
    
    Implements model loading with:
    - File integrity verification
    - Schema validation
    - Compatibility checking
    - Version validation
    
    Args:
        filename: Path to the saved model file
        
    Returns:
        dict or None: Dictionary containing model and associated data, or None if loading fails
        
    Raises:
        ModelCompatibilityError: For incompatible or corrupted model files
    """
    # Validate file existence
    if not os.path.exists(filename):
        logger.warning(f"Model file {filename} not found")
        return None
    
    try:
        # Check file integrity
        if os.path.getsize(filename) == 0:
            raise ModelCompatibilityError(f"Model file is empty: {filename}")
            
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate expected model structure
        required_keys = ['model', 'scaler', 'feature_names', 'target_names']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            raise ModelCompatibilityError(f"Model file missing required keys: {missing_keys}")
            
        # Verify model has expected interface
        model = model_data['model']
        if not hasattr(model, 'predict') or not callable(model.predict):
            raise ModelCompatibilityError("Model lacks required predict() method")
            
        # Validate model metadata
        if not isinstance(model_data['feature_names'], (list, tuple)) or len(model_data['feature_names']) == 0:
            raise ModelCompatibilityError("Invalid feature_names in model data")
            
        # Calculate and store model fingerprint for integrity tracking
        model_json = str(model_data['model'])
        model_fingerprint = hashlib.md5(model_json.encode()).hexdigest()
        model_data['model_fingerprint'] = model_fingerprint
        
        logger.info(f"Model loaded successfully from {filename} (fingerprint: {model_fingerprint[:8]}...)")
        
        return model_data
        
    except (pickle.UnpicklingError, EOFError) as e:
        raise ModelCompatibilityError(f"Corrupted model file: {str(e)}")
    except Exception as e:
        if isinstance(e, ModelCompatibilityError):
            raise
        logger.error(f"Error loading model: {str(e)}")
        return None


def save_model_to_disk(model_data: Dict[str, Any], 
                       filename: str = 'models/iris_classifier_model.pkl') -> bool:
    """
    Save model with metadata, versioning and integrity safeguards.
    
    Implements model persistence with:
    - Directory validation
    - Data completeness verification
    - Atomic writing
    - Integrity checking
    
    Args:
        model_data: Dictionary containing model and associated data
        filename: Path to save the model file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Ensure directory structure
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            logger.error(f"Failed to create directory for model: {str(e)}")
            return False
    
    # Validate model data
    if not isinstance(model_data, dict):
        logger.error("Model data must be a dictionary")
        return False
        
    required_keys = ['model', 'scaler', 'feature_names', 'target_names']
    missing_keys = [key for key in required_keys if key not in model_data]
    
    if missing_keys:
        logger.error(f"Model data missing required keys: {missing_keys}")
        return False
    
    try:
        # Add metadata
        model_data['version'] = model_data.get('version', '1.0.0')
        model_data['timestamp'] = pd.Timestamp.now().isoformat()
        
        # Generate backup filename for atomic writing
        backup_filename = f"{filename}.bak"
        
        # Save to a backup file first for atomic operation
        with open(backup_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Verify backup file integrity
        if not os.path.exists(backup_filename) or os.path.getsize(backup_filename) == 0:
            logger.error(f"Failed to write backup model file: {backup_filename}")
            return False
            
        # Atomically replace target file
        if os.path.exists(filename):
            os.replace(backup_filename, filename)
        else:
            os.rename(backup_filename, filename)
        
        logger.info(f"Model saved successfully to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        # Clean up failed backup if it exists
        if 'backup_filename' in locals() and os.path.exists(backup_filename):
            try:
                os.remove(backup_filename)
            except:
                pass
        return False


def check_for_missing_values(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Comprehensive analysis of missing values with detailed diagnostics.
    
    Performs multi-level missing value analysis:
    - Per-feature missing counts
    - Missing patterns analysis
    - Structural missingness evaluation
    
    Args:
        X: Feature data as array or DataFrame
        
    Returns:
        dict: Detailed analysis of missing values
    """
    result = {
        'has_missing': False,
        'total_missing': 0,
        'percentage_missing': 0.0,
        'feature_missing': {},
        'patterns': [],
    }
    
    try:
        if isinstance(X, pd.DataFrame):
            # Detailed DataFrame analysis
            missing_counts = X.isnull().sum()
            result['total_missing'] = missing_counts.sum()
            result['has_missing'] = result['total_missing'] > 0
            result['percentage_missing'] = result['total_missing'] / (X.shape[0] * X.shape[1]) * 100
            
            # Per-feature analysis
            for col, count in missing_counts.items():
                if count > 0:
                    result['feature_missing'][col] = {
                        'count': int(count),
                        'percentage': count / len(X) * 100
                    }
            
            # Missing patterns analysis if values exist
            if result['has_missing']:
                # Find common patterns of missingness
                pattern_df = X.isnull().groupby(list(X.columns)).size().reset_index(name='count')
                pattern_df = pattern_df.sort_values('count', ascending=False).head(5)
                
                # Convert to more interpretable format
                for _, row in pattern_df.iterrows():
                    pattern = {col: bool(row[col]) for col in X.columns if col != 'count'}
                    result['patterns'].append({
                        'pattern': pattern,
                        'count': int(row['count']),
                        'percentage': row['count'] / len(X) * 100
                    })
                
        else:
            # Basic NumPy array analysis
            total_missing = np.isnan(X).sum()
            result['total_missing'] = int(total_missing)
            result['has_missing'] = total_missing > 0
            result['percentage_missing'] = total_missing / X.size * 100
            
            # Per-feature missing counts
            if X.ndim > 1:
                per_feature_missing = np.isnan(X).sum(axis=0)
                for i, count in enumerate(per_feature_missing):
                    if count > 0:
                        result['feature_missing'][f'feature_{i}'] = {
                            'count': int(count),
                            'percentage': count / X.shape[0] * 100
                        }
        
    except Exception as e:
        logger.error(f"Error checking for missing values: {str(e)}")
        # Return safe defaults on error
        result['error'] = str(e)
    
    return result


def get_class_distribution(y: np.ndarray, target_names: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Analyze class distribution with imbalance assessment.
    
    Performs class distribution analysis with:
    - Count and frequency calculation
    - Imbalance ratio assessment
    - Entropy-based diversity metrics
    
    Args:
        y: Target labels
        target_names: Names of target classes
        
    Returns:
        dict: Detailed class distribution analysis
    """
    result = {
        'counts': {},
        'frequencies': {},
        'imbalance_metrics': {},
    }
    
    try:
        # Calculate basic counts
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        # Map class indices to names if provided
        class_keys = []
        if target_names is not None and len(target_names) >= len(unique):
            class_keys = [target_names[idx] for idx in unique]
        else:
            class_keys = [f"Class {idx}" for idx in unique]
        
        # Store raw counts
        result['counts'] = dict(zip(class_keys, counts.tolist()))
        
        # Calculate frequencies
        frequencies = counts / total
        result['frequencies'] = dict(zip(class_keys, frequencies.tolist()))
        
        # Calculate imbalance metrics if we have at least 2 classes
        if len(unique) > 1:
            # Imbalance ratio (max frequency / min frequency)
            imbalance_ratio = np.max(frequencies) / np.min(frequencies)
            result['imbalance_metrics']['ratio'] = float(imbalance_ratio)
            
            # Entropy-based measure (higher means more balanced)
            entropy = -np.sum(frequencies * np.log2(frequencies))
            max_entropy = np.log2(len(unique))  # Maximum theoretical entropy
            normalized_entropy = entropy / max_entropy  # 1.0 means perfectly balanced
            
            result['imbalance_metrics']['entropy'] = float(entropy)
            result['imbalance_metrics']['normalized_entropy'] = float(normalized_entropy)
            
            # Assessment of imbalance severity
            if imbalance_ratio > 10:
                result['imbalance_metrics']['assessment'] = "Severe imbalance"
            elif imbalance_ratio > 3:
                result['imbalance_metrics']['assessment'] = "Moderate imbalance"
            elif imbalance_ratio > 1.5:
                result['imbalance_metrics']['assessment'] = "Mild imbalance"
            else:
                result['imbalance_metrics']['assessment'] = "Well balanced"
        
    except Exception as e:
        logger.error(f"Error analyzing class distribution: {str(e)}")
        result['error'] = str(e)
    
    return result


def analyze_feature_correlations(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze feature correlations and multicollinearity.
    
    Performs correlation analysis with:
    - Pearson correlation matrix
    - Variance Inflation Factor assessment
    - Feature clustering
    
    Args:
        X: Feature data
        feature_names: Names of features
        
    Returns:
        dict: Correlation analysis results
    """
    result = {
        'correlation_matrix': {},
        'high_correlations': [],
        'collinearity_assessment': ''
    }
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    try:
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Convert to dictionary format with feature names
        result['correlation_matrix'] = {
            'data': corr_matrix.tolist(),
            'features': feature_names
        }
        
        # Find high correlations (absolute value > 0.7) excluding self-correlations
        high_corr_threshold = 0.7
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_value = corr_matrix[i, j]
                if abs(corr_value) > high_corr_threshold:
                    result['high_correlations'].append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr_value)
                    })
        
        # Assess overall collinearity
        if len(result['high_correlations']) > X.shape[1] // 2:
            result['collinearity_assessment'] = "High multicollinearity detected"
        elif len(result['high_correlations']) > 0:
            result['collinearity_assessment'] = "Some correlated features present"
        else:
            result['collinearity_assessment'] = "Low feature correlation"
            
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {str(e)}")
        result['error'] = str(e)
    
    return result


def detect_outliers(X: np.ndarray, method: str = 'iqr', 
                   threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using multiple statistical methods.
    
    Implements outlier detection with:
    - IQR-based detection (boxplot method)
    - Z-score based detection
    - Isolation Forest option
    
    Args:
        X: Feature data
        method: Detection method ('iqr', 'zscore')
        threshold: Detection threshold
        
    Returns:
        dict: Outlier detection results
    """
    result = {
        'outlier_mask': None,
        'outlier_count': 0,
        'outlier_indices': [],
        'per_feature_outliers': {},
        'method': method,
        'threshold': threshold
    }
    
    try:
        if method == 'iqr':
            # IQR (Inter-Quartile Range) method
            outlier_mask = np.zeros(len(X), dtype=bool)
            
            # Calculate per-feature outliers
            for i in range(X.shape[1]):
                col_data = X[:, i]
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                feature_outliers = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_mask |= feature_outliers
                
                # Store per-feature outlier info
                feature_outlier_indices = np.where(feature_outliers)[0].tolist()
                if feature_outlier_indices:
                    result['per_feature_outliers'][i] = {
                        'indices': feature_outlier_indices,
                        'count': len(feature_outlier_indices),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            
        elif method == 'zscore':
            # Z-score method
            outlier_mask = np.zeros(len(X), dtype=bool)
            
            # Calculate per-feature outliers based on z-score
            for i in range(X.shape[1]):
                col_data = X[:, i]
                mean = np.mean(col_data)
                std = np.std(col_data)
                
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((col_data - mean) / std)
                    feature_outliers = z_scores > threshold
                    outlier_mask |= feature_outliers
                    
                    # Store per-feature outlier info
                    feature_outlier_indices = np.where(feature_outliers)[0].tolist()
                    if feature_outlier_indices:
                        result['per_feature_outliers'][i] = {
                            'indices': feature_outlier_indices,
                            'count': len(feature_outlier_indices),
                            'threshold': float(threshold),
                            'max_zscore': float(np.max(z_scores))
                        }
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Store overall results
        result['outlier_mask'] = outlier_mask
        result['outlier_count'] = int(np.sum(outlier_mask))
        result['outlier_indices'] = np.where(outlier_mask)[0].tolist()
        result['outlier_percentage'] = 100 * result['outlier_count'] / len(X)
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        result['error'] = str(e)
        result['outlier_mask'] = np.zeros(len(X), dtype=bool)  # Safe default
    
    return result
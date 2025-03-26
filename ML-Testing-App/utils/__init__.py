# utils/__init__.py

"""
Utilities package for ML Testing Dashboard
------------------------------------------
Contains data handling and visualization modules
"""

from .data_handler import (
    load_iris_dataset,
    create_dataframe,
    split_and_scale_data,
    load_model_from_disk,
    save_model_to_disk,
    check_for_missing_values,
    get_class_distribution
)

from .visualization import (
    get_plot_as_base64,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_boundaries,
    plot_performance_comparison,
    plot_roc_curve,
    plot_outliers,
    plot_feature_distributions,
    plot_pairplot
)
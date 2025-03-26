# ML Testing Dashboard | Attempt by a Tester for Testers

A comprehensive testing framework for machine learning models, focusing on KNN classification with the Iris dataset.

## Features

This dashboard provides multiple testing approaches:

- **Basic Testing**: Model training, evaluation, and visualization
- **Advanced Testing**: Cross-validation and model explainability
- **Robustness Testing**: Performance with outliers and edge cases
- **Hyperparameter Optimization**: Finding optimal model parameters
- **Adversarial Testing**: Testing model vulnerability to adversarial examples
- **Data Drift Simulation**: Assessing model performance under different data distributions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setting Up the Environment

```bash
# Clone this repository
git clone https://github.com/yourusername/ml-testing-dashboard.git
cd ml-testing-dashboard

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

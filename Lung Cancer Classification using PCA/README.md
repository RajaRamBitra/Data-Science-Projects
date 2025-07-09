# 🫁 Lung Cancer PCA Classification

> **Comprehensive machine learning pipeline for lung cancer classification using microRNA data with advanced dimensionality reduction techniques**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-Mathematical%20Computing-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Information](#-dataset-information)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [📖 Usage Guide](#-usage-guide)
- [🔬 Methodology](#-methodology)
- [📈 Key Results](#-key-results)
- [💻 Technologies Used](#-technologies-used)
- [🤝 Contributing](#-contributing)
- [👨‍💻 Author](#-author)
- [📄 License](#-license)

### Project Overview

This project implements a comprehensive machine learning pipeline for lung cancer classification using microRNA expression data. The analysis includes **from-scratch implementations** of Principal Component Analysis (PCA), multiple kernel PCA variants, and various classification algorithms, providing both educational value and practical insights into biomedical data analysis.

### Key Features

- **From-Scratch Implementations**: Custom PCA, Kernel PCA, and classification algorithms
- **Comprehensive Analysis**: 4 detailed Jupyter notebooks covering the entire ML pipeline
- **Multiple Techniques**: Standard PCA, Kernel PCA (RBF, Polynomial, Linear, Combined)
- **Various Classifiers**: Minimum Distance, Bayes, Naive Bayes, KNN, LDA
- **Performance Optimization**: Component selection and hyperparameter tuning
- **Detailed Visualizations**: Performance comparisons and analysis plots

### Dataset Information

- **Dataset**: `Lung.csv`
- **Type**: microRNA expression data
- **Samples**: 1,091 lung tissue samples
- **Features**: 1,881 microRNA expression levels
- **Target**: Binary classification (cancer vs normal)
- **Train/Test Split**: 872/219 samples (80/20)

### Data Characteristics
- High-dimensional biomedical data
- Requires dimensionality reduction for effective analysis
- Suitable for demonstrating advanced ML techniques

### Project Structure

```
lung-cancer-pca-classification/
├── README.md                                   # Project documentation
├── Lung.csv                                    # Dataset file
└── notebooks/                                  # Main analysis notebooks
    ├── 01_Principal_Component_Analysis.ipynb   # PCA implementation & comparison
    ├── 02_Kernel_PCA_Implementation.ipynb      # Advanced kernel methods  
    ├── 03_Classification_and_Evaluation.ipynb  # Multiple classifiers
    └── 04_Optimization_and_Analysis.ipynb      # Performance optimization
```

### Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/lung-cancer-pca-classification.git
cd lung-cancer-pca-classification
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 4. Run the Notebooks
Start with `01_Principal_Component_Analysis.ipynb` and proceed sequentially through all four notebooks.

## 📖 Usage Guide

### Sequential Analysis Workflow

1. **Notebook 1**: Principal Component Analysis
   - Load and explore the lung cancer dataset
   - Implement PCA from scratch with detailed explanations
   - Compare with scikit-learn implementation
   - Visualize variance retention and component selection

2. **Notebook 2**: Kernel PCA Implementation
   - Implement multiple kernel variants (RBF, Polynomial, Linear)
   - Develop combined kernel approaches
   - Compare kernel performance characteristics

3. **Notebook 3**: Classification and Evaluation
   - Implement multiple classification algorithms
   - Evaluate performance on reduced-dimension data
   - Compare classifier effectiveness

4. **Notebook 4**: Optimization and Analysis
   - Systematic component optimization (1-30 components)
   - Comprehensive performance analysis
   - Generate final visualisations

### Quick Start Example
```python
# Load the data
import pandas as pd
import numpy as np

df = pd.read_csv('Lung.csv')
data = df.iloc[:,:-1].to_numpy()
labels = df.iloc[:, -1].to_numpy()

print(f"Dataset shape: {data.shape}")
print(f"Number of classes: {len(np.unique(labels))}")
```

### Methodology

### Dimensionality Reduction Techniques

1. **Standard PCA**
   - Eigendecomposition approach
   - Variance threshold-based component selection (95%)
   - Comparison with scikit-learn implementation

2. **Kernel PCA Variants**
   - **RBF Kernel**: For non-linear pattern capture
   - **Polynomial Kernel**: For polynomial relationships
   - **Linear Kernel**: Baseline comparison
   - **Combined Kernels**: Weighted kernel combination

### Classification Algorithms

- **From-Scratch Implementations**:
  - Minimum Distance Classifier
  - Bayes Classifier with Gaussian assumptions

- **Scikit-learn Implementations**:
  - Naive Bayes
  - K-Nearest Neighbors (k=5)
  - Linear Discriminant Analysis

### Optimization Strategy

- Component range testing (1-30 components)
- Cross-validation for robust evaluation
- Performance vs computational efficiency analysis

### Key Results

### PCA Performance
- **From-scratch PCA**: 9 components, 95% variance retention, 7.2s processing time
- **Scikit-learn PCA**: 8 components, exact 95% variance, 1.6s processing time
- **Efficiency**: Scikit-learn 4.5x faster due to SVD optimization

### Classification Performance
- **Best Accuracy**: Linear Discriminant Analysis with optimized components
- **Most Robust**: Naive Bayes across different component numbers
- **Optimal Range**: 5-15 components for most classifiers

### Optimization Insights
- Diminishing returns beyond 20 components
- Classifier-specific optimal component numbers
- Trade-offs between accuracy and computational efficiency

### Technologies Used

### Core Libraries
- **NumPy**: Numerical computing and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and metrics

### Development Environment
- **Jupyter Notebook**: Interactive development and documentation
- **Python 3.8+**: Primary programming language

### Key Algorithms Implemented
- Principal Component Analysis (eigendecomposition)
- Kernel PCA (multiple kernel types)
- Various classification algorithms
- Performance optimization techniques

### Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution
- Additional kernel implementations
- More classification algorithms
- Enhanced visualization techniques
- Performance optimizations
- Documentation improvements

### Author

**Raja Ram Bitra**
- 📧 Email: [rambitra01@gmail.com]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐱 GitHub: [Your GitHub Profile]

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Educational Value

This project serves as an excellent resource for:

- **Students** learning dimensionality reduction techniques
- **Researchers** working with biomedical data
- **Data Scientists** implementing ML pipelines from scratch
- **Practitioners** optimizing classification performance

### Keywords

`machine-learning` `pca` `kernel-pca` `classification` `biomedical-data` `microRNA` `lung-cancer` `dimensionality-reduction` `python` `jupyter-notebook` `scikit-learn` `data-science`

---

⭐ **If you found this project helpful, please give it a star!** ⭐ 

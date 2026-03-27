# Understanding How Kernel Choice Affects Decision Boundaries

# Overview

This project investigates how different **Support Vector Machine (SVM) kernels** influence classification performance across datasets with varying complexity.

We compare:

* Linear
* Polynomial
* Radial Basis Function (RBF)
* Sigmoid

The study includes both **synthetic datasets** and a **real-world medical dataset**, focusing on performance, decision boundaries, and model tuning.


# Objectives

* Compare SVM kernels across different data patterns
* Visualize decision boundaries
* Evaluate the effect of **feature scaling**
* Analyze **hyperparameter tuning (C & gamma)**
* Provide practical recommendations


# Datasets

#  Synthetic Data

| Dataset            | Description               | Purpose           |
| ------------------ | ------------------------- | ----------------- |
| Linearly Separable | Straight-line separation  | Baseline          |
| Moons              | Interleaving half-circles | Non-linear        |
| Circles            | Concentric classes        | Highly non-linear |

# Real Data

* Breast Cancer Wisconsin Dataset
* 569 samples, 30 features
* Binary classification (malignant vs benign)


# Methodology

python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


* Model: `SVC` (Support Vector Classifier)
* Train/Test Split:

   Synthetic: 70/30
   Real Data: 80/20
  
* Preprocessing:

  StandardScaler (feature scaling)
  
* Validation:

   5-fold cross-validation


#  Results Summary

#  Synthetic Data Performance

| Dataset | Linear | Polynomial | RBF      | Sigmoid |
| ------- | ------ | ---------- | -------- | ------- |
| Linear  | 0.84   | 0.83       | 0.86     | 0.72    |
| Moons   | 0.87   | 0.93       | **0.99** | 0.65    |
| Circles | 0.56   | 0.65       | **1.00** | 0.59    |

 **RBF consistently outperforms other kernels**


#  Real Dataset (Breast Cancer)

| Kernel     | Test Accuracy |
| ---------- | ------------- |
| Linear     | 97.37%        |
| Polynomial | 91.23%        |
| RBF        | **98.25%**    |
| Sigmoid    | 92.98%        |

 **RBF achieves the best overall performance**


#  Feature Scaling Impact

* Essential for SVM performance
* Prevents large-scale features from dominating
* Improves stability (especially for sigmoid kernel)


#  Hyperparameter Tuning

# Key Parameters:

 **C (Regularization)**

  * High → overfitting
  * Low → smoother decision boundary

 **Gamma (RBF kernel)**

  * High → overfitting
  * Low → underfitting

#  Recommended Starting Values:

python
C = 1
gamma = 0.01  # to 0.1 range


#  Learning Curve Insights

* Training accuracy remains high
* Validation improves with more data
* Small gap → good generalization

# Practical Recommendations

* Start with **RBF kernel**
* Always apply **feature scaling**
* Use **cross-validation** for tuning
* Monitor support vectors
* Use **linear kernel** for high-dimensional data


#  Limitations

* Hard to interpret
* Expensive for large datasets
* Sensitive to scaling
* Requires tuning


#  Ethical Considerations

* Limited interpretability in critical domains
* Risk of bias from unrepresentative data
* Use explainability tools (e.g., SHAP, LIME)


# Conclusion

* **RBF is the most versatile kernel**
* Feature scaling is essential
* Proper tuning of **C and gamma** is critical
* Best suited for **small–medium datasets**


# References

* Géron, A. (2022) – *Hands-On Machine Learning*
* Hsu et al. (2003) – *SVM Practical Guide*
* Molnar, C. (2022) – *Interpretable ML*
* Raschka & Mirjalili (2019) – *Python ML*

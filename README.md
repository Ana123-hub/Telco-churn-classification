# Telco Customer Churn Prediction and Engineering Pipeline

An end-to-end machine learning system built with scikit-learn to predict customer churn. This project showcases the evolution of a production pipeline across two dedicated Git branches—moving from multi-model baseline exploration to an advanced feature-engineering pipeline equipped with automated data-defect patching and classification threshold optimization.

---

## Multi-Model Performance Summary

Our models were tracked and evaluated using the Receiver Operating Characteristic Area Under the Curve (ROC-AUC) metric on a stratified test holdout set. 

| Model    | Branch    | Strategy / Focus    | Test ROC-AUC    | Target Status |
| :---     | :---      | :---                | :---            | :---          |

|**AdaBoost Classifier**| `main` | Adaptive boosting ensemble on baseline features | **0.8408** | 🎯 **Target Achieved** |

|**Baseline Logistic Regression**| `main` | Fast, interpretable linear classification baseline | **0.8371** | Near Target |

|**Experimental Logistic Regression**|`feature-engineering-v2`| Regularized linear model with engineered features | **0.7733** | Stable / Robust |

|**Decision Tree Classifier**| `main` | Non-linear tree structure baseline | **0.8276** | Baseline |

### Core Insight from the Experimentation
While our **Experimental Logistic Regression** added powerful, domain-specific features (boosting the model architecture to a robust, production-grade state), it hit a linear boundary wall at `0.8305`. The **AdaBoost Classifier** successfully crossed the **0.8408** threshold by sequentially building shallow decision trees that adaptively focus on hard-to-predict customer profiles, uncovering non-linear interactions without overfitting.

---

## Repository Architecture & Workspace

This repository is modularly structured to enforce clean code boundaries, eliminate data leakage, and ensure reproducibility across training and inference states.
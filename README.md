# Loan Default Prediction with Machine Learning

This project is an end-to-end case study focused on predicting loan defaults using classification models. It includes the complete machine learning pipeline: data loading and cleaning, exploratory data analysis, feature engineering, model training, evaluation, and hyperparameter tuning. Both logistic regression and random forest classifiers are implemented and compared.

A key challenge addressed in the project is class imbalance—defaults represent only ~21% of the dataset. Several strategies were applied to handle this, including class weighting (`balanced` and manual), resampling (upsampling/downsampling), and synthetic oversampling (SMOTE). Model performance was evaluated using metrics like accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices. The best performing model was a downsampled random forest, which provided improved recall while maintaining reasonable precision and interpretability.

## Structure
- `Extract Transform and Load (ETL) Data CreditRiskML.ipynb`: Load and preprocess the data
- `Exploratory_data_analysis_(EDA)_CreditRiskML.ipynb`: Exploratory data analysis
- `Feature_Engineering_CreditRiskML.ipynb`: Feature transformations and encoding
- `modelA_Logistic_Regression_CreditRiskML.ipynb`: Logistic regression model
- `modelA_Logistic_Regression_CreditRiskML.ipynb`: Custom model evaluation metrics
- `Random_Forest_classification_CreditRiskML.ipynb`: Random forest with hyperparameter tuning
- `Accuracy_and_Class_balancing_CreditRiskML.ipynb`: Handling class imbalance (weights, resampling, SMOTE)

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

## Outcome
By iterating through multiple balancing and modeling techniques, the study demonstrates how thoughtful preprocessing and evaluation can significantly enhance classification performance in imbalanced datasets.

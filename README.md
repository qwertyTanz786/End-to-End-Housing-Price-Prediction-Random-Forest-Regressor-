# California Housing Price Prediction using Random Forest Regressor

## Project Overview

This project is a Machine Learning regression model that predicts the median house value based on housing features such as median income, house age, total rooms, population, and more.

The objective is to build an end-to-end regression pipeline using the Random Forest Regressor and evaluate its performance using standard regression metrics.

---

## Dataset Information

- Dataset: California Housing Dataset
- Target Variable: median_house_value
- Problem Type: Regression

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---

## Steps Performed

1. Data Loading
2. Data Cleaning
   - Dropped categorical column (ocean_proximity)
   - Handled missing values using median
3. Feature and Target Separation
4. Train-Test Split (80/20)
5. Model Training using RandomForestRegressor
6. Model Evaluation
7. Model Saving using Joblib

---

## Model Used

Random Forest Regressor

Random Forest is an ensemble learning algorithm that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.

---

## Results

### Model Performance

R² Score: 0.81

Root Mean Squared Error (RMSE):  49884.52


---
---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Cross-validation implementation

---

## Author

Tanishq Panchal  
Computer Science Student | Aspiring Data Scientist

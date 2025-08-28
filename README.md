Understanding Lasso Regression
This project explores Lasso Regression, a linear regression technique with L1 regularization, to understand its impact on model performance and overfitting. The project uses synthetic datasets to demonstrate how the regularization parameter (alpha) affects the model's slope and fit.
Project Overview

Libraries Used: numpy, pandas, matplotlib, scikit-learn (Lasso, make_regression, train_test_split, LinearRegression)
Objective: To analyze the effect of Lasso Regression's alpha parameter on model fit, including overfitting and underfitting scenarios, using synthetic datasets.
Key Concepts: Linear regression, Lasso regression, regularization, overfitting, underfitting.

Dataset 1: Linear Data
Data Generation
A synthetic linear dataset was created using make_regression from scikit-learn:

Parameters:
n_samples=100
n_features=1
n_informative=1
n_targets=1
noise=20
random_state=13



Data Splitting
The dataset was split into training and testing sets using train_test_split from scikit-learn.
Linear Regression Model
A linear regression model was trained on the dataset, yielding:

Coefficient: 27.9522
Intercept: -2.35

Lasso Regression Analysis
Lasso Regression models were trained with different alpha values (0, 1, 5, 10, 30) to observe the effect of regularization:

At alpha=0, the model behaves like linear regression (no regularization).
As alpha increases, the slope decreases, reducing overfitting.
At alpha=30, the slope becomes nearly flat, approaching the mean of the target variable (y_mean), indicating underfitting.

Visualization
The plot below shows the effect of different alpha values on the Lasso Regression model's slope:
<img width="822" height="505" alt="download" src="https://github.com/user-attachments/assets/e19dba14-f990-4187-ba82-daf9951101ce" />

Dataset 2: Non-Linear Data
Data Generation
A second synthetic dataset with a non-linear relationship was created:

Parameters:
m=100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)



Data Visualization
A scatter plot of the data shows a curved, non-linear relationship:
<img width="543" height="413" alt="download (1)" src="https://github.com/user-attachments/assets/fdc48e56-d7b4-4ed4-b964-c0880720b9ae" />

Lasso Regression Analysis
Lasso Regression models were trained on this dataset with alpha values (0, 0.1, 1):

At alpha=0, the model overfits, closely following the curved data.
At alpha=0.1, the model achieves a balanced fit.
At alpha=1, the model starts to underfit, losing the ability to capture the non-linear relationship.

Visualization
The plot below illustrates the effect of different alpha values on the Lasso Regression model's fit for the non-linear dataset:
<img width="988" height="505" alt="download (2)" src="https://github.com/user-attachments/assets/85529138-04ec-4905-a632-3b067188dc3a" />

Key Observations

Dataset 1 (Linear): Increasing alpha reduces overfitting but can lead to underfitting, as seen with a flat slope at alpha=30.
Dataset 2 (Non-Linear): Lasso Regression struggles with non-linear data. Small alpha values overfit, while larger values cause underfitting.

How to Run

Install required libraries:pip install numpy pandas matplotlib scikit-learn


Clone this repository.
Run the Python script containing the code for data generation, model training, and visualization.
Replace placeholder image tags with actual plot images.

Future Work

Explore other regularization techniques like Ridge Regression or Elastic Net.
Test polynomial features to better fit non-linear data with Lasso Regression.
Experiment with a wider range of alpha values.

License
This project is licensed under the MIT License.

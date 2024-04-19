Air Quality Prediction in Beijing
---------------------------------
MEC E 788 LEC X51 - Winter 2024 - Applied Machine Learning Course Project
Group 2
---------------------------------
Overview:
This project applies machine learning techniques to perform temporal analysis on air qulity dataset for beijing. The goal is to predict pollutants in the futures and Air qulity health index using different methods from simple regressors to Bi-LSTM and GRU. This README provides a general overview of the project, instructions on how to set up, train, and evaluate the models, and guidance on the organization of the codebase.
---------------------------------
Project Outline:
I. Initial EDA
II. Simple Models
III. Complex Models
---------------------------------
I. Initital EDA
Step 1: adding all the datasets to the notebook directy
Step 2: Run the notebook.
All the blocks would run automatically.
The first part belongs to analyzing the dataset including yearly, monthly, and hourly analysis. Also stations are compared too.
Second part is using cos and sin transformations and the AQHI as a new feature is added. Then, outliers were detected and interpolated by the adjacent datapoint
In the third part the cyclic function was utlized for datatime, and eventually the weekends, seasons and holidays were added.

--------------------------------
II. Simple Models

Linear Regression, Lasso Regression, Linear Regression with Backward Elimination
Linear Regression with PCA, Random Forest, XGBoost, Support Vector Regressors

step 1: Importing all the necessary libraries
step 2: Uploading the dataset
Step 3: Additional Feature Extraction
Step 4: Time Series Component
Step 5: Target Definition
Step 6: Grid Search with Blocked Cross Validation
Step 7: PCA
Step 8: Model Comparison

BEST SIMPLE MODEL: Linear Regression with Backward Elimination.

Evaluation Metrics for the best simple model

RMSE: 0.541

--------------------------------
III. Complex Models

1.RNN and LSTM:
step 1: importing all the necessary libraries
step 2: uploading the dataset
step 3: defining all the functions required for performing grid searchs
step 4: for LSTM, running a grid search to figure out the best combination of past data needed and future timestep to make prediction while keeping the hyperparameters fixed.
step 5: for LSTM, figuring out the best set of hyperparameters through a grid search for making prediction using the chosen number of past timesteps and prediction time. 
step 6: for RNN, running a grid search to figure out the best combination of past data needed and future timestep to make prediction while keeping the hyperparameters fixed.
step 7: for RNN, figuring out the best set of hyperparameters through a grid search for making prediction using the chosen number of past timesteps and prediction time. 

2.Bi-LSTM and GRU:
step 1: importing all the necessary libraries
step 2: uploading the dataset
step 3: defining all the functions required for performing grid searchs
step 4: for BiLSTM, running a grid search to figure out the best combination of past data needed and future timestep to make prediction while keeping the hyperparameters fixed.
step 5: for BiLSTM, figuring out the best set of hyperparameters through a grid search for making prediction using the chosen number of past timesteps and prediction time. 
step 6: for GRU, running a grid search to figure out the best combination of past data needed and future timestep to make prediction while keeping the hyperparameters fixed.
step 7: for GRU, figuring out the best set of hyperparameters through a grid search for making prediction using the chosen number of past timesteps and prediction time. 


BEST MODEL: RNN (Using 3 hrs of past data and predicting 1 hr into the future)

Model Specifications

Number of features: 16
Train, Validation & Test data ratio = 7:2:1
Number of Hidden Layers: 2 
Activation Functions: ReLu, ReLu,Linear
Optimizer: Adam (Adaptive Moment Estimation)
Epochs: 100 (Patience level for Early Stopping criterion: 10)
RNN Units(1st hidden layer): 64
Fully connected nodes(2nd Hidden Layer): 128
Learning Rate: 0.001
Batch size: 32

Evaluation Metrics for the best model

RMSE: 0.536
MAE:  0.302
R2:  0.947


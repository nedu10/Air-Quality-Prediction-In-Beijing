# Air Quality Prediction in Beijing
**MEC E 788 LEC X51 - Winter 2024 - Applied Machine Learning Course Project**  
**Group 2**

## Overview:
This project applies machine learning techniques to perform temporal analysis on air quality datasets for Beijing. The goal is to predict pollutants in the future and Air Quality Health Index (AQHI) using different methods from simple regressors to Bi-LSTM and GRU. This README provides a general overview of the project, instructions on how to set up, train, and evaluate the models, and guidance on the organization of the codebase.

## Description of Folder Structure:
I. Initial Exploratory Data Analysis (EDA): 
   This file documents all the initial exploratory data analysis conducted on the dataset.

II. Feature Engineering and Simple Models: 
   This file includes the feature engineering process applied to generate a new dataset used for evaluating simple models. It also houses the implementation of simple models on the dataset.

III. RNN and LSTM (Complex Models): 
   Due to its size, this file cannot be added directly to GitHub. You can access it via Google Colab using (https://drive.google.com/file/d/1USu-qstOnkWqO2Zn2pL-IJ2GQD8SDmD0/view?usp=sharing).

IV. Bi-LSTM and GRU (Complex Models): 
   Similar to the RNN and LSTM file, this one cannot be added directly to GitHub due to size limitations. You can access it through Google Colab using (https://drive.google.com/file/d/1vPWtkZsJg-CmKTJ7xh7f-Oan3JoKyNsK/view?usp=sharing).

V. Model Cards: 
   Two model cards are available—one for the best simple model and the other for the best complex model. You can find them in the Model Cards Folder.

VI. Best Model Folder: 
   This folder contains two files—one for the best simple model and the other for the best complex model.

VII. Dataset Folder: 
   Contains the initial dataset used in the project.


## Instruction
To run the models, navigate to the Best Models folder and follow the instructions provided in the notebook files.

## Project Outline:
I. Initial EDA  
II. Simple Models  
III. Complex Models  

## I. Initial EDA
1. Adding all the datasets to the notebook directly.
2. Run the notebook; all blocks will run automatically.
   - Analyzing the dataset including yearly, monthly, and hourly analysis. Stations are also compared.
   - Using cosine and sine transformations, adding AQHI as a new feature. Outliers were detected and interpolated by the adjacent data point.
   - Utilizing the cyclic function for datetime, and adding weekends, seasons, and holidays.

## II. Simple Models
- Linear Regression, Lasso Regression, Linear Regression with Backward Elimination
- Linear Regression with PCA, Random Forest, XGBoost, Support Vector Regressors

1. Importing all necessary libraries.
2. Uploading the dataset.
3. Additional Feature Extraction.
4. Time Series Component.
5. Target Definition.
6. Grid Search with Blocked Cross Validation.
7. PCA.
8. Model Comparison.

**Dataset After For Simple Models.**
X_train: https://drive.google.com/file/d/1g-WchMrKqXkk8P9yY7Qk1v4MnMg5zXHo/view?usp=drive_link
Y_train: https://drive.google.com/file/d/1uCkugZ30YCAL3CyEj4vLRyR4KU-WH7dj/view?usp=drive_link
X_test: https://drive.google.com/file/d/1yU6uCm8YYGd6ok_Ox0O8TwLLzxdshOPQ/view?usp=drive_link
Y_test: https://drive.google.com/file/d/1dOm6vCVqncJUnPGvhlU6tlrt3GnVA4SK/view?usp=drive_link

**Best Simple Model: Linear Regression with Backward Elimination.**

Evaluation Metrics for the best simple model:
- RMSE: 0.541

## III. Complex Models
1. **RNN and LSTM:**
   - Importing all necessary libraries.
   - Uploading the dataset.
   - Defining all functions required for performing grid searches.
   - For LSTM, running a grid search to find the best combination of past data needed and future time step for prediction, keeping hyperparameters fixed.
   - For LSTM, finding the best set of hyperparameters through a grid search for making predictions using the chosen number of past timesteps and prediction time.
   - For RNN, similar grid search process as for LSTM.
   
2. **Bi-LSTM and GRU:**
   - Importing all necessary libraries.
   - Uploading the dataset.
   - Defining all functions required for performing grid searches.
   - For BiLSTM, running a grid search to find the best combination of past data needed and future time step for prediction, keeping hyperparameters fixed.
   - For BiLSTM, finding the best set of hyperparameters through a grid search for making predictions using the chosen number of past timesteps and prediction time.
   - For GRU, similar grid search process as for BiLSTM.

**Best Model: RNN (Using 3 hrs of past data and predicting 1 hr into the future)**

### Model Specifications:
- Number of features: 16
- Train, Validation & Test data ratio = 7:2:1
- Number of Hidden Layers: 2
- Activation Functions: ReLu, ReLu, Linear
- Optimizer: Adam (Adaptive Moment Estimation)
- Epochs: 100 (Patience level for Early Stopping criterion: 10)
- RNN Units (1st hidden layer): 64
- Fully connected nodes (2nd Hidden Layer): 128
- Learning Rate: 0.001
- Batch size: 32

Evaluation Metrics for the best model:
- RMSE: 0.536
- MAE: 0.302
- R2: 0.947

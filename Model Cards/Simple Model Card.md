# Model Card: Simple Model for Air Quality Prediction

## Model Details

- **Model Name:** Linear Regression with Backward Elimination
- **Model Version:** 1.0
- **Model Type:** Regression
- **Developers:** Chinedu Ifediorah, Faiyaz Bin Naser Chowdhury, Amirhossein Iranmehr, Muhammad Talha

- **Release Date:** 19th April 2024

## Intended Use

- **Primary Use:** Time series forecasting for air quality in Beijing.
- **Intended Users:** Environmental researchers, policymakers, and stakeholders interested in air quality monitoring.
- **Out-of-Scope Use Cases:** Real-time air quality monitoring or emergency response systems.

## Model/Data Description

- **Data Used:** Air quality data from Beijing, including pollutants, meteorological factors, and temporal information. Sources include government databases and environmental monitoring stations.
- **Features:** Pollutant concentrations (e.g., PM2.5, PM10, CO, NO2, S02, 03), meteorological variables (e.g., temperature, Pressure, Wind Direction, etc), time of day, and day of week.
- **Model Architecture:** Linear regression with backward elimination to select significant features and optimize predictive accuracy.

## Training and Evaluation

- **Training Procedure:** Model trained using historical air quality data from Beijing (e.g., hourly measurements, Lag Values, Rolling Features, Seasonnal Features etc) with backward elimination to remove non-significant features. Training environment includes Python with libraries like pandas, scikit-learn, and statsmodels.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) for model evaluation on test datasets.
- **Baseline Comparison:** More complex models (e.g., RNN) for accuracy and computational efficiency.

## Ethical Considerations

- **Fairness and Bias:** Measures implemented to mitigate biases in the training data and model predictions, particularly concerning outliers resulting from unforeseen impacts at the monitoring stations.
- **Privacy:** Data anonymization and compliance with privacy regulations to protect individual data privacy in the training dataset.
- **Security:** Data encryption, access controls, and secure storage practices to prevent unauthorized access to sensitive air quality data.

## Limitations and Recommendations

- **Known Limitations:** Decreased accuracy for long-term predictions (e.g., 24 hours) compared to complex models like RNN due to linear regression's simplicity and assumptions.
- **Recommendations for Use:** Use for short-term (1-8 hour) air quality forecasts; consider ensemble methods or complex models for improved accuracy in longer-term predictions.

## Additional Information

- **References:** [List of references to research papers, datasets, and relevant documentation]
- **License:** [License information for model usage and redistribution]
- **Contact Information:** cifedior@ualberta.ca




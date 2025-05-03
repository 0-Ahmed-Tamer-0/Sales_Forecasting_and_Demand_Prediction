#  Sales and Demand Forecasting using Machine Learning & Deep Learning

This project aims to build predictive models for forecasting **retail sales (Units Sold)** and **demand (Demand Forecast)** using both **traditional machine learning models**, **deep learning models**, and **time series forecasting techniques**.

Effective forecasting of sales and demand is essential for businesses to optimize inventory management, improve customer satisfaction, and streamline their supply chain.

---

##  About the Dataset

The dataset contains over **73,000 daily records** across multiple stores and products. It includes the following features:

- **Sales** (Units Sold)
- **Demand** (Demand Forecast)
- **Inventory levels**
- **Pricing**
- **Weather data**
- **Promotions**
- **Holidays**
- **Product and store information**
- **Date/Time** features

The dataset is ideal for tasks such as:

- Demand and sales forecasting
- Time series analysis
- Dynamic pricing strategies
- Inventory optimization

---

##  Project Objectives

The main objectives of this project include:

1. **Predicting future Units Sold** and **Demand Forecast** for different products across stores.
2. **Evaluating a wide range of machine learning, deep learning, and time series models** for both sales and demand forecasting.
3. **Comparing model performances** using evaluation metrics such as MAE, MSE, RMSE, and R².
4. **Understanding the impact of external factors** like weather and promotions on sales and demand trends.

---

##  Models Used

The project involves using a combination of **machine learning models**, **deep learning models**, and **time series models** for sales and demand forecasting. Below is the detailed list of models used:

###  **Machine Learning Models**
- `XGBRegressor - Units Sold`
- `XGBRegressor - Demand Forecast`
- `RandomForestRegressor - Units Sold`
- `RandomForestRegressor - Demand Forecast`
- `GradientBoostingRegressor - Units Sold`
- `GradientBoostingRegressor - Demand Forecast`
- `SVR - Units Sold`
- `SVR - Demand Forecast`
- `Ridge Regression - Units Sold`
- `Ridge Regression - Demand Forecast`

###  **Deep Learning Models**
- `MLP - Sales Prediction`
- `MLP - Demand Forecast Prediction`
- `LSTM - Sales Prediction`
- `LSTM - Demand Forecast Prediction`
- `BiLSTM - Sales Prediction`
- `BiLSTM - Demand Forecast Prediction`
- `CNN - Sales Prediction`
- `CNN - Demand Forecast Prediction`
- `CNN+LSTM - Sales Prediction`
- `CNN+LSTM - Demand Forecast Prediction`
- `BiLSTM+CNN - Sales Prediction`
- `BiLSTM+CNN - Demand Forecast Prediction`

###  **Time Series Models**
- `Prophet - Units Sold`
- `Prophet - Demand Forecast`
- `ARIMA - Units Sold`
- `ARIMA - Demand Forecast`
- `SARIMAX - Units Sold`
- `SARIMAX - Demand Forecast`

---

##  Evaluation Metrics

Models were evaluated using the following metrics:

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

These metrics help assess model performance in terms of accuracy and error minimization. The higher the R² and the lower the error metrics, the better the model.

---

##  Top Model Performance

Here are the best-performing models based on the evaluation metrics:

| Model                            | Target              | MAE      | MSE       | RMSE     | R² Score   |
|----------------------------------|----------------------|----------|-----------|----------|------------|
| **MLP**                          | Demand Forecast      | 0.004381 | 0.000037  | 0.006100 | **0.999308** |
| **BiLSTM + CNN**                 | Sales Prediction     | 0.004398 | 0.000044  | 0.006643 | **0.999232** |

>  These two models showed remarkable performance, with near-perfect R² scores, making them the top models for demand and sales forecasting.

---

##  Visualizations

To aid in model evaluation and analysis, various visualizations were created:

- **Actual vs Predicted Plots** for both Units Sold and Demand Forecast.
- **Feature Importance** for machine learning models.
- **Correlation Heatmaps** to visualize relationships between features.
- **Sales and Demand Trends** by time, store, and product.
- **Impact of Holidays and Promotions** on sales and demand.

These visualizations provide insights into model performance and help in interpreting the results of forecasting.

---

##  Tools & Libraries

The following tools and libraries were used throughout the project:

- **Data Preprocessing & Analysis**: `Pandas`, `NumPy`
- **Machine Learning**: `Scikit-learn`, `XGBoost`, `LightGBM`
- **Deep Learning**: `Keras`, `TensorFlow`, `PyTorch`
- **Time Series Models**: `Prophet`, `statsmodels`
- **Visualization**: `Matplotlib`, `Seaborn`, `Plotly`
- **Notebooks**: Jupyter Notebook

---

##  Project Structure

The project is organized into the following structure:
Sales_Demand_Forecasting/

│

├── data/ # Raw and processed datasets

├── notebooks/ # EDA and model training notebooks

├── models/ # Trained model files

├── src/ # Python scripts for preprocessing and modeling

├── visuals/ # Plots and figures

└── README.md # Project documentation





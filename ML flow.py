# XGBoost
# Set MLflow tracking
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_experiment("Sales forecasting")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define model parameters
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

with mlflow.start_run(run_name="xgboost"):
    # Print parameters
    print("Training XGBoost model with parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")

    # Create and train model
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(Xf_train, Yt_train.values.ravel())  # Ensure Y is 1D array

    # Make predictions
    y_pred = xgb_model.predict(Xf_test)

    # Ensure Yt_test is numpy array and flattened
    y_true = np.array(Yt_test).ravel()

    # Calculate regression metrics
    mse = mean_squared_error(Yt_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Yt_test, y_pred)
    r2 = r2_score(Yt_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Create and save metrics bar plot
    plt.figure(figsize=(6, 4))
    metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}
    plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green"])
    plt.title("XGBoost Regression Metrics")
    plt.ylabel("Score")
    metrics_plot_path = "xgboost_metrics.png"
    plt.savefig(metrics_plot_path)
    plt.close()
    mlflow.log_artifact(metrics_plot_path)

    # Create and save true vs predicted scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color="blue", alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title("XGBRegressor - Units Sold")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    scatter_plot_path = "true_vs_predicted.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    mlflow.log_artifact(scatter_plot_path)

    plt.figure(figsize=(6, 4))
    plt.scatter(Yt_test["Demand Forecast"], y_pred[:, 1], color="blue", alpha=0.5)
    min_val = min(Yt_test["Demand Forecast"].min(), y_pred[:, 1].min())
    max_val = max(Yt_test["Demand Forecast"].max(), y_pred[:, 1].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title("XGBRegressor - Demand Forecast")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    demand_plot_path = "demand_forecast_vs_predicted.png"
    plt.savefig(demand_plot_path)
    plt.close()
    mlflow.log_artifact(demand_plot_path)
    # Log model
    mlflow.xgboost.log_model(xgb_model, "xgboost_regressor")

    # Add tags
    mlflow.set_tag("problem_type", "regression")
    mlflow.set_tag("model_family", "xgboost")

    # Print evaluation metrics
    print("\nModel evaluation:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
# -------------------------------------------------------
# Random forest tree
mlflow.set_experiment("Sales forecasting")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define model parameters
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 5,
    "random_state": 42,
}
with mlflow.start_run(run_name="random forest tree"):
    print("Training Random forest model with parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")
    # Create and train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(Xf_train, Yt_train)
    # Make predictions
    y_pred_rf = rf_model.predict(Xf_test)

    # Ensure Yt_test is numpy array and flattened
    y_true = np.array(Yt_test).ravel()
    # Calculate regression metrics
    mse = mean_squared_error(Yt_test, y_pred_rf)
    rmse = np.sqrt(mse)
    mse = mean_squared_error(Yt_test, y_pred_rf)
    mae = mean_absolute_error(Yt_test, y_pred_rf)
    r2 = r2_score(Yt_test, y_pred_rf)

    # Log parameters
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    # mlflow.log_params()

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    plt.figure(figsize=(6, 4))
    metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}
    plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green"])
    plt.title("Random forest Regression Metrics")
    plt.ylabel("Score")
    metrics_plot_path = "Randomforest_metrics.png"
    plt.savefig(metrics_plot_path)
    plt.close()
    mlflow.log_artifact(metrics_plot_path)

    # Create and save true vs predicted scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color="blue", alpha=0.5)
    min_val = min(y_true.min(), y_pred_rf.min())
    max_val = max(y_true.max(), y_pred_rf.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title("RandomforestRegressor - Units Sold")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    scatter_plot_path = "true_vs_predicted.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    mlflow.log_artifact(scatter_plot_path)

    plt.figure(figsize=(6, 4))
    plt.scatter(Yt_test["Demand Forecast"], y_pred_rf[:, 1], color="blue", alpha=0.5)
    min_val = min(Yt_test["Demand Forecast"].min(), y_pred_rf[:, 1].min())
    max_val = max(Yt_test["Demand Forecast"].max(), y_pred_rf[:, 1].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title("RandomforestRegressor - Demand Forecast")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    demand_plot_path = "demand_forecast_vs_predicted.png"
    plt.savefig(demand_plot_path)
    plt.close()
    mlflow.log_artifact(demand_plot_path)

    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_regressor")

    # Add tags
    mlflow.set_tag("problem_type", "regression")
    mlflow.set_tag("model_family", "random_forest")

    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
# ----------------------
# GradientBoostingRegressor
# Set MLflow tracking
mlflow.set_experiment("Sales forecasting")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define model parameters
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": 42,
}

with mlflow.start_run(run_name="gradient_boosting"):
    # Print parameters
    print("Training Gradient Boosting model with parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")

        # Create and train model
        print("Training Gradient Boosting model with parameters:")
        for param, value in params.items():
            print(f"{param}: {value}")
        # Create and train model
        gb_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model.fit(Xf_train, Yt_train)
        # Make predictions
        y_pred_gb = gb_model.predict(Xf_test)

        # Ensure Yt_test is numpy array and flattened
        y_true = np.array(Yt_test).ravel()
        # Calculate regression metrics
        mse = mean_squared_error(Yt_test, y_pred_gb)
        rmse = np.sqrt(mse)
        mse = mean_squared_error(Yt_test, y_pred_gb)
        mae = mean_absolute_error(Yt_test, y_pred_gb)
        r2 = r2_score(Yt_test, y_pred_gb)

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        # mlflow.log_params()

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        plt.figure(figsize=(6, 4))
        metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}
        plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green"])
        plt.title("Gradient Boosting Regression Metrics")
        plt.ylabel("Score")
        metrics_plot_path = "Gradient Boosting_metrics.png"
        plt.savefig(metrics_plot_path)
        plt.close()
        mlflow.log_artifact(metrics_plot_path)

        # Create and save true vs predicted scatter plot
        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, y_pred_gb, color="blue", alpha=0.5)
        min_val = min(y_true.min(), y_pred_gb.min())
        max_val = max(y_true.max(), y_pred_gb.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        plt.title("Gradient Boosting Regressor - Units Sold")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        scatter_plot_path = "true_vs_predicted.png"
        plt.savefig(scatter_plot_path)
        plt.close()
        mlflow.log_artifact(scatter_plot_path)

        plt.figure(figsize=(6, 4))
        plt.scatter(
            Yt_test["Demand Forecast"], y_pred_gb[:, 1], color="blue", alpha=0.5
        )
        min_val = min(Yt_test["Demand Forecast"].min(), y_pred_gb[:, 1].min())
        max_val = max(Yt_test["Demand Forecast"].max(), y_pred_gb[:, 1].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        plt.title("Gradient Boosting Regressor - Demand Forecast")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        demand_plot_path = "demand_forecast_vs_predicted.png"
        plt.savefig(demand_plot_path)
        plt.close()
        mlflow.log_artifact(demand_plot_path)

        # Log model
        mlflow.sklearn.log_model(gb_model, "Gradient Boosting_regressor")

        # Add tags
        mlflow.set_tag("problem_type", "regression")
        mlflow.set_tag("model_family", "Gradient Boosting")
        # Print evaluation metrics
        print("\nModel evaluation:")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

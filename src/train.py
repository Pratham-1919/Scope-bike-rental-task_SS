import numpy as np
from xgboost import XGBRegressor  
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
 
from src.preprocessing import run_preprocessing
from src.pipeline import build_full_pipeline
 
TARGET_COL = "Rented Bike Count"
RAW_CSV_PATH = r"E:\bike sharing task(7 July)\Notebook\SeoulBikeData.csv"
 
 
def split_X_y(df):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y
 
 
def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def tune_linear_regression(X_train, y_train):
    
    pipeline = build_full_pipeline(LinearRegression())

    
    pipeline.fit(X_train, y_train)
    return pipeline



def tune_decision_tree(X_train, y_train):
    param_dist = {
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", "log2", None]
    }
    
    pipeline = build_full_pipeline(DecisionTreeRegressor(random_state=42))
    
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best DecisionTree params:", search.best_params_)
    return search.best_estimator_
 
 
def tune_random_forest(X_train, y_train):
    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }
    pipeline = build_full_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
 
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best RandomForest params:", search.best_params_)
    return search.best_estimator_
 
 
def tune_xgboost(X_train, y_train):
 
    param_dist = {
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [3, 5, 7, 9],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 0.1, 0.3],
    "model__reg_alpha": [0, 0.1, 1],
    "model__reg_lambda": [1, 3, 5],
}
    pipeline = build_full_pipeline(
        XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror")
    )
 
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best XGBoost params:", search.best_params_)
    return search.best_estimator_
 
 
def main():
    train_df, val_df, test_df = run_preprocessing(RAW_CSV_PATH)
 
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)
 
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    dt_model = tune_decision_tree(X_train, y_train)
    dt_train_pred = dt_model.predict(X_train)
    dt_train_metrics = evaluate(y_train, dt_train_pred, label="DecisionTree (train)")
    
    dt_val_pred = dt_model.predict(X_val)
    dt_val_metrics = evaluate(y_val, dt_val_pred, label="DecisionTree (val)")
 
    rf_model = tune_random_forest(X_train, y_train)
    rf_train_pred = rf_model.predict(X_train)
    rf_metrics = evaluate(y_train, rf_train_pred, label="RandomForest (train)")

    rf_val_pred = rf_model.predict(X_val)
    rf_val_metrics = evaluate(y_val, rf_val_pred, label="RandomForest (val)")

 
    xgb_model = tune_xgboost(X_train, y_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_metrics = evaluate(y_val, xgb_val_pred, label="XGBoost (val)")

    xgb_train_pred = xgb_model.predict(X_train)
    xgb_train_metrics = evaluate(y_train, xgb_train_pred, label="XGBoost (train)")

    linear_model = tune_linear_regression(X_train, y_train)
    linear_train_pred = linear_model.predict(X_train)
    linear_metric = evaluate(y_train, linear_train_pred, label="Linear(train)")

    linear_val_pred = linear_model.predict(X_val)
    linear_val_metric = evaluate(y_val, linear_val_pred, label="Linear(Val)")


    model = {
    "random_forest": (rf_model, rf_val_metrics),
    "xgboost": (xgb_model, xgb_metrics),
    "decision_tree": (dt_model, dt_val_metrics),
    "linear_regression": (linear_model, linear_val_metric)
    }

    champion_name, (champion_model, champion_metrics) = max(
    model.items(),
    key=lambda item: item[1][1]["r2"]
    )

    print(f"Champion model: {champion_name}")
    print(f"Validation RMSE: {champion_metrics['rmse']:.2f}")
 
    test_pred = champion_model.predict(X_test)
    evaluate(y_test, test_pred, label=f"{champion_name} (test, final check)")

    import joblib
    joblib.dump(champion_model, "models/bike_rental_model.pkl")
 
    return champion_name, champion_model
 
 
if __name__ == "__main__":
    main()
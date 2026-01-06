# src/models/train.py
"""
Model Training Pipeline with Imputation Strategies and Cross-Validation

This script trains multiple model configurations with different imputation
strategies and selects the best performer based on cross-validation RMSE,
which is more reliable for production deployment.
"""
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import kagglehub

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load Boston Housing dataset from Kaggle.
    
    Returns:
        X: Feature DataFrame
        y: Target Series (MEDV - median value of homes)
    """
    logger.info("Downloading dataset from Kaggle...")
    
    path = kagglehub.dataset_download("altavish/boston-housing-dataset")
    logger.info(f"Dataset path: {path}")
    
    df = pd.read_csv(Path(path) / "HousingData.csv")
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    if 'MEDV' in df.columns:
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Report missing values
    null_counts = X.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        logger.info(f"Missing values detected: {total_nulls} total")
        for col in null_counts[null_counts > 0].index:
            pct = (null_counts[col] / len(X)) * 100
            logger.info(f"   {col}: {null_counts[col]} ({pct:.1f}%)")
    else:
        logger.info("No missing values detected")
    
    logger.info(f"Final features: {X.shape[1]}")
    logger.info(f"Feature names: {X.columns.tolist()}")
    
    return X, y


def eval_metrics(actual, pred):
    """
    Calculate regression metrics.
    
    Args:
        actual: True values
        pred: Predicted values
    
    Returns:
        Tuple of (rmse, mae, r2, mape)
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    return rmse, mae, r2, mape


def get_imputation_strategies():
    """
    Define imputation strategies for comparison.
    
    Strategies:
        - mean: Simple mean imputation (fast, baseline)
        - median: Median imputation (robust to outliers)
        - knn: K-Nearest Neighbors imputation (uses feature correlations)
    
    Returns:
        Dictionary mapping strategy names to imputer instances
    """
    return {
        "mean": SimpleImputer(strategy='mean'),
        "median": SimpleImputer(strategy='median'),
        "knn": KNNImputer(n_neighbors=5)
    }


def calculate_production_score(rmse, cv_rmse, cv_std):
    """
    Calculate production-ready score combining test performance and generalization.
    
    This weighted score prioritizes:
    - Cross-validation RMSE (50%): Generalization capability
    - Test RMSE (40%): Current performance
    - CV standard deviation (10%): Model stability
    
    Args:
        rmse: Root Mean Squared Error on test set
        cv_rmse: Mean RMSE from cross-validation
        cv_std: Standard deviation of CV RMSE
    
    Returns:
        Combined score (lower is better)
    """
    if np.isnan(cv_rmse):
        return float('inf')
    
    return 0.4 * rmse + 0.5 * cv_rmse + 0.1 * cv_std


def main():
    """Main training pipeline execution."""
    
    logger.info("="*70)
    logger.info("BOSTON HOUSING PRICE PREDICTION - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Define imputation strategies and models
    imputation_strategies = get_imputation_strategies()
    
    base_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            seed=42, 
            verbosity=0
        )
    }

    # Initialize MLflow experiment
    mlflow.set_experiment("boston_housing_with_imputation")
    
    # Track best model based on production score
    best_score = float("inf")
    best_config = None
    best_pipeline = None
    best_metrics = {}

    total_runs = len(imputation_strategies) * len(base_models)
    logger.info(f"\nTraining {total_runs} model configurations...")

    # Training loop: Imputer x Model
    for imp_name, imputer in imputation_strategies.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"IMPUTATION STRATEGY: {imp_name.upper()}")
        logger.info(f"{'='*70}")
        
        for model_name, model in base_models.items():
            run_name = f"{model_name}_{imp_name}"
            logger.info(f"\nTraining: {run_name}")
            
            with mlflow.start_run(run_name=run_name):
                # Build pipeline: Imputer -> Scaler -> Model
                pipeline = Pipeline([
                    ('imputer', imputer),
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # Cross-validation evaluation
                try:
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train, 
                        cv=5, 
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1
                    )
                    cv_rmse = -cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    logger.warning(f"  CV failed: {e}")
                    cv_rmse = np.nan
                    cv_std = np.nan
                
                # Train on full training set
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                # Calculate metrics
                rmse, mae, r2, mape = eval_metrics(y_test, predictions)
                
                # Calculate production score
                prod_score = calculate_production_score(rmse, cv_rmse, cv_std)
                
                # Log to MLflow
                mlflow.log_param("imputer", imp_name)
                mlflow.log_param("model", model_name)
                mlflow.log_params(model.get_params())
                
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("production_score", prod_score)
                
                if not np.isnan(cv_rmse):
                    mlflow.log_metric("cv_rmse", cv_rmse)
                    mlflow.log_metric("cv_std", cv_std)

                # Save model artifact in MLflow
                mlflow.sklearn.log_model(pipeline, "model")

                # Console output
                logger.info(f"  RMSE (test):      {rmse:.4f}")
                logger.info(f"  R²:               {r2:.4f}")
                if not np.isnan(cv_rmse):
                    logger.info(f"  CV RMSE:          {cv_rmse:.4f} ± {cv_std:.4f}")
                logger.info(f"  Production Score: {prod_score:.4f}")

                # Champion selection based on production score
                if prod_score < best_score:
                    best_score = prod_score
                    best_config = {
                        "model": model_name,
                        "imputer": imp_name
                    }
                    best_pipeline = pipeline
                    best_metrics = {
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "mape": mape,
                        "cv_rmse": cv_rmse if not np.isnan(cv_rmse) else None,
                        "cv_std": cv_std if not np.isnan(cv_std) else None,
                        "production_score": prod_score
                    }

    # Report champion model
    logger.info("\n" + "="*70)
    logger.info("CHAMPION MODEL SELECTED")
    logger.info("="*70)
    logger.info(f"Model:            {best_config['model']}")
    logger.info(f"Imputer:          {best_config['imputer']}")
    logger.info(f"RMSE (test):      {best_metrics['rmse']:.4f}")
    logger.info(f"CV RMSE:          {best_metrics['cv_rmse']:.4f}")
    logger.info(f"R²:               {best_metrics['r2']:.4f}")
    logger.info(f"Production Score: {best_metrics['production_score']:.4f}")
    logger.info("="*70 + "\n")
    
    # Register champion in MLflow
    champion_name = f"champion_model_{best_config['model']}_{best_config['imputer']}"
    with mlflow.start_run(run_name=champion_name):
        mlflow.log_param("imputer", best_config['imputer'])
        mlflow.log_param("model", best_config['model'])
        mlflow.log_param("is_champion", True)
        mlflow.log_params(best_pipeline.named_steps['model'].get_params())
        
        # Log metrics (filter None values)
        metrics_to_log = {
            k: v for k, v in best_metrics.items() 
            if v is not None and not np.isnan(v)
        }
        mlflow.log_metrics(metrics_to_log)
        
        # Register in Model Registry
        mlflow.sklearn.log_model(
            best_pipeline, 
            "model",
            registered_model_name="boston_housing_champion"
        )

    # Persist champion model to disk
    model_path = MODELS_DIR / "model_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Model saved at: {model_path}")
    
    # Save metadata
    metadata = {
        "model_name": best_config['model'],
        "imputer_strategy": best_config['imputer'],
        "metrics": {
            k: float(v) if v is not None and not np.isnan(v) else None 
            for k, v in best_metrics.items()
        },
        "trained_at": datetime.now().isoformat(),
        "features": X.columns.tolist(),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "selection_criterion": "production_score",
        "selection_rationale": (
            "Selected based on weighted production score: "
            "50% CV_RMSE (generalization), 40% test RMSE (performance), "
            "10% CV_STD (stability)"
        )
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved at: {metadata_path}")
    
    logger.info("\nTraining pipeline completed successfully")


if __name__ == "__main__":
    main()
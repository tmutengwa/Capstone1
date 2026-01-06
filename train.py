"""
Food Demand Forecasting - Model Training Script

Compares multiple models and trains the best one using
sklearn Pipeline with DictVectorizer.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load all required datasets"""
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_QoiMO9B.csv')
    fulfillment_center = pd.read_csv('data/fulfilment_center_info.csv')
    meal_info = pd.read_csv('data/meal_info.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Fulfillment centers: {fulfillment_center.shape}")
    print(f"Meals: {meal_info.shape}")

    return train, test, fulfillment_center, meal_info


def combine_data(train, test, fulfillment_center, meal_info):
    """Combine train and test data with additional information"""
    print("\nCombining datasets...")

    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    combined = combined.merge(fulfillment_center, on='center_id', how='left')
    combined = combined.merge(meal_info, on='meal_id', how='left')

    print(f"Combined shape: {combined.shape}")
    return combined


def create_features(df):
    """
    Create engineered features
    Keep categorical variables as strings for DictVectorizer
    """
    print("\nEngineering features...")
    df = df.copy()

    # Price features
    df['discount'] = df['base_price'] - df['checkout_price']
    df['discount_percentage'] = (df['discount'] / df['base_price']) * 100
    df['discount_percentage'] = df['discount_percentage'].fillna(0)

    # Promotional features
    df['total_promotion'] = df['emailer_for_promotion'] + df['homepage_featured']

    # Time-based features
    df['week_mod_4'] = df['week'] % 4
    df['week_mod_13'] = df['week'] % 13
    df['week_mod_52'] = df['week'] % 52

    # Convert categorical to strings for DictVectorizer
    categorical_cols = ['center_id', 'meal_id', 'city_code', 'region_code',
                       'center_type', 'category', 'cuisine']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def prepare_features(combined_df):
    """Prepare features for DictVectorizer pipeline"""
    print("\nPreparing features...")

    train_data = combined_df[combined_df['is_train'] == 1].copy()
    test_data = combined_df[combined_df['is_train'] == 0].copy()

    # Include ALL features (categorical will be one-hot encoded by DictVectorizer)
    exclude_cols = ['id', 'num_orders', 'is_train']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]

    print(f"Number of input features: {len(feature_cols)}")

    X_train = train_data[feature_cols]
    y_train = train_data['num_orders']
    X_test = test_data[feature_cols]

    return X_train, y_train, X_test, feature_cols


def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def compare_models(X_train, y_train):
    """
    Compare multiple models using DictVectorizer pipelines
    Returns the best model name and pipeline
    """
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)

    # Define models to compare
    models = {
        # 'Linear Regression': LinearRegression(),
        # 'Lasso': Lasso(alpha=1.0, random_state=42),
        # 'Ridge': Ridge(alpha=1.0, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15,
                                               random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                       random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=42, n_jobs=-1)
    }

    # Convert to dict format
    X_train_dict = X_train.to_dict('records')

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_dict, y_train, test_size=0.2, random_state=42
    )

    results = []
    print(f"\n{'Model':<20} {'Train RMSLE':<15} {'Val RMSLE':<15} {'100*Val RMSLE':<15}")
    print("="*70)

    for name, model in models.items():
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('dict_vectorizer', DictVectorizer(sparse=False)),
                ('model', model)
            ])

            # Train
            pipeline.fit(X_tr, y_tr)

            # Predict
            y_train_pred = pipeline.predict(X_tr)
            y_val_pred = pipeline.predict(X_val)

            # Calculate RMSLE
            train_rmsle = rmsle(y_tr, y_train_pred)
            val_rmsle = rmsle(y_val, y_val_pred)

            results.append({
                'Model': name,
                'Train RMSLE': train_rmsle,
                'Val RMSLE': val_rmsle,
                'Score (100*RMSLE)': 100 * val_rmsle,
                'Pipeline': pipeline
            })

            print(f"{name:<20} {train_rmsle:<15.4f} {val_rmsle:<15.4f} {100*val_rmsle:<15.4f}")

        except Exception as e:
            print(f"{name:<20} Error: {str(e)}")

    # Sort by validation RMSLE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val RMSLE')

    print("\n" + "="*70)
    print("BEST MODEL:")
    best = results_df.iloc[0]
    print(f"  Name: {best['Model']}")
    print(f"  Validation RMSLE: {best['Val RMSLE']:.4f}")
    print(f"  Score (100*RMSLE): {best['Score (100*RMSLE)']:.4f}")
    print("="*70)

    return best['Model'], results_df


def tune_hyperparameters(model_name, X_train, y_train):
    """
    Tune hyperparameters for the best model
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING: {model_name}")
    print("="*70)

    # Define parameter grids
    param_grids = {
        'XGBoost': {
            'model__n_estimators': [100],
            'model__max_depth': [5],
            'model__learning_rate': [0.1],
            'model__subsample': [0.8]
        },
        'Random Forest': {
            'model__n_estimators': [100],
            'model__max_depth': [15],
            'model__min_samples_split': [5]
        },
        'Gradient Boosting': {
            'model__n_estimators': [100],
            'model__max_depth': [5],
            'model__learning_rate': [0.1]
        }
    }

    if model_name not in param_grids:
        print(f"No tuning grid defined for {model_name}")
        print("Using default parameters")
        return None

    # Get base model
    models_map = {
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    base_model = models_map[model_name]

    # Create pipeline
    pipeline = Pipeline([
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('model', base_model)
    ])

    # Convert to dict
    X_train_dict = X_train.to_dict('records')

    # Grid search
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
    grid_search = GridSearchCV(
        pipeline,
        param_grids[model_name],
        cv=3,
        scoring=rmsle_scorer,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting grid search...")
    grid_search.fit(X_train_dict, y_train)

    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Evaluate
    y_pred = grid_search.best_estimator_.predict(X_train_dict)
    train_rmsle = rmsle(y_train, y_pred)

    print(f"\nTuned Model Performance:")
    print(f"  Train RMSLE: {train_rmsle:.4f}")
    print(f"  Score (100*RMSLE): {100*train_rmsle:.4f}")

    return grid_search.best_estimator_


def save_pipeline(pipeline, feature_cols):
    """Save the trained pipeline"""
    print("\n" + "="*70)
    print("SAVING PIPELINE")
    print("="*70)

    with open('final_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("✓ Pipeline saved as 'final_model.pkl'")

    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✓ Feature columns saved as 'feature_cols.pkl'")

    # Show pipeline info
    dict_vec = pipeline.named_steps['dict_vectorizer']
    model = pipeline.named_steps['model']

    print(f"\nPipeline Details:")
    print(f"  Steps: {list(pipeline.named_steps.keys())}")
    print(f"  Input features: {len(feature_cols)}")
    print(f"  Vectorized features: {len(dict_vec.feature_names_)}")
    print(f"  Model: {type(model).__name__}")


def main():
    """Main training pipeline"""
    print("="*70)
    print("FOOD DEMAND FORECASTING - MODEL TRAINING")
    print("Using DictVectorizer Pipeline")
    print("="*70)

    # Load data
    train, test, fulfillment_center, meal_info = load_data()

    # Combine datasets
    combined = combine_data(train, test, fulfillment_center, meal_info)

    # Feature engineering
    combined_with_features = create_features(combined)

    # Prepare features
    X_train, y_train, X_test, feature_cols = prepare_features(combined_with_features)

    # Sample data for faster comparison and tuning
    SAMPLE_SIZE = 50000
    if len(X_train) > SAMPLE_SIZE:
        print(f"\nSampling {SAMPLE_SIZE} records for model comparison and tuning...")
        # Use numpy to generate random indices
        sample_indices = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
        X_train_sample = X_train.iloc[sample_indices].copy()
        y_train_sample = y_train.iloc[sample_indices].copy()
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    # Compare models and get best one (using sample)
    best_model_name, results_df = compare_models(X_train_sample, y_train_sample)

    # Tune hyperparameters for best model (using sample)
    tuned_pipeline = tune_hyperparameters(best_model_name, X_train_sample, y_train_sample)

    # If tuning didn't apply, use the best from comparison
    if tuned_pipeline is None:
        best_result = results_df.iloc[0]
        final_pipeline = best_result['Pipeline']
        print(f"\nUsing {best_model_name} with default parameters")
    else:
        final_pipeline = tuned_pipeline

    # Retrain on FULL data
    print(f"\nRetraining {best_model_name} on FULL dataset ({len(X_train)} records)...")
    X_train_dict = X_train.to_dict('records')
    # Reset the pipeline to ensure clean retraining
    final_pipeline.fit(X_train_dict, y_train)

    # Save pipeline
    save_pipeline(final_pipeline, feature_cols)

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Files created:")
    print(f"  - final_model.pkl (complete pipeline)")
    print(f"  - feature_cols.pkl (feature names)")
    print(f"\nNext steps:")
    print(f"  1. Start API: uvicorn predict:app --reload")
    print(f"  2. Test API: python test_api.py")
    print("="*70)


if __name__ == "__main__":
    main()

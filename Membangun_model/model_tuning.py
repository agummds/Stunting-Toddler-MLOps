import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from dotenv import load_dotenv
from modelling import load_preprocessed_data, plot_confusion_matrix, evaluate_model

# Load environment variables
load_dotenv()

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/agummds/Stunting-Toddler.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def hyperparameter_tuning_grid(X_train, y_train, cv=3):
    """Perform grid search hyperparameter tuning"""
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Create a RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train.values.ravel())
    
    # Return best model and parameters
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def hyperparameter_tuning_random(X_train, y_train, cv=3, n_iter=20):
    """Perform randomized search hyperparameter tuning"""
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(5, 30, 5).tolist() + [None],
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf': np.arange(1, 10, 1),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Create a RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    
    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train.values.ravel())
    
    # Return best model and parameters
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_

def plot_hyperparameter_scores(cv_results, param_name, top_n=10, save_path=None):
    """Plot effects of hyperparameter on score"""
    # Extract relevant data
    scores = cv_results['mean_test_score']
    params = cv_results['params']
    
    # Create dataframe
    param_values = [p[param_name] for p in params]
    df = pd.DataFrame({'param_value': param_values, 'score': scores})
    
    # Group by parameter value and calculate mean score
    grouped = df.groupby('param_value').mean().reset_index()
    grouped = grouped.sort_values('score', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='param_value', y='score', data=grouped.head(top_n))
    plt.title(f'Impact of {param_name} on Model Performance')
    plt.xlabel(param_name)
    plt.ylabel('Mean F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Create dirs for outputs
    os.makedirs("tuning_results", exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="hyperparameter_tuning"):
        # Log dataset info
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # 1. Perform randomized search (faster)
        print("Starting Randomized Search...")
        best_model_random, best_params_random, cv_results_random = hyperparameter_tuning_random(
            X_train, y_train, cv=5, n_iter=30
        )
        
        # Log best parameters from randomized search
        for param, value in best_params_random.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Save visualization of top parameters' impact
        for param in best_params_random.keys():
            save_path = f"tuning_results/{param}_impact.png"
            plot_hyperparameter_scores(cv_results_random, param, save_path=save_path)
            mlflow.log_artifact(save_path)
        
        # 2. Fine-tune with GridSearch around best parameters
        print("Starting Grid Search around best parameters...")
        # Narrow down parameter grid based on randomized search results
        focused_param_grid = {
            'n_estimators': [
                max(50, best_params_random['n_estimators'] - 50),
                best_params_random['n_estimators'],
                min(500, best_params_random['n_estimators'] + 50)
            ],
            'max_depth': [
                best_params_random['max_depth'] - 2 if best_params_random['max_depth'] is not None and best_params_random['max_depth'] > 2 else 1,
                best_params_random['max_depth'],
                best_params_random['max_depth'] + 2 if best_params_random['max_depth'] is not None else None
            ],
            'min_samples_split': [
                max(2, best_params_random['min_samples_split'] - 1),
                best_params_random['min_samples_split'],
                best_params_random['min_samples_split'] + 1
            ],
            'min_samples_leaf': [
                max(1, best_params_random['min_samples_leaf'] - 1),
                best_params_random['min_samples_leaf'],
                best_params_random['min_samples_leaf'] + 1
            ]
        }
        
        # Remove None entries for clean grid search
        for key, value_list in focused_param_grid.items():
            focused_param_grid[key] = [v for v in value_list if v is not None]
            if len(focused_param_grid[key]) == 0:
                focused_param_grid[key] = [None]
        
        # Create grid search with focused parameters
        rf = RandomForestClassifier(
            random_state=42,
            bootstrap=best_params_random['bootstrap'],
            max_features=best_params_random.get('max_features', 'sqrt'),
            class_weight=best_params_random.get('class_weight', None)
        )
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=focused_param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train.values.ravel())
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log best parameters from grid search
        for param, value in best_params.items():
            mlflow.log_param(f"final_{param}", value)
        
        # 3. Evaluate final model
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        # Save model locally
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        mlflow.sklearn.save_model(best_model, os.path.join(model_dir, "tuned_model"))
        
        # Create and log confusion matrix
        plot_confusion_matrix(y_test, y_pred, "tuned_confusion_matrix.png")
        mlflow.log_artifact("tuned_confusion_matrix.png")
        
        # Log feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Feature Importance (Tuned Model)')
        plt.tight_layout()
        plt.savefig("tuned_feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("tuned_feature_importance.png")
        
        # Compare tuned model with baseline
        print("=== Model Comparison ===")
        print(f"Tuned model F1 score: {metrics['f1']:.4f}")
        
        # Create learning curve plot
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            n_samples = int(X_train.shape[0] * size)
            model = RandomForestClassifier(**best_params, random_state=42)
            model.fit(X_train.iloc[:n_samples], y_train.iloc[:n_samples].values.ravel())
            
            train_pred = model.predict(X_train.iloc[:n_samples])
            test_pred = model.predict(X_test)
            
            train_f1 = f1_score(y_train.iloc[:n_samples], train_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            train_scores.append(train_f1)
            test_scores.append(test_f1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score')
        plt.plot(train_sizes, test_scores, 'o-', label='Test score')
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size Fraction')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig("learning_curve.png")
        plt.close()
        
        mlflow.log_artifact("learning_curve.png")

if __name__ == "__main__":
    main()
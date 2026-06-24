import os 
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
                "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
            }
            params = {

                "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
                },

                "Naive Bayes": {
                "var_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-7]
                },

                "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None,3, 5, 10, 20]
                },

                "Random Forest": {
                "n_estimators": [100,200,300],
                "max_depth":[5,10,20,None]
                },

                "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [50, 100]
                },

                "XGBoost": {
                "learning_rate": [0.1, 0.01,0.05, 0.001],
                "n_estimators": [100,200,300],
                "max_depth":[3,5,7]
                },
                "CatBoost": {
                    "depth": [4, 6,8],
                    "learning_rate": [0.01,0.05, 0.1],
                    "iterations": [200,300,500]
                },

                "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [50, 100]
                },
                

                }

            logging.info("Evaluating models...")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models=models,params = params)
            logging.info("Model training completed successfully")

            # Extract the best performing model score and name
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with an F1 score above 0.6",sys)
                
            logging.info(f"Best model found: {best_model_name} with F1 score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            f1 = f1_score(y_test, predicted)
            return f1
            
        except Exception as e:
            raise CustomException(e, sys)

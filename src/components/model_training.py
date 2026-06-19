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
            
            logging.info("Evaluating models...")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models=models)
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

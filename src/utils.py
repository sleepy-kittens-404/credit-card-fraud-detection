import sys
import os
import dill
from src.exception import CustomException
from sklearn.metrics import f1_score
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_tr, y_tr, x_tst, y_tst, models):
    try:
        report = {}
        # Cleanly iterate over name and model object directly
        for model_name, model in models.items():
            model.fit(x_tr, y_tr)
            
            # Predict on test set
            y_pred_test = model.predict(x_tst)
            test_model_score = f1_score(y_tst, y_pred_test)

            # Store using the string name as the key
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e, sys)



import sys
import os
import dill
from src.exception import CustomException
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


def evaluate_model(x_tr, y_tr, x_tst, y_tst, models, params):

    try:
        report = {}

        for model_name, model in models.items():

            gs = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=1
            )

            gs.fit(x_tr, y_tr)

            model.set_params(**gs.best_params_)

            model.fit(x_tr, y_tr)

            y_pred = model.predict(x_tst)

            test_score = f1_score(y_tst, y_pred)

            report[model_name] = test_score

            print(f"{model_name}")
            print(f"Best Params: {gs.best_params_}")
            print(f"F1 Score: {test_score}")
            print("-" * 50)

        return report

    except Exception as e:
        raise e



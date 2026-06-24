import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            pred_proba = model.predict_proba(data_scaled)

            return pred, pred_proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, row: dict):
        # row is a dict of all 30 feature columns (Time, V1-V28, Amount)
        self.row = row

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.row])
        except Exception as e:
            raise CustomException(e, sys)
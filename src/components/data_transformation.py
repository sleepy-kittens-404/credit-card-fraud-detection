import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler ,RobustScaler
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
           
            preprocessor = ColumnTransformer(transformers=[("Robust Scaler",RobustScaler(),["Amount"]),
            "Standard Scaler",StandardScaler(),["Time"]])
            logging.info("Scaling of 'Amount' and 'Time' completed")
            return preprocessor
        except Exception as e:
            raise(CustomException(e,sys))
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
        except Exception as e:
            raise CustomException(e,sys)

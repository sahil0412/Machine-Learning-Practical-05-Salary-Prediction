import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder 
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def convert_income_level(self, income_level):
        if income_level == '<=50K':
            return 0
        elif income_level == '>50K':
            return 1
        else:
            return None  # Handle other cases if necessary

        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Seprate Dependent and Independent Varables')
            
            # Changing depending variable to be 0 or 1
            target_column_name = 'income'
            train_df[target_column_name] = train_df[target_column_name].map(self.convert_income_level)
            test_df[target_column_name] = test_df[target_column_name].map(self.convert_income_level)
            # preprocessing_obj = self.get_data_transformation_object()

            
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Data preprocessing
            categorical_features=["education","workclass","marital.status","occupation","relationship","race","sex","native.country"]
            numerical_features=["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]
            
            preprocessing_data = {}
            input_feature_train_arr = pd.DataFrame()
            input_feature_test_arr = pd.DataFrame()
            
            # Fit and transform the categorical columns
            for column in categorical_features:
                l_encoder = LabelEncoder()
                input_feature_train_arr[column] = l_encoder.fit_transform(input_feature_train_df[column])
                preprocessing_data[f'{column}_label_encoder'] = l_encoder

            # Fit and transform the numerical columns
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            input_feature_train_arr[numerical_features] = imputer.fit_transform(input_feature_train_df[numerical_features])
            input_feature_train_arr[numerical_features] = scaler.fit_transform(input_feature_train_df[numerical_features])
            
            preprocessing_data['numerical_imputer'] = imputer
            preprocessing_data['numerical_scaler'] = scaler

            logging.info("Completed fit-transform for Training Data")
            logging.info(f"Preprocessing Object is: {preprocessing_data}")
            
            # Transform the categorical columns in the test data
            for column in categorical_features:
                l_encoder = preprocessing_data[f'{column}_label_encoder']
                input_feature_test_arr[column] = input_feature_test_df[column].map(lambda s: -1 if s not in l_encoder.classes_ else l_encoder.transform([s])[0])

            imputer = preprocessing_data['numerical_imputer']
            scaler = preprocessing_data['numerical_scaler']

            input_feature_test_arr[numerical_features] = imputer.transform(input_feature_test_df[numerical_features])
            input_feature_test_arr[numerical_features] = scaler.transform(input_feature_test_df[numerical_features])

            logging.info("Completed transform for Testing Data")

            
            ## Trnasformating using preprocessor obj
            # input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_data
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
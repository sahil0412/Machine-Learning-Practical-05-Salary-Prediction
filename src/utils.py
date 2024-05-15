import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        model.fit(X_train, y_train)
        print('Training Score : ', model.score(X_train, y_train))
        print('Testing Score  : ', model.score(X_test, y_test))
        # Predict Testing data
        test_accuracy = model.score(X_test, y_test)
        y_test_pred =model.predict(X_test)
        cs = confusion_matrix(y_test, y_test_pred)
        return test_accuracy, cs, model
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
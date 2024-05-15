import sys
sys.path.append("./src")
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initaite_data_transformation(train_data_path, test_data_path)
    train_model = ModelTrainer()
    train_model.initiate_model_training(train_arr, test_arr)
import numpy as np
import pandas as pd
import sys
sys.path.append("D:/project_hub/Predição de Pacientes com Diabetes")
from src.model_utils import *

#caminhos
MODEL_PATH = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"
SPLIT_SAVE_DIR = "D:/project_hub/Predição de Pacientes com Diabetes/splits/"


if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    for fold in range(1, 6): #folds usados
        print(f"\n--- Avaliação para o Fold {fold} ---")
        
        #carregando os conjuntos
        train_file = f"{SPLIT_SAVE_DIR}train_fold_{fold}.csv"
        test_file = f"{SPLIT_SAVE_DIR}test_fold_{fold}.csv"
        train_data = pd.read_csv(train_file, header=None).values
        test_data = pd.read_csv(test_file, header=None).values
        
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]
        
        #avaliando o modelo
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(results)

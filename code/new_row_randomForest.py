import numpy as np
import pickle
import sys
sys.path.append('d:/project_hub/Predição de Pacientes com Diabetes')
from utils.model_utils import load_model

model = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"
modelo = load_model(model)

nova_linha = np.array([[5, 17, 24.6, 0.387, 34]])

predict = modelo.predict(nova_linha)

print(predict)
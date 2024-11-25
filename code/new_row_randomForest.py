import numpy as np
import pickle
import sys
sys.path.append('d:/project_hub/Predição de Pacientes com Diabetes')
from src.model_utils import load_model

model = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"
modelo = load_model(model)

nova_linha = np.array([[5, 176, 72, 17, 24.6, 0.387, 34]])

predict = modelo.predict(nova_linha)

resultado = "Diabético" if predict[0] == 1 else "Não Diabético"
print(f"Predição da nova linha: {resultado}")

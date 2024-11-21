import numpy as np
import pickle

model = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"

with open(model, "rb") as f:
    modelo = pickle.load(f)

new_row = np.array([[5, 176, 72, 17, 24.6, 0.387, 34]])

predict = modelo.predict(new_row)

resultado = "Diabético" if predict[0] == 1 else "Não Diabético"
print(f"Predição da nova linha: {resultado}")

##usar data.describe().T
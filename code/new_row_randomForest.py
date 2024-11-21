import numpy as np
import pickle
import pandas as pd

csv = "D:/project_hub/Predição de Pacientes com Diabetes/input/diabetes_dataset.csv"
model = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"

data = pd.read_csv(csv)

def valor_aleatorio(coluna):
    valor_min = data[coluna].min()
    valor_max = data[coluna].max()

    if data[coluna].dtype == int:
        valor = np.random.randint(valor_min, valor_max+1)
        return valor
    else:
        num_decimals = len(str(valor_max).split('.')[1]) if '.' in str(valor_max) else 0
        valor = round(np.random.uniform(valor_min, valor_max), num_decimals)
        return valor

nova_linha = [
    valor_aleatorio("Gravidez"),
    valor_aleatorio("Glicose"),
    valor_aleatorio("PressaoSanguinea"),
    valor_aleatorio("EspessuraDaPele"),
    valor_aleatorio("IMC"),
    valor_aleatorio("DiabetesPedigree"),
    valor_aleatorio("Idade")
]

nova_linha = np.array(nova_linha)
#print(nova_linha)


with open(model, "rb") as f:
    modelo = pickle.load(f)

#new_row = np.array([[5, 176, 72, 17, 24.6, 0.387, 34]])

predict = modelo.predict(nova_linha.reshape(1, -1))

resultado = "Diabético" if predict[0] == 1 else "Não Diabético"
print(nova_linha.reshape(1, -1))
print(f"Predição da nova linha: {resultado}")
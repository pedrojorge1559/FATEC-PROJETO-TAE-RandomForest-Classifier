import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

data = pd.read_csv("D:/project_hub/Predição de Pacientes com Diabetes/input/diabetes_dataset.csv")

x = np.array([data["Gravidez"], data["Glicose"], data["PressaoSanguinea"], data["EspessuraDaPele"],
              data["IMC"], data["DiabetesPedigree"], data["Idade"]]).T
y = np.array(data["Diabetico"])

#folds
kf = KFold(n_splits=6)

melhor_acuracia = 0
melhor_modelo = None

for fold, (train_index, test_index) in enumerate(kf.split(x)):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #treinando com os folds
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    #avaliando o modelo em cada fold
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo no fold {fold + 1}: {accuracy:.2f}")
    
    #setando melhor modelo
    if accuracy > melhor_acuracia:
        melhor_acuracia = accuracy
        melhor_modelo = rf

#salvando modelo melhor avaliado
if melhor_modelo is not None:
    with open(f"D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl", "wb") as f:
        pickle.dump(melhor_modelo, f)
    print(f"melhor acuracia ({melhor_acuracia:.2f}), salvo")
else:
    print("not")
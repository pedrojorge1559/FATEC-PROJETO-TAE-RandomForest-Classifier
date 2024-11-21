import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import sys
sys.path.append("D:/project_hub/Predição de Pacientes com Diabetes")
from src.model_utils import save_model

DATA_PATH = "D:/project_hub/Predição de Pacientes com Diabetes/input/diabetes_dataset.csv"
MODEL_PATH = "D:/project_hub/Predição de Pacientes com Diabetes/models/melhor_modelo_random_forest.pkl"
SPLIT_SAVE_DIR = "D:/project_hub/Predição de Pacientes com Diabetes/splits/"

data = pd.read_csv(DATA_PATH)
x = np.array([data["Gravidez"], data["Glicose"], data["PressaoSanguinea"], data["EspessuraDaPele"],
              data["IMC"], data["DiabetesPedigree"], data["Idade"]]).T
y = np.array(data["Diabetico"])

#kfold features
kf = KFold(n_splits=5, shuffle=True, random_state=42)

melhor_acuracia = 0
melhor_modelo = None

for fold, (train_index, test_index) in enumerate(kf.split(x)):

    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #salvando os conjuntos para validação
    train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
    test_data = np.hstack((X_test, y_test.reshape(-1, 1)))
    train_file = f"{SPLIT_SAVE_DIR}train_fold_{fold + 1}.csv"
    test_file = f"{SPLIT_SAVE_DIR}test_fold_{fold + 1}.csv"
    pd.DataFrame(train_data).to_csv(train_file, index=False, header=False)
    pd.DataFrame(test_data).to_csv(test_file, index=False, header=False)
    
    #treinando o modelo
    rf = RandomForestClassifier(n_estimators=277, random_state=42, max_depth=9, 
                                 min_samples_split=7, min_samples_leaf=1, max_features=3)
    rf.fit(X_train, y_train)
    
    #avaliacao do fold
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo no fold {fold + 1}: {accuracy:.2f}")
    
    #setando melhor modelo
    if accuracy > melhor_acuracia:
        melhor_acuracia = accuracy
        melhor_modelo = rf

#salvando modelo
if melhor_modelo is not None:
    save_model(melhor_modelo, MODEL_PATH)
    print(f"melhor modelo salvo : {melhor_acuracia:.2f}")
else:
    print("not")

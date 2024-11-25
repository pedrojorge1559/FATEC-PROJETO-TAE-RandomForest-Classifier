import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
                    
def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Avalia um modelo treinado usando métricas como acurácia, matriz de confusão e curva ROC.

    Parâmetros:
    - model: modelo treinado.
    - X_train, X_test: dados de treino e teste.
    - y_train, y_test: rótulos de treino e teste.

    Retorna:
    - Um dicionário com métricas de avaliação.
    """
    # Acurácia no treino e no teste
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Acurácia no treino: {train_acc:.2f}")
    print(f"Acurácia no teste: {test_acc:.2f}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, test_preds)
    print("Matriz de Confusão:")
    print(cm)

    # Curva ROC e AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plotando a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


#results = evaluate_model(modelo, X_train, X_test, y_train, y_test)
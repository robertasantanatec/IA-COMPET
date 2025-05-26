from sklearn.metrics import accuracy_score, classification_report

def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório:\n", classification_report(y_test, y_pred))

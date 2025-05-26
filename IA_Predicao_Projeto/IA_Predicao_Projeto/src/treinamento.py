from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def treinar_modelo(X_train, y_train, modelo_nome='random_forest'):
    if modelo_nome == 'random_forest':
        modelo = RandomForestClassifier(random_state=42)
    elif modelo_nome == 'logistic_regression':
        modelo = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Modelo não suportado")

    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return accuracy_score(y_test, y_pred)

def salvar_modelo(modelo, caminho='models/modelo_final.pkl'):
    joblib.dump(modelo, caminho)

def executar_treinamento(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = {}
    resultados = {}

    for nome in ['random_forest', 'logistic_regression']:
        modelo = treinar_modelo(X_train, y_train, modelo_nome=nome)
        acc = avaliar_modelo(modelo, X_test, y_test)
        modelos[nome] = modelo
        resultados[nome] = acc
        print(f"Acurácia do modelo {nome}: {acc:.4f}")

    melhor_modelo_nome = max(resultados, key=resultados.get)
    print(f"Melhor modelo: {melhor_modelo_nome} com acurácia {resultados[melhor_modelo_nome]:.4f}")

    salvar_modelo(modelos[melhor_modelo_nome])
    print(f"Modelo salvo em 'models/modelo_final.pkl'")

if __name__ == "__main__":
    from preprocessamento import carregar_dados, preprocessar_dados

    df = carregar_dados('data/bodyfat.csv')
    X, y = preprocessar_dados(df)
    executar_treinamento(X, y)
    df = df.dropna()

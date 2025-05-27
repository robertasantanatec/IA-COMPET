import pandas as pd
from sklearn.preprocessing import StandardScaler

def carregar_dados(caminho):
    return pd.read_csv(caminho)

def criar_target_categorico(df):
    # Definindo faixas para categorizar BodyFat em 'baixo', 'medio' e 'alto'
    bins = [0, 15, 25, 100]
    labels = ['baixo', 'medio', 'alto']
    df['target'] = pd.cut(df['BodyFat'], bins=bins, labels=labels)
    return df

def preprocessar_dados(df):
    import joblib
    
    df = df.dropna()  # Remove linhas com dados faltantes
    
    df = criar_target_categorico(df)
    
    # Remove linhas com target NaN (que não caíram em nenhum bin)
    df = df.dropna(subset=['target'])

    X = df.drop(['BodyFat', 'target'], axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Salva o scaler para usar na interface depois
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y



if __name__ == "__main__":
    df = carregar_dados('data/bodyfat.csv')
    X_scaled, y = preprocessar_dados(df)
    print("Shape X:", X_scaled.shape)
    print("Distribuição da variável target:")
    print(y.value_counts())

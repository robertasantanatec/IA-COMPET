from src.preprocessamento import carregar_dados, preprocessar_dados
from src.treinamento import treinar_modelo, salvar_modelo
from src.avaliacao import avaliar_modelo

# Carrega os dados do novo caminho
df = carregar_dados('data/bodyfat.csv')


X, y = preprocessar_dados(df)
modelo = treinar_modelo(X, y)
salvar_modelo(modelo, 'models/modelo_final.pkl')
avaliar_modelo(modelo, X, y)

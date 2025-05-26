# IA-COMPET

Este projeto tem como objetivo usar inteligência artificial para **classificar o percentual de gordura corporal** de uma pessoa em três categorias:

- Baixo
- Médio
- Alto

---

## 📊 Fonte dos Dados

Os dados foram retirados do site Kaggle:  
🔗 [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)

Essa base tem informações como:

- Idade  
- Peso  
- Altura  
- Medidas corporais (abdômen, peito, coxa, etc.)  
- Percentual de gordura corporal (BodyFat)

---

## ⚙️ Etapas do Projeto

1. **Importação dos dados (CSV)**  
2. **Análise exploratória básica** com gráficos e visualizações  
3. **Criação do alvo (target)**: classificamos o percentual de gordura em "baixo", "médio" ou "alto"  
4. **Pré-processamento**: remoção de valores nulos, normalização dos dados  
5. **Divisão entre treino e teste**  
6. **Treinamento de dois modelos de classificação**:  
   - Random Forest  
   - K-Nearest Neighbors (KNN)  
7. **Avaliação dos modelos** usando acurácia

---


---

## 🧪 Bibliotecas usadas

- pandas  
- matplotlib  
- seaborn  
- scikit-learn

---

## 🚀 Como rodar o projeto?

1. Clone o repositório
2. Instale as bibliotecas (se quiser):
```bash
pip install -r requirements.txt


## 🏆 Resultado

O modelo que teve o melhor desempenho foi o **Random Forest**, com boa taxa de acerto na previsão das classes de gordura corporal.



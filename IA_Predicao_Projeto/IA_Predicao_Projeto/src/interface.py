import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Carregar modelo e scaler
modelo = joblib.load('models/modelo_final.pkl')
scaler = joblib.load('models/scaler.pkl')  # <-- carregar o scaler aqui

# Carregar dados para pegar os nomes das colunas de entrada
df = pd.read_csv('data/bodyfat.csv')
colunas_features = df.drop(['BodyFat', 'target'], axis=1, errors='ignore').columns.tolist()

st.title("Predição com IA")

entradas = []
for col in colunas_features:
    valor = st.number_input(f"Digite o valor para {col}")
    entradas.append(valor)

if st.button("Prever"):
    entrada_array = np.array([entradas])
    entrada_scaled = scaler.transform(entrada_array)  # <-- aplica o scaler aqui
    resultado = modelo.predict(entrada_scaled)
    st.write(f"Resultado: {resultado[0]}")

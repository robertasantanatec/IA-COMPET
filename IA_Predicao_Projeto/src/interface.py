import gradio as gr
import joblib
import numpy as np
import pandas as pd
 
modelo = joblib.load("models/modelo_final.pkl")
scaler = joblib.load("models/scaler.pkl")
 
df = pd.read_csv("data/bodyfat.csv")
colunas_features = df.drop(
    ["BodyFat", "target"], axis=1, errors="ignore"
).columns.tolist()
 
 
def prever(*entradas):
    entrada_array = np.array([entradas])
    entrada_scaled = scaler.transform(entrada_array)
    resultado = modelo.predict(entrada_scaled)
    return f"Resultado: {resultado[0]}"
 
 
inputs = [gr.Number(label=col) for col in colunas_features]
 
interface = gr.Interface(
    fn=prever,
    inputs=inputs,
    outputs=gr.Textbox(label="Classe prevista"),
    title="Predição com IA - Classificação de Gordura Corporal",
)
 
interface.launch()
 
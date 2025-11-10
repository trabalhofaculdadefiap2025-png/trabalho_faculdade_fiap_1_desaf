import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.title('Sistema de Suporte ao Diagnóstico de Câncer de Mama (Fase 2)')
st.write('Insira os dados do paciente para obter uma análise inicial otimizada (GA + LLM).')

MODEL_FILE = 'decision_tree_model_GA_optimized.joblib'
SCALER_FILE = 'minmax_scaler.joblib'
LLM_MODEL = 'gemini-2.5-flash'

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    st.success('Modelo Otimizado (GA) e Scaler carregados com sucesso!')
except FileNotFoundError:
    st.error('Erro: Arquivos de modelo ou scaler não encontrados. Verifique se eles estão na mesma pasta.')
    st.stop()

st.header('Dados do Paciente')
st.markdown('---')

st.subheader('Features Selecionadas')

# Apenas as 8 colunas usadas no treinamento
radius_mean = st.number_input('Radius_mean', format="%.4f")
perimeter_mean = st.number_input('Perimeter_mean', format="%.4f")
area_mean = st.number_input('Area_mean', format="%.4f")
radius_worst = st.number_input('Radius_worst', format="%.4f")
perimeter_worst = st.number_input('Perimeter_worst', format="%.4f")
area_worst = st.number_input('Area_worst', format="%.4f")
area_se = st.number_input('Area_se', format="%.4f")
texture_worst = st.number_input('Texture_worst', format="%.4f")

# --- Botão e Lógica da Predição ---
st.markdown('---')
if st.button('Obter Diagnóstico'):

    # 1. Coleta e Pré-processamento (Como na Fase 1)
    input_data_list = [
        radius_mean,
        perimeter_mean,
        area_mean,
        radius_worst,
        perimeter_worst,
        area_worst,
        area_se,
        texture_worst
    ]

    input_data = np.array([input_data_list])

    # Pré-processar os dados de entrada com o scaler
    input_data_scaled = scaler.transform(input_data)

    # 2. Previsão com o Modelo OTIMIZADO (GA)
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    diagnostico_modelo = "Maligno" if prediction[0] == 1 else "Benigno"
    probabilidade = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

    # 3. Exibir Resultado do Modelo ML
    st.subheader('Resultado da Análise (Modelo ML Otimizado)')
    if prediction[0] == 1:
        st.error(f'### Diagnóstico ML: Provável tumor **{diagnostico_modelo.upper()}** 🔴')
        st.write(f'O modelo Decision Tree (Otimizado por GA) prevê com **{probabilidade:.2f}** de confiança.')
    else:
        st.success(f'### Diagnóstico ML: Provável tumor **{diagnostico_modelo.upper()}** 🟢')
        st.write(f'O modelo Decision Tree (Otimizado por GA) prevê com **{probabilidade:.2f}** de confiança.')

    st.markdown('---')
    st.subheader('Interpretação da IA (LLM)')

    # --- 4. Geração de Explicação com o LLM (FASE 2) ---

    # Verifica se a chave foi carregada
    if not GEMINI_API_KEY:
        st.error("Chave de API do Gemini não encontrada. Verifique o arquivo .env.")
        st.stop()

    try:
        # Configura o LLM com a chave
        genai.configure(api_key=GEMINI_API_KEY)

        # Prepara os dados de entrada para o prompt
        feature_names = ['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst',
                         'perimeter_worst', 'area_worst', 'area_se', 'texture_worst']
        dados_entrada_df = pd.DataFrame([input_data_list], columns=feature_names)

        # Prompt Engineering (Instrução para o LLM)
        prompt = f"""
        Você é um Analista de IA Médica. Sua tarefa é transformar o diagnóstico do modelo de Machine Learning
        em um resumo acionável de 3-4 frases para um médico.

        Dados do Modelo:
        - Previsão do Modelo: {diagnostico_modelo}
        - Confiança (Probabilidade): {probabilidade:.2f}
        - Valores de Entrada do Paciente (Todas as 8 Features): {dados_entrada_df.to_dict('records')}

        Sua resposta deve:
        1. Confirmar o diagnóstico com a confiança.
        2. Gerar uma breve explicação, mencionando as features que mais impactaram a decisão (especialmente aquelas com valores extremos, como o 'perimeter_worst').
        3. Concluir com uma recomendação (ex: "Recomendação de rotina" ou "Urgência de biópsia").
        """

        # Gerar a explicação
        with st.spinner("Gerando explicação detalhada com o LLM..."):
            model_llm = genai.GenerativeModel(LLM_MODEL)
            response = model_llm.generate_content(prompt)

        # Exibir a Explicação do LLM
        st.info(response.text)

    except Exception as e:
        st.error(f"Ocorreu um erro ao conectar com o LLM (Gemini): {e}")
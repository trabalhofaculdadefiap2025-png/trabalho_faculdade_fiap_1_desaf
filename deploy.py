import streamlit as st
import joblib
import numpy as np

# --- Título da Aplicação ---
st.title('Sistema de Suporte ao Diagnóstico de Câncer de Mama')
st.write('Insira os dados do paciente para obter uma análise inicial.')

# --- Carregar o Modelo e o Scaler ---
MODEL_FILE = 'decision_tree_model.joblib'
SCALER_FILE = 'minmax_scaler.joblib'

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    st.success('Modelo e Scaler carregados com sucesso! Pronto para a análise.')
except FileNotFoundError:
    st.error('Erro: Arquivos de modelo ou scaler não encontrados. Verifique se eles estão na mesma pasta.')
    st.stop()

# --- Criar o Formulário de Entrada de Dados (APENAS 8 COLUNAS) ---
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
    # Coletar os dados de entrada em uma lista na ordem correta
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

    # Fazer a previsão
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Exibir o resultado
    st.subheader('Resultado da Análise')
    if prediction[0] == 1:
        st.error('### Diagnóstico: Provável tumor **Maligno** 🔴')
        st.write(f'A probabilidade de ser maligno é de **{prediction_proba[0][1]:.2f}**.')
        st.write('Recomendado acompanhamento médico imediato e exames complementares.')
    else:
        st.success('### Diagnóstico: Provável tumor **Benigno** 🟢')
        st.write(f'A probabilidade de ser benigno é de **{prediction_proba[0][0]:.2f}**.')
        st.write('Recomendado acompanhamento médico de rotina.')
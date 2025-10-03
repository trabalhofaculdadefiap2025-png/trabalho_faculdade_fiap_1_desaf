# [TECH CHALLENGE IADT] Sistema de Suporte ao Diagnóstico de Câncer de Mama

## 💡 Descrição do Projeto
Este projeto implementa uma solução de Machine Learning (Decision Tree e Random Forest) treinada em dados balanceados e escalonados para auxiliar médicos na análise inicial de exames de câncer de mama. O sistema utiliza um modelo de classificação para prever a probabilidade de um tumor ser maligno ou benigno. O deploy é feito via Streamlit, oferecendo uma interface intuitiva para uso clínico.

---

## 💾 Dataset e Fonte de Dados
O projeto utiliza o conjunto de dados de diagnóstico de câncer de mama (Wisconsin Breast Cancer) para modelagem.

**Link para Download:** [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data]

---

## 💻 Estrutura do Repositório

| Arquivo/Pasta | Função |
| :--- | :--- |
| `deploy.py` | Código-fonte do aplicativo Streamlit (a interface de usuário). |
| `algoritmo_aplicado.py` | Script de treinamento, balanceamento e validação dos modelos (Decision Tree e Random Forest). |
| `analise_exploratoria.py` | Script para EDA, visualização de correlações e análise bivariada. |
| `decision_tree_model.joblib` | **Modelo de Decision Tree** treinado, salvo para uso no deploy. |
| `minmax_scaler.joblib` | Objeto **MinMaxScaler** treinado, salvo para pré-processar dados de entrada. |
| `requirements.txt` | Lista de todas as bibliotecas Python necessárias. |
| `Dockerfile` | Receita para construir o container Docker, garantindo a reprodutibilidade. |

---

## ⚙️ Instruções de Execução (via Docker)

Para rodar o aplicativo de forma isolada e sem instalar dependências locais, siga os passos abaixo:

### Pré-requisito
Você precisa ter o **Docker** instalado e funcionando em sua máquina.

### Passo 1: Clonar o Repositório
```bash
git clone [INSIRA A URL DO SEU REPOSITÓRIO AQUI]
cd [NOME_DO_SEU_REPOSITORIO]

### Passo 2: Construir a Imagem Docker
docker build -t suporte-diagnostico .

### Passo 3: Rodar o Container
docker run -p 8501:8501 suporte-diagnostico

🔬 Resultados Obtidos e Relatório
1. Estratégias de Pré-processamento
Seleção de Features: Foquei  em 8 features principais (Radius_mean,Perimeter_mean,Area_mean,Radius_worst,Perimeter_worst,Area_worst,Area_se e Texture_worst.) para reduzir a multicolinearidade e simplificar o modelo, mantendo o poder preditivo.
Observação: Para a escolha dessas colunas foi feita uma análise usando Mapa de calor e Box Plot.  

Balanceamento de Classes: Indentifiquei um  o desequilíbrio na base de treino e apliquei  o RandomOverSampler (imblearn) apenas nos dados de treino para equalizar o número de casos Malignos e Benignos e eliminar o viés, evitando que o modelo visse apenas muitos dados de uma determina classe e pouco de outra.

Escalonamento: O MinMaxScaler foi usado para normalizar as features, garantindo que o modelo Decision Tree as trate com pesos iguais.

2. Modelos Usados e Justificativa
Modelo: Decision Tree Classifier
Justificativa: Ofereceu um excelente equilíbrio entre Precisão e Recall e, crucialmente, alta Interpretabilidade (Feature Importance). Foi o modelo escolhido para o deploy.

Modelo:Random Forest Classifier
Justificativa: Ofereceu um excelente equilíbrio entre Precisão e Recall e, crucialmente, alta Interpretabilidade, porém o tempo de processamento foi elevado, fazendo com que ele fosse descartado para a etapa do deploy.

🔬 3. Métricas e Análise de Desempenho
Modelo DECISION TREE CLASSIFIER
A. Feature Importance (Importância das Features)
O Feature Importance é uma técnica que mostra quais características do tumor foram mais decisivas na tomada de decisão do modelo.

Feature	Valor de Importância
perimeter_worst	0.751879
texture_worst	0.097178
area_mean	0.052566
area_worst	0.032957
radius_mean	0.022843
area_se	0.021878
radius_worst	0.016609
perimeter_mean	0.004091

Interpretação: A característica perimeter_worst (perímetro do tumor no pior caso) é, de longe, o fator mais importante para a decisão do modelo (responsável por mais de 75% da importância total). Isso justifica a seleção dessa feature, provando que ela é a principal preditora do diagnóstico.

B. Relatório de Classificação (Decision Tree)
Classe	Precision	Recall	F1-Score	Support
Benigno (0)	0.92	0.90	0.91	61
Maligno (1)	0.88	0.90	0.89	51
Accuracy (Total)			0.90	112

Explicação das Métricas:

Precision (Precisão): De todas as vezes que o modelo previu Maligno, ele acertou 88% delas.

Recall (Sensibilidade): De todos os casos Malignos que realmente existiam, o modelo identificou corretamente 90% deles. Em um diagnóstico médico, o Recall alto é crucial para evitar Falsos Negativos (ignorar um câncer).

F1-Score: É a média harmônica entre Precision e Recall. Um F1-Score de 0.89 para a classe Maligna indica um desempenho geral muito forte.

Accuracy (Acurácia): O modelo acertou o diagnóstico em 90% das vezes.

C. Matriz de Confusão (Decision Tree)
Previsão: Benigno	Previsão: Maligno	Total Real
Benigno	55 (Verdadeiros Negativos)	6 (Falsos Positivos)	61
Maligno	5 (Falsos Negativos)	46 (Verdadeiros Positivos)	51
Total Previsto	60	52	112

Explicação: A matriz de confusão visualiza a performance. O resultado mais importante é a baixa incidência de Falsos Negativos (casos reais de câncer que o modelo classificou como benignos). O modelo Decision Tree demonstra segurança ao manter esse erro crítico em um nível baixo.

Modelo de Comparação: RANDOM FOREST CLASSIFIER
O Random Forest foi usado para validar se um modelo mais complexo traria ganhos significativos.

Relatório de Classificação (Random Forest)
Classe	Precision	Recall	F1-Score	Support
Benigno (0)	0.88	0.95	0.91	61
Maligno (1)	0.93	0.84	0.89	51
Accuracy (Total)			0.90	112

Matriz de Confusão (Random Forest)
Previsão: Benigno	Previsão: Maligno	Total Real
Benigno	58 (Verdadeiros Negativos)	3 (Falsos Positivos)	61
Maligno	8 (Falsos Negativos)	43 (Verdadeiros Positivos)	51
Total Previsto	66	46	112

Conclusão: O Random Forest teve uma Precision ligeiramente maior para a classe Maligna (0.93 vs. 0.88), mas um Recall pior (0.84 vs. 0.90). Como o Recall (identificar o câncer real) é mais importante em um diagnóstico, a Decision Tree é o modelo mais adequado e seguro para o deploy, apesar da complexidade do Random Forest.





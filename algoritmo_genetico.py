import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


from deap import base, creator, tools, algorithms


# 1. Carregar Dados e Definir Features
df = pd.read_csv(r"C:\Users\333025\Documents\tech_01\database\data.csv")

features_to_use = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst',
    'perimeter_worst', 'area_worst', 'area_se', 'texture_worst',
    'diagnosis'
]
df_selected = df[features_to_use]

# Preparação de X e y
X = df_selected.drop('diagnosis', axis=1)
y = df_selected['diagnosis']
y = y.map({'M': 1, 'B': 0})

# Divisão de Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceamento (Oversampling)
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Escalonamento
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)



# Funções auxiliares para mapear valores categóricos (atendendo ao requisito de codificação)
def map_criterion(value):
    return 'gini' if value == 0 else 'entropy'


def map_splitter(value):
    return 'best' if value == 0 else 'random'


# A função fitness recebe o indivíduo (cromossomo) e retorna a performance.
# O GA tenta MAXIMIZAR este valor.
def evaluate_individual(individual):
    """
    Treina e avalia o Decision Tree com base nos hiperparâmetros (genes) fornecidos.

    Genes (individual): [max_depth, min_samples_leaf, criterion, splitter]
    """

    max_depth = individual[0]  # Inteiro
    min_samples_leaf = individual[1]  # Inteiro
    criterion = map_criterion(individual[2])  # Categórico ('gini' ou 'entropy')
    splitter = map_splitter(individual[3])  # Categórico ('best' ou 'random')

    # 1. Instanciar o modelo com os hiperparâmetros decodificados
    dtree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        random_state=42
    )

    # 2. Treinar o modelo nos dados balanceados e escalonados
    dtree.fit(X_train_scaled, y_train_resampled)

    # 3. Fazer a predição no conjunto de teste (Original)
    y_pred = dtree.predict(X_test_scaled)

    # 4. Calcular a métrica de fitness (Usaremos o Recall para a classe Maligna - 1)
    # Queremos MAXIMIZAR a capacidade de identificar corretamente o câncer.
    fitness_score = recall_score(y_test, y_pred, pos_label=1)

    # Retorna o valor do fitness (como uma tupla, obrigatório pelo DEAP)
    return fitness_score,


# --- 3. CONFIGURAÇÃO DO ALGORITMO GENÉTICO (DEAP) ---

# A. Criar a classe Fitness e o Indivíduo
# fitness_weights=(1.0,) significa que queremos MAXIMIZAR o valor (Recall)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# B. Configurar o Toolbox (Receita do GA)
toolbox = base.Toolbox()

# 1. Codificação (Definição dos Genes)
# Definir as faixas de valores (atendendo ao requisito de codificação)
toolbox.register("attr_max_depth", np.random.randint, 2, 16)  # 2 a 15
toolbox.register("attr_min_samples_leaf", np.random.randint, 1, 11)  # 1 a 10
toolbox.register("attr_criterion", np.random.randint, 0, 2)  # 0 ou 1
toolbox.register("attr_splitter", np.random.randint, 0, 2)  # 0 ou 1

# 2. Inicialização do Indivíduo (Cromossomo)
# O cromossomo é a lista de genes: [max_depth, min_samples_leaf, criterion, splitter]
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_max_depth,
                  toolbox.attr_min_samples_leaf,
                  toolbox.attr_criterion,
                  toolbox.attr_splitter), n=1)

# 3. Inicialização da População
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 4. Operadores Genéticos (Atendendo ao requisito de operadores)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)  # Seleção por Torneio
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Cruzamento Uniforme
toolbox.register("mutate", tools.mutUniformInt, low=[2, 1, 0, 0], up=[15, 10, 1, 1], indpb=0.05)  # Mutação com limites


# --- 4. EXECUÇÃO DO ALGORITMO GENÉTICO ---

def run_ga(pop_size, cx_rate, mut_rate, n_generations):
    # Logging inicial para tracking de desempenho (atendendo ao requisito de logging)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop = toolbox.population(n=pop_size)  # Cria a população

    # O algoritmo 'eaSimple' executa o GA
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=cx_rate,  # Taxa de Cruzamento
                                   mutpb=mut_rate,  # Taxa de Mutação
                                   ngen=n_generations,  # Número de gerações
                                   stats=stats,
                                   verbose=True)

    return pop, log


def run_ga(pop_size, cx_rate, mut_rate, n_generations):
    # Função para rodar o GA (mantida igual)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop = toolbox.population(n=pop_size)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=cx_rate,
                                   mutpb=mut_rate,
                                   ngen=n_generations,
                                   stats=stats,
                                   verbose=True)

    # Retorna a população final e o livro de logs
    return pop, log


# --- 5. EXECUÇÃO DE TODOS OS EXPERIMENTOS E SALVAMENTO DO MELHOR MODELO ---

def run_all_experiments():
    # Lista para armazenar o melhor resultado de cada experimento
    best_individuals = []

    # Configurações de todos os experimentos (GA)
    experiments = [
        {"name": "Experimento 1 (Base)", "pop_size": 50, "cx_rate": 0.7, "mut_rate": 0.2, "n_gen": 20},
        {"name": "Experimento 2 (Pop. Maior)", "pop_size": 100, "cx_rate": 0.7, "mut_rate": 0.2, "n_gen": 20},
        {"name": "Experimento 3 (Mutação Alta)", "pop_size": 50, "cx_rate": 0.7, "mut_rate": 0.5, "n_gen": 20},
    ]

    for exp in experiments:
        print(f"\n--- {exp['name']} ---")
        pop, logbook = run_ga(exp['pop_size'], exp['cx_rate'], exp['mut_rate'], exp['n_gen'])

        # Selecionar o melhor indivíduo do experimento
        best_ind = tools.selBest(pop, 1)[0]
        best_individuals.append(best_ind)

        print(f"Melhores Hiperparâmetros: {best_ind}")
        print(f"Recall Final (Fitness): {best_ind.fitness.values[0]:.4f}")

    # --- Comparação Final e Salvamento ---

    # Encontrar o melhor indivíduo entre todos os experimentos
    all_pop = [ind for pop in best_individuals for ind in [pop]]
    best_of_all = tools.selBest(all_pop, 1)[0]

    print("\n\n=======================================================")
    print("  RESULTADO FINAL: MODELO OTIMIZADO PELO GA")
    print("=======================================================")
    print(f"Melhores Hiperparâmetros (Geral): {best_of_all}")
    print(f"Melhor Recall Máximo Alcançado: {best_of_all.fitness.values[0]:.4f}")

    # Decodificar e Salvar o Melhor Modelo Otimizado
    max_depth = best_of_all[0]
    min_samples_leaf = best_of_all[1]
    criterion = map_criterion(best_of_all[2])
    splitter = map_splitter(best_of_all[3])

    dtree_optimized = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        random_state=42
    )

    dtree_optimized.fit(X_train_scaled, y_train_resampled)

    # Salvar o modelo OTIMIZADO para o deploy, substituindo o anterior!
    joblib.dump(dtree_optimized, 'decision_tree_model_GA_optimized.joblib')
    print("\n=> Modelo Otimizado pelo GA salvo com sucesso como 'decision_tree_model_GA_optimized.joblib'.")


if __name__ == "__main__":
    run_all_experiments()


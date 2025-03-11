import matplotlib.pyplot as plt
import seaborn as sns

# Visualize data and results
def plot_data(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='SquareFeet', y='Price', data=df)
    plt.title('Tamanho da Casa vs Preço')
    plt.xlabel('Tamanho (em pés quadrados)')
    plt.ylabel('Preço')
    plt.show()

def plot_model_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green')
    plt.title('Preço Real vs Preço Previsto')
    plt.xlabel('Preço Real')
    plt.ylabel('Preço Previsto')
    plt.show()

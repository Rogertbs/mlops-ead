# 🧠 Treinamento de Modelo de Classificação de Saúde Fetal

Este script Python (`train_notebook.py`) implementa um pipeline completo de Machine Learning, focando no **pré-processamento de dados**, **treinamento** de um modelo de **Deep Learning** (Rede Neural Densa) e **rastreamento** do experimento usando **MLflow**.

---

## 🎯 O que o Código Faz

O código executa as seguintes etapas:

1.  **Carregamento de Dados:** Baixa o dataset `fetal_health_reduced.csv` do GitHub (repositório `lectures-cdas-2023`).
2.  **Pré-processamento:**
    * **Normaliza** (StandardScaler) as features do dataset.
    * **Divide** os dados em conjuntos de treino e teste (`test_size=0.3`).
    * **Ajusta** os labels da variável target (`fetal_health`) subtraindo 1 para adequar-se à indexação de classes (0, 1, 2).
3.  **Configuração do Modelo:** Cria uma **Rede Neural Sequencial Densa** (DNN) para classificação de 3 classes, usando:
    * Duas camadas ocultas (`Dense`) com 10 unidades e ativação **ReLU**.
    * Uma camada de saída (`Dense`) com 3 unidades e ativação **Softmax**.
    * Compilação com `loss='sparse_categorical_crossentropy'` e otimizador **Adam**.
4.  **Rastreamento (MLflow):** Configura as credenciais e o URI de tracking para registrar o experimento no **DagsHub**. O `mlflow.keras.autolog` é ativado para rastrear automaticamente parâmetros e métricas durante o treinamento.
5.  **Treinamento:** Treina o modelo usando o método `.fit()` por **50 epochs**, registrando todas as informações no MLflow.

---

## 💻 Modelo de Deep Learning

| Tipo | Arquitetura | Objetivo |
| :--- | :--- | :--- |
| **Modelo** | Rede Neural Densa (DNN) Sequencial | Classificação |
| **Dados** | Indicadores de Saúde Fetal | Prever 3 classes de saúde fetal. |
| **Camadas** | `InputLayer`, `Dense(10, relu)`, `Dense(10, relu)`, `Dense(3, softmax)` | |
| **Otimizador** | Adam | |
| **Loss** | `sparse_categorical_crossentropy` | Adequado para classificação multi-classe inteira. |

---

## 🚀 Como Rodar o Código

Para garantir que todas as dependências estejam isoladas, utilize um **Ambiente Virtual (`venv`)**.

### 1. Pré-requisitos

Você precisará ter o Python (versão compatível com TensorFlow/Keras, idealmente Python 3.9+) instalado.

### 2. Configurar o Ambiente Virtual

Crie e ative o ambiente virtual:

```bash
# Crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente virtual
# No Linux/macOS:
source venv/bin/activate
# No Windows (Command Prompt):
# venv\Scripts\activate.bat
# No Windows (PowerShell):
# venv\Scripts\Activate.ps1

### 3. Instalar Dependências
Este script requer TensorFlow/Keras, MLflow, Pandas e Scikit-learn.

Crie um arquivo requirements.txt com o seguinte conteúdo:

```bash
tensorflow
keras
mlflow
pandas
matplotlib
scikit-learn
Em seguida, instale:

Bash

```bash
pip install -r requirements.txt

### 4. Executar o Script
Com o ambiente ativado e as dependências instaladas, execute o arquivo:


```bash
python train_notebook.py

O script será executado, o modelo será treinado por 50 epochs e o experimento será registrado no MLflow.



### 3. Após configurar o ambiente virtual e as variáveis de .env execute o fastApi
```bash
uvicorn app.main:app --host 0.0.0.0 --reload


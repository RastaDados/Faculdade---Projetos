# Código Do Projeto

```python
#Imports das bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
#Importando o warnings para o python não encher o saco com avisos, k
import warnings
warnings.filterwarnings('ignore')

#Carregar os dados
#O Nome é mock_data.csv, gerei com inteligência artificial
data = pd.read_csv('dados/MOCK_DATA.csv')

#Criando os rótulos manualmente, garantindo uma diversidade nos dados
def criar_rotulo(row):
    #Garantindo que alguns casos sejam aprovados
    if row.name % 5 == 0:  #Aprovando 20% dos casos
        return 'Concedido'
    #Setand os critérios para negar o empréstimo
    if row['idade'] < 18 or row['renda'] < 20000:
        return 'Negado'
    if 'Ruim' in row['hist_credito'] or 'Inadimplente' in row['hist_credito']:
        return 'Negado'
    #Aprovando alguns casos aleatórios
    if np.random.rand() > 0.7 and row['renda'] > 30000:
        return 'Concedido'
    return 'Negado'

data['emprestimo'] = data.apply(criar_rotulo, axis=1)

#Processando os features
def extrair_melhor_historico(hist):
    for termo in ['Excelente', 'Bom', 'Regular', 'Ruim', 'Inadimplente']:
        if termo in hist:
            return termo
    return 'Sem histórico'

data['historico'] = data['hist_credito'].apply(extrair_melhor_historico)

#Codificando as variáveis categóricas
le_historico = LabelEncoder().fit(['Excelente', 'Bom', 'Regular', 'Ruim', 'Inadimplente', 'Sem histórico'])
le_emprego = LabelEncoder().fit(data['emprego'])

data['historico_encoded'] = le_historico.transform(data['historico'])
data['emprego_encoded'] = le_emprego.transform(data['emprego'])

#Modelagem dos dados
features = ['idade', 'renda', 'historico_encoded', 'emprego_encoded']
X = data[features]
y = data['emprestimo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

#Avaliando o modelo
print("\nRelatório de Classificação:\n", classification_report(y_test, model.predict(X_test)))

#Criando a função de classificação
def classificar_emprestimo(idade, renda, hist_credito, emprego):
    #Processando o histórico
    historico = extrair_melhor_historico(hist_credito)
    
    try:
        hist_encoded = le_historico.transform([historico])[0]
    except ValueError:
        hist_encoded = le_historico.transform(['Sem histórico'])[0]
    
    try:
        emprego_encoded = le_emprego.transform([emprego])[0]
    except ValueError:
        emprego_encoded = -1
    
    proba = model.predict_proba([[idade, renda, hist_encoded, emprego_encoded]])[0]
    decisao = model.predict([[idade, renda, hist_encoded, emprego_encoded]])[0]
    
    return decisao, proba

#Testando com exemplos que deveriam ser aprovados
exemplos = [
    (35, 85000, "Excelente - Bom", "Senior Developer"),
    (40, 120000, "Excelente", "Director"),
    (30, 60000, "Bom - Regular", "Manager"),
    (45, 90000, "Bom", "Architect")
]

print("\nTestando exemplos que devem ser aprovados:")
for idx, (idade, renda, hist, emprego) in enumerate(exemplos, 1):
    decisao, proba = classificar_emprestimo(idade, renda, hist, emprego)
    print(f"\nExemplo {idx}:")
    print(f"Idade: {idade}, Renda: {renda}, Histórico: {hist}, Emprego: {emprego}")
    print(f"Decisão: {decisao}")
    print(f"Probabilidades: [Negado: {proba[0]:.2f}, Concedido: {proba[1]:.2f}]")
```

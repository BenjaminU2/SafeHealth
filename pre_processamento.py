"""
Projecto 10 - Script 05: Pré-processamento do dataset
NOTA: O texto do FakeRecogna já veio pré-processado (lematizado, sem pontuação).
Por isso este script faz uma limpeza leve, sem relematizar.
"""

import pandas as pd
import re
import numpy as np

# ---------------------------------------------------------------
# 1. Carregar o dataset
# ---------------------------------------------------------------
df = pd.read_csv('dataset_final_projecto10.csv', encoding='utf-8-sig')
print(f"Dataset carregado: {len(df)} exemplos")
print(f"Distribuição original:\n{df['rotulo'].value_counts()}\n")

# ---------------------------------------------------------------
# 2. Limpeza leve (o texto já está pré-processado)
# ---------------------------------------------------------------
def limpar_texto(texto):
    texto = str(texto).lower().strip()
    # remove URLs residuais
    texto = re.sub(r'http\S+|www\S+', '', texto)
    # remove números soltos
    texto = re.sub(r'\b\d+\b', '', texto)
    # remove espaços múltiplos
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Verifica se a coluna 'texto' existe
if 'texto' not in df.columns:
    print("ERRO: Coluna 'texto' não encontrada!")
    print(f"Colunas disponíveis: {list(df.columns)}")
    exit()

df['texto_limpo'] = df['texto'].apply(limpar_texto)

# remove textos que ficaram muito curtos após limpeza
df = df[df['texto_limpo'].str.len() >= 15].reset_index(drop=True)
print(f"Após limpeza: {len(df)} exemplos")

# ---------------------------------------------------------------
# 3. Codificar rótulos numericamente
# ---------------------------------------------------------------
# verdadeiro = 1, falso = 0

# Verifica se a coluna 'rotulo' existe
if 'rotulo' not in df.columns:
    print("ERRO: Coluna 'rotulo' não encontrada!")
    print(f"Colunas disponíveis: {list(df.columns)}")
    exit()

# Mostra valores únicos para debug
print(f"Valores únicos em 'rotulo': {df['rotulo'].unique()}")

df['label'] = df['rotulo'].map({'verdadeiro': 1, 'falso': 0})

# confirmar que não ficaram NaN
nan_count = df['label'].isna().sum()
if nan_count > 0:
    print(f"ATENÇÃO: {nan_count} exemplos com rótulos não reconhecidos!")
    print(f"Valores não mapeados: {df[df['label'].isna()]['rotulo'].unique()}")
    df = df.dropna(subset=['label'])
    print(f"Removidos {nan_count} exemplos com rótulos inválidos")

df['label'] = df['label'].astype(int)

print(f"Distribuição de rótulos:\n{df['rotulo'].value_counts()}\n")
print(f"Distribuição numérica (label):\n{df['label'].value_counts()}\n")

# ---------------------------------------------------------------
# 4. Guardar dataset pré-processado
# ---------------------------------------------------------------
df.to_csv('dataset_preprocessado.csv', index=False, encoding='utf-8-sig')
print("Guardado: dataset_preprocessado.csv")
print("Colunas disponíveis:", list(df.columns))
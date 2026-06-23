"""
Projecto 10 - Script 07: Análise de fairness e gráficos para o relatório
Análise de desempenho por origem do dataset e geração de gráficos finais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score,
                              precision_score, recall_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos bonitos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("PROJECTO 10 - SCRIPT 07: ANÁLISE DE FAIRNESS E GRÁFICOS")
print("=" * 80)

# ---------------------------------------------------------------
# 1. CARREGAR DADOS E TREINAR O MELHOR MODELO
# ---------------------------------------------------------------
print("\n📂 PASSO 1: Carregando dados e treinando modelo...")

df = pd.read_csv('dataset_preprocessado.csv', encoding='utf-8-sig')
X = df['texto_limpo'].astype(str).values
y = df['label'].values
origens = df['origem_dataset'].values

print(f"✅ Dataset carregado: {len(df)} exemplos")
print(f"   • Origem datasets: {df['origem_dataset'].unique()}")

# Divisão treino/teste (80/20)
X_train, X_test, y_train, y_test, orig_train, orig_test = train_test_split(
    X, y, origens, test_size=0.2, stratify=y, random_state=42
)

print(f"   • Treino: {len(X_train)} exemplos")
print(f"   • Teste: {len(X_test)} exemplos")

# Treinar modelo TF-IDF + SVM (melhor da avaliação anterior)
modelo = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ('clf', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42))
])

print("\n⏳ Treinando modelo TF-IDF + SVM...")
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# ---------------------------------------------------------------
# 2. RELATÓRIO GERAL
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 PASSO 2: RELATÓRIO GERAL (CONJUNTO DE TESTE)")
print("=" * 80)

print(classification_report(y_test, y_pred, target_names=['Falso (0)', 'Verdadeiro (1)']))

# Métricas gerais
acc_geral = accuracy_score(y_test, y_pred)
f1_geral = f1_score(y_test, y_pred, average='weighted')
prec_geral = precision_score(y_test, y_pred, average='weighted')
rec_geral = recall_score(y_test, y_pred, average='weighted')

print(f"\n📈 Métricas Gerais:")
print(f"   • Accuracy : {acc_geral:.4f}")
print(f"   • F1       : {f1_geral:.4f}")
print(f"   • Precision: {prec_geral:.4f}")
print(f"   • Recall   : {rec_geral:.4f}")

# ---------------------------------------------------------------
# 3. ANÁLISE DE FAIRNESS POR ORIGEM
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("⚖️ PASSO 3: ANÁLISE DE FAIRNESS POR ORIGEM")
print("=" * 80)

origens_unicas = np.unique(orig_test)
fairness_rows = []

print("\n📋 Desempenho por origem do dataset:")
print("-" * 70)
print(f"{'Origem':<25} {'n':>5} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 70)

for origem in origens_unicas:
    mask = orig_test == origem
    n = mask.sum()
    
    if n < 10:
        print(f"{origem:<25} {n:>5}  {'---':>10} {'---':>10} {'---':>10} {'---':>10} (poucos exemplos)")
        continue
    
    acc = accuracy_score(y_test[mask], y_pred[mask])
    f1 = f1_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
    prec = precision_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
    rec = recall_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
    
    print(f"{origem:<25} {n:>5} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")
    
    fairness_rows.append({
        'origem': origem,
        'n': n,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    })

df_fairness = pd.DataFrame(fairness_rows)
df_fairness.to_csv('resultados_fairness.csv', index=False, encoding='utf-8-sig')
print("\n💾 Guardado: resultados_fairness.csv")

# ---------------------------------------------------------------
# 4. GRÁFICO 1 - MATRIZ DE CONFUSÃO
# ---------------------------------------------------------------
print("\n📊 PASSO 4: GERANDO GRÁFICOS...")

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Falso (0)', 'Verdadeiro (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')

# Adicionar porcentagens
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        texto = f'{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)'
        ax.text(j, i, texto, ha='center', va='center', 
                color='white' if cm[i, j] > total/4 else 'black',
                fontsize=11, fontweight='bold')

ax.set_title('Matriz de Confusão - TF-IDF + SVM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: grafico_matriz_confusao.png")

# ---------------------------------------------------------------
# 5. GRÁFICO 2 - COMPARAÇÃO DE DESEMPENHO POR ORIGEM (FAIRNESS)
# ---------------------------------------------------------------
if len(df_fairness) >= 2:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_fairness))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, df_fairness['accuracy'], width, 
                   label='Accuracy', color='#4C72B0', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, df_fairness['f1'], width, 
                   label='F1-score', color='#DD8452', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, df_fairness['precision'], width, 
                   label='Precision', color='#55A868', edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, df_fairness['recall'], width, 
                   label='Recall', color='#C44E52', edgecolor='black')
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Origem do Dataset', fontsize=12)
    ax.set_ylabel('Pontuação', fontsize=12)
    ax.set_title('Análise de Fairness por Origem do Dataset (TF-IDF + SVM)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_fairness['origem'], rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adicionar linha de referência da média geral
    ax.axhline(y=acc_geral, color='blue', linestyle='--', alpha=0.5, 
               label=f'Média geral (Acc={acc_geral:.3f})')
    ax.axhline(y=f1_geral, color='orange', linestyle='--', alpha=0.5, 
               label=f'Média geral (F1={f1_geral:.3f})')
    
    plt.tight_layout()
    plt.savefig('grafico_fairness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: grafico_fairness.png")

# ---------------------------------------------------------------
# 6. GRÁFICO 3 - PALAVRAS MAIS ASSOCIADAS A CADA CLASSE
# ---------------------------------------------------------------
print("\n🔍 PASSO 5: ANÁLISE DE PALAVRAS-CHAVE")

vectorizer = modelo.named_steps['tfidf']
clf = modelo.named_steps['clf']
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]

top_n = 20
indices_falso = np.argsort(coefs)[:top_n]        # coeficientes negativos = classe falso
indices_verdadeiro = np.argsort(coefs)[-top_n:][::-1]  # coeficientes positivos = classe verdadeiro

print(f"\n📌 Top {top_n} palavras associadas a DESINFORMAÇÃO (falso):")
for i in indices_falso:
    print(f"   {feature_names[i]:35s}  coef: {coefs[i]:.6f}")

print(f"\n📌 Top {top_n} palavras associadas a INFORMAÇÃO VERDADEIRA:")
for i in indices_verdadeiro:
    print(f"   {feature_names[i]:35s}  coef: {coefs[i]:.6f}")

# Gráfico de palavras-chave
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Palavras para a classe Falso
palavras_falso = [feature_names[i] for i in indices_falso]
valores_falso = [coefs[i] for i in indices_falso]
bars1 = ax1.barh(palavras_falso, valores_falso, color='#C44E52', edgecolor='black')
ax1.set_title('Top 20 Palavras - Desinformação (Falso)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Coeficiente SVM', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bar, val in zip(bars1, valores_falso):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', ha='left', fontsize=8)

# Palavras para a classe Verdadeiro
palavras_verd = [feature_names[i] for i in indices_verdadeiro]
valores_verd = [coefs[i] for i in indices_verdadeiro]
bars2 = ax2.barh(palavras_verd, valores_verd, color='#55A868', edgecolor='black')
ax2.set_title('Top 20 Palavras - Informação Verdadeira', fontsize=14, fontweight='bold')
ax2.set_xlabel('Coeficiente SVM', fontsize=12)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bar, val in zip(bars2, valores_verd):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', ha='left', fontsize=8)

plt.suptitle('Padrões Linguísticos Associados a Cada Classe (TF-IDF + SVM)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_palavras_chave.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n   ✅ Guardado: grafico_palavras_chave.png")

# ---------------------------------------------------------------
# 7. GRÁFICO 4 - COMPARAÇÃO DE MÉTRICAS GERAIS
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

metricas_nomes = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
metricas_valores = [acc_geral, f1_geral, prec_geral, rec_geral]
cores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

bars = ax.bar(metricas_nomes, metricas_valores, color=cores, edgecolor='black', alpha=0.8)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Pontuação', fontsize=12)
ax.set_title('Métricas Gerais do Modelo TF-IDF + SVM', fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Adicionar valores nas barras
for bar, val in zip(bars, metricas_valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('grafico_metricas_gerais.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: grafico_metricas_gerais.png")

# ---------------------------------------------------------------
# 8. RESUMO FINAL
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("📋 RESUMO FINAL DO SCRIPT 07")
print("=" * 80)

print(f"""
📊 DESEMPENHO GERAL:
   • Accuracy : {acc_geral:.4f}
   • F1-Score : {f1_geral:.4f}
   • Precision: {prec_geral:.4f}
   • Recall   : {rec_geral:.4f}

⚖️ FAIRNESS POR ORIGEM:
""")

for _, row in df_fairness.iterrows():
    print(f"   • {row['origem']}:")
    print(f"     - Accuracy: {row['accuracy']:.4f} | F1: {row['f1']:.4f} | n={row['n']}")

print(f"""
📁 ARQUIVOS GERADOS:
   1. grafico_matriz_confusao.png - Matriz de confusão do modelo
   2. grafico_fairness.png - Análise de fairness por origem
   3. grafico_palavras_chave.png - Top palavras por classe
   4. grafico_metricas_gerais.png - Métricas gerais do modelo
   5. resultados_fairness.csv - Tabela de fairness
   6. resultados_modelos.csv - Tabela de resultados (já existente)
""")
print("=" * 80)
print("✅ SCRIPT CONCLUÍDO COM SUCESSO!")
print("=" * 80)
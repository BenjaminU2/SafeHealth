"""
Projecto 10 - Script 06: Representações de texto e treino dos modelos
Compara: TF-IDF + SVM, TF-IDF + NaiveBayes, Word2Vec + SVM, fastText + SVM
COM GRÁFICOS INDIVIDUAIS E COMPARAÇÃO FINAL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix
from gensim.models import Word2Vec, FastText
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos bonitos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# ---------------------------------------------------------------
# 1. CARREGAR E ANALISAR O DATASET
# ---------------------------------------------------------------
print("=" * 80)
print("PROJECTO 10 - SCRIPT 06: TREINO DE MODELOS")
print("=" * 80)

print("\n📂 PASSO 1: Carregando dataset pré-processado...")
df = pd.read_csv('dataset_preprocessado.csv', encoding='utf-8-sig')
X = df['texto_limpo'].astype(str).tolist()
y = df['label'].values
textos_tokenizados = [texto.split() for texto in X]

print(f"✅ Dataset carregado com sucesso!")
print(f"   • Total de exemplos: {len(X)}")
print(f"   • Classe 0 (falso): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print(f"   • Classe 1 (verdadeiro): {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

# ---------------------------------------------------------------
# 2. CONFIGURAR VALIDAÇÃO CRUZADA
# ---------------------------------------------------------------
print("\n🔄 PASSO 2: Configurando validação cruzada...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted')
}

print(f"✅ Validação cruzada: {kf.n_splits} folds estratificados")

# ---------------------------------------------------------------
# 3. FUNÇÕES AUXILIARES
# ---------------------------------------------------------------
resultados_todos = []

def plot_resultados_individuais(nome, resultados, X, y, modelo_final=None):
    """Gera gráficos individuais para cada modelo"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Boxplot das métricas nos 5 folds
    ax1 = axes[0]
    metricas = ['accuracy', 'f1', 'precision', 'recall']
    dados_box = []
    for metrica in metricas:
        dados_box.append(resultados[f'test_{metrica}'])
    
    # CORREÇÃO: sem o parâmetro 'labels'
    bp = ax1.boxplot(dados_box, patch_artist=True, showmeans=True)
    ax1.set_xticklabels([m.capitalize() for m in metricas])
    
    # Cores para as caixas
    cores = ['#45b7d1', '#ff6b6b', '#4ecdc4', '#96ceb4']
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    
    ax1.set_title(f'{nome} - Distribuição das Métricas (5 folds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pontuação')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores médios
    for i, metrica in enumerate(metricas):
        media = np.mean(resultados[f'test_{metrica}'])
        ax1.text(i+1, 0.02, f'μ={media:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='darkblue')
    
    # 2. Evolução das métricas por fold
    ax2 = axes[1]
    folds = range(1, 6)
    for metrica in metricas:
        ax2.plot(folds, resultados[f'test_{metrica}'], 'o-', label=metrica.capitalize(), 
                linewidth=2, markersize=8)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Pontuação')
    ax2.set_title(f'{nome} - Desempenho por Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(folds)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Matriz de Confusão (se tiver modelo final)
    ax3 = axes[2]
    if modelo_final is not None:
        y_pred = modelo_final.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=['Falso (0)', 'Verdadeiro (1)'],
                    yticklabels=['Falso (0)', 'Verdadeiro (1)'])
        ax3.set_title(f'{nome} - Matriz de Confusão', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predito')
        ax3.set_ylabel('Real')
        
        # Calcular métricas da matriz
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ax3.text(0.5, -0.15, f'Acc: {acc:.3f} | Sens: {sens:.3f} | Spec: {spec:.3f}', 
                ha='center', va='top', transform=ax3.transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    nome_arquivo = f'grafico_{nome.replace(" ", "_").replace("+", "e")}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   📊 Gráfico salvo: {nome_arquivo}")

def avaliar_modelo(nome, pipeline, X, y, kf, salvar_grafico=True):
    """Avalia o modelo e gera gráficos individuais"""
    
    print(f"\n⏳ Treinando: {nome}...")
    
    resultados = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=-1, 
                                return_estimator=True)
    
    # Calcular médias
    media_acc = resultados['test_accuracy'].mean()
    media_f1 = resultados['test_f1'].mean()
    media_prec = resultados['test_precision'].mean()
    media_rec = resultados['test_recall'].mean()
    
    print(f"✅ {nome} - Resultados:")
    print(f"   • Accuracy : {media_acc:.4f} ± {resultados['test_accuracy'].std():.4f}")
    print(f"   • F1       : {media_f1:.4f} ± {resultados['test_f1'].std():.4f}")
    print(f"   • Precision: {media_prec:.4f} ± {resultados['test_precision'].std():.4f}")
    print(f"   • Recall   : {media_rec:.4f} ± {resultados['test_recall'].std():.4f}")
    
    # Salvar resultados
    resultado = {
        'modelo': nome,
        'accuracy': media_acc,
        'f1': media_f1,
        'precision': media_prec,
        'recall': media_rec,
        'accuracy_std': resultados['test_accuracy'].std(),
        'f1_std': resultados['test_f1'].std(),
        'resultados': resultados
    }
    
    # Gerar gráficos individuais
    if salvar_grafico:
        # Treinar modelo final para matriz de confusão
        modelo_final = pipeline
        modelo_final.fit(X, y)
        plot_resultados_individuais(nome, resultados, X, y, modelo_final)
    
    return resultado

# ---------------------------------------------------------------
# 4. TREINO DOS MODELOS (UM POR UM)
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("📝 PASSO 3: TREINO DOS MODELOS")
print("=" * 80)

# 4.1 MODELO 1: TF-IDF + SVM
print("\n🔹 MODELO 1: TF-IDF + SVM")
print("   • Representação: TF-IDF com 20.000 features e n-gramas (1,2)")
print("   • Classificador: LinearSVC com class_weight='balanced'")

pipe_tfidf_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ('clf', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42))
])

resultados_todos.append(avaliar_modelo("TF-IDF + SVM", pipe_tfidf_svm, X, y, kf))

# 4.2 MODELO 2: TF-IDF + ComplementNB
print("\n🔹 MODELO 2: TF-IDF + Naive Bayes (Complement)")
print("   • Representação: TF-IDF com 20.000 features e n-gramas (1,2)")
print("   • Classificador: ComplementNB - ideal para dados desbalanceados")

pipe_tfidf_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ('clf', ComplementNB())
])

resultados_todos.append(avaliar_modelo("TF-IDF + ComplementNB", pipe_tfidf_nb, X, y, kf))

# 4.3 MODELO 3: Word2Vec + SVM
print("\n🔹 MODELO 3: Word2Vec + SVM")
print("   • Representação: Word2Vec (100 dimensões, janela=5)")
print("   • Classificador: SVM RBF com class_weight='balanced'")

print("\n⏳ Treinando Word2Vec... (pode demorar 1-2 minutos)")

w2v_model = Word2Vec(
    sentences=textos_tokenizados,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    epochs=10
)

print(f"✅ Word2Vec treinado! Vocabulário: {len(w2v_model.wv)} palavras")

def texto_para_vetor_w2v(texto, model):
    palavras = texto.split()
    vetores = [model.wv[p] for p in palavras if p in model.wv]
    if vetores:
        return np.mean(vetores, axis=0)
    else:
        return np.zeros(model.vector_size)

X_w2v = np.array([texto_para_vetor_w2v(t, w2v_model) for t in X])

svm_w2v = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=False)
resultados_w2v = cross_validate(svm_w2v, X_w2v, y, cv=kf, scoring=scoring, n_jobs=-1)

print(f"✅ Word2Vec + SVM - Resultados:")
print(f"   • Accuracy : {resultados_w2v['test_accuracy'].mean():.4f} ± {resultados_w2v['test_accuracy'].std():.4f}")
print(f"   • F1       : {resultados_w2v['test_f1'].mean():.4f} ± {resultados_w2v['test_f1'].std():.4f}")
print(f"   • Precision: {resultados_w2v['test_precision'].mean():.4f} ± {resultados_w2v['test_precision'].std():.4f}")
print(f"   • Recall   : {resultados_w2v['test_recall'].mean():.4f} ± {resultados_w2v['test_recall'].std():.4f}")

# Gráfico individual para Word2Vec
svm_w2v.fit(X_w2v, y)
resultado_w2v = {
    'modelo': 'Word2Vec + SVM',
    'accuracy': resultados_w2v['test_accuracy'].mean(),
    'f1': resultados_w2v['test_f1'].mean(),
    'precision': resultados_w2v['test_precision'].mean(),
    'recall': resultados_w2v['test_recall'].mean(),
    'accuracy_std': resultados_w2v['test_accuracy'].std(),
    'f1_std': resultados_w2v['test_f1'].std(),
    'resultados': resultados_w2v
}
resultados_todos.append(resultado_w2v)
plot_resultados_individuais("Word2Vec + SVM", resultados_w2v, X_w2v, y, svm_w2v)

# 4.4 MODELO 4: fastText + SVM
print("\n🔹 MODELO 4: fastText + SVM")
print("   • Representação: fastText (100 dimensões, janela=5)")
print("   • Classificador: SVM RBF com class_weight='balanced'")

print("\n⏳ Treinando fastText... (pode demorar 1-2 minutos)")

ft_model = FastText(
    sentences=textos_tokenizados,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    epochs=10
)

print(f"✅ fastText treinado! Vocabulário: {len(ft_model.wv)} palavras")

def texto_para_vetor_ft(texto, model):
    palavras = texto.split()
    vetores = [model.wv[p] for p in palavras if p in model.wv]
    if vetores:
        return np.mean(vetores, axis=0)
    else:
        return np.zeros(model.vector_size)

X_ft = np.array([texto_para_vetor_ft(t, ft_model) for t in X])

svm_ft = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=False)
resultados_ft = cross_validate(svm_ft, X_ft, y, cv=kf, scoring=scoring, n_jobs=-1)

print(f"✅ fastText + SVM - Resultados:")
print(f"   • Accuracy : {resultados_ft['test_accuracy'].mean():.4f} ± {resultados_ft['test_accuracy'].std():.4f}")
print(f"   • F1       : {resultados_ft['test_f1'].mean():.4f} ± {resultados_ft['test_f1'].std():.4f}")
print(f"   • Precision: {resultados_ft['test_precision'].mean():.4f} ± {resultados_ft['test_precision'].std():.4f}")
print(f"   • Recall   : {resultados_ft['test_recall'].mean():.4f} ± {resultados_ft['test_recall'].std():.4f}")

# Gráfico individual para fastText
svm_ft.fit(X_ft, y)
resultado_ft = {
    'modelo': 'fastText + SVM',
    'accuracy': resultados_ft['test_accuracy'].mean(),
    'f1': resultados_ft['test_f1'].mean(),
    'precision': resultados_ft['test_precision'].mean(),
    'recall': resultados_ft['test_recall'].mean(),
    'accuracy_std': resultados_ft['test_accuracy'].std(),
    'f1_std': resultados_ft['test_f1'].std(),
    'resultados': resultados_ft
}
resultados_todos.append(resultado_ft)
plot_resultados_individuais("fastText + SVM", resultados_ft, X_ft, y, svm_ft)

# ---------------------------------------------------------------
# 5. GRÁFICO DE COMPARAÇÃO FINAL
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 PASSO 4: COMPARAÇÃO FINAL ENTRE MODELOS")
print("=" * 80)

df_resultados = pd.DataFrame(resultados_todos)
df_resultados = df_resultados.round(4)

print("\n📋 Tabela Comparativa de Modelos:")
print(df_resultados[['modelo', 'accuracy', 'f1', 'precision', 'recall']].to_string(index=False))

# Gráfico de comparação
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5.1 Barras comparativas
ax1 = axes[0, 0]
metricas = ['accuracy', 'f1', 'precision', 'recall']
modelos = df_resultados['modelo'].values
x = np.arange(len(modelos))
width = 0.2

for i, metrica in enumerate(metricas):
    values = df_resultados[metrica].values
    bars = ax1.bar(x + i*width, values, width, label=metrica.capitalize())
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Modelos', fontsize=12)
ax1.set_ylabel('Pontuação', fontsize=12)
ax1.set_title('Comparação de Métricas por Modelo', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(modelos, rotation=15, ha='right', fontsize=10)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# 5.2 F1-Score com desvio padrão
ax2 = axes[0, 1]
cores = ['#45b7d1', '#ff6b6b', '#4ecdc4', '#96ceb4']
bars = ax2.bar(modelos, df_resultados['f1'].values, 
              yerr=df_resultados['f1_std'].values,
              color=cores, capsize=10, edgecolor='black', alpha=0.8)
ax2.set_xlabel('Modelos', fontsize=12)
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('F1-Score por Modelo (com desvio padrão)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
for bar, val in zip(bars, df_resultados['f1'].values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5.3 Heatmap de correlação
ax3 = axes[1, 0]
correlacao = df_resultados[['accuracy', 'f1', 'precision', 'recall']].corr()
sns.heatmap(correlacao, annot=True, fmt='.3f', cmap='coolwarm', 
            square=True, linewidths=1, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Correlação entre Métricas', fontsize=14, fontweight='bold')

# 5.4 Radar chart
ax4 = axes[1, 1]
from math import pi
categorias = ['accuracy', 'f1', 'precision', 'recall']
N = len(categorias)
angulos = [n / float(N) * 2 * pi for n in range(N)]
angulos += angulos[:1]

for i, (idx, row) in enumerate(df_resultados.iterrows()):
    valores = row[categorias].values.tolist()
    valores += valores[:1]
    ax4.plot(angulos, valores, 'o-', linewidth=2, label=row['modelo'], 
             color=cores[i % len(cores)])
    ax4.fill(angulos, valores, alpha=0.1, color=cores[i % len(cores)])

ax4.set_xticks(angulos[:-1])
ax4.set_xticklabels([cat.capitalize() for cat in categorias])
ax4.set_ylim(0, 1)
ax4.set_title('Comparação Radar dos Modelos', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
ax4.grid(True)

plt.tight_layout()
plt.savefig('00_comparacao_final_modelos.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfico de comparação salvo: 00_comparacao_final_modelos.png")

# ---------------------------------------------------------------
# 6. RELATÓRIO FINAL
# ---------------------------------------------------------------
print("\n" + "=" * 80)
print("🏆 PASSO 5: ANÁLISE FINAL")
print("=" * 80)

melhor_f1 = df_resultados.loc[df_resultados['f1'].idxmax()]
melhor_acc = df_resultados.loc[df_resultados['accuracy'].idxmax()]

print(f"\n📌 MELHOR POR F1-SCORE:")
print(f"   • Modelo: {melhor_f1['modelo']}")
print(f"   • F1: {melhor_f1['f1']:.4f} ± {melhor_f1['f1_std']:.4f}")
print(f"   • Accuracy: {melhor_f1['accuracy']:.4f} ± {melhor_f1['accuracy_std']:.4f}")

print(f"\n📌 MELHOR POR ACCURACY:")
print(f"   • Modelo: {melhor_acc['modelo']}")
print(f"   • Accuracy: {melhor_acc['accuracy']:.4f} ± {melhor_acc['accuracy_std']:.4f}")
print(f"   • F1: {melhor_acc['f1']:.4f} ± {melhor_acc['f1_std']:.4f}")

# Salvar tabela final
df_resultados.to_csv('resultados_modelos.csv', index=False, encoding='utf-8-sig')

print("\n📁 ARQUIVOS GERADOS:")
print("   • 00_comparacao_final_modelos.png - Comparação entre todos os modelos")
print("   • grafico_TF-IDF_SVM.png - Análise individual")
print("   • grafico_TF-IDF_ComplementNB.png - Análise individual")
print("   • grafico_Word2Vec_SVM.png - Análise individual")
print("   • grafico_fastText_SVM.png - Análise individual")
print("   • resultados_modelos.csv - Tabela com todos os resultados")
print("\n" + "=" * 80)
print("✅ SCRIPT CONCLUÍDO COM SUCESSO!")
print("=" * 80)
"""
Projecto 10 - Script 06: Representações de texto e treino dos modelos
Compara: TF-IDF + SVM, TF-IDF + NaiveBayes, Word2Vec + SVM, fastText + SVM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec, FastText
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 1. Carregar dataset pré-processado
# ---------------------------------------------------------------
df = pd.read_csv('dataset_preprocessado.csv', encoding='utf-8-sig')
X = df['texto_limpo'].astype(str).tolist()
y = df['label'].values
textos_tokenizados = [texto.split() for texto in X]

print(f"Total de exemplos: {len(X)}")
print(f"Classe 0 (falso): {sum(y==0)} | Classe 1 (verdadeiro): {sum(y==1)}\n")

# ---------------------------------------------------------------
# 2. Configurar validação cruzada (k=5, estratificada)
# ---------------------------------------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1':       make_scorer(f1_score, average='weighted'),
    'precision':make_scorer(precision_score, average='weighted', zero_division=0),
    'recall':   make_scorer(recall_score, average='weighted')
}

def avaliar_modelo(nome, pipeline, X, y, kf):
    resultados = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    print(f"--- {nome} ---")
    print(f"  Accuracy : {resultados['test_accuracy'].mean():.4f} ± {resultados['test_accuracy'].std():.4f}")
    print(f"  F1       : {resultados['test_f1'].mean():.4f} ± {resultados['test_f1'].std():.4f}")
    print(f"  Precision: {resultados['test_precision'].mean():.4f} ± {resultados['test_precision'].std():.4f}")
    print(f"  Recall   : {resultados['test_recall'].mean():.4f} ± {resultados['test_recall'].std():.4f}")
    print()
    return {
        'modelo': nome,
        'accuracy': resultados['test_accuracy'].mean(),
        'f1': resultados['test_f1'].mean(),
        'precision': resultados['test_precision'].mean(),
        'recall': resultados['test_recall'].mean(),
        'accuracy_std': resultados['test_accuracy'].std(),
        'f1_std': resultados['test_f1'].std(),
    }

resultados_todos = []

# ---------------------------------------------------------------
# 3. MODELO 1: TF-IDF + SVM (com class_weight para desbalanceamento)
# ---------------------------------------------------------------
print("=== MODELOS TF-IDF ===\n")

pipe_tfidf_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf',   LinearSVC(class_weight='balanced', max_iter=2000, random_state=42))
])
resultados_todos.append(avaliar_modelo("TF-IDF + SVM", pipe_tfidf_svm, X, y, kf))

# ---------------------------------------------------------------
# 4. MODELO 2: TF-IDF + Naive Bayes (ComplementNB - melhor para desbalanceado)
# ---------------------------------------------------------------
pipe_tfidf_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf',   ComplementNB())
])
resultados_todos.append(avaliar_modelo("TF-IDF + Naive Bayes (Complement)", pipe_tfidf_nb, X, y, kf))

# ---------------------------------------------------------------
# 5. MODELO 3: Word2Vec + SVM
# ---------------------------------------------------------------
print("=== MODELOS EMBEDDINGS ===\n")
print("A treinar Word2Vec... (pode demorar 1-2 minutos)")

w2v_model = Word2Vec(
    sentences=textos_tokenizados,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    epochs=10
)

def texto_para_vetor_w2v(texto, model):
    palavras = texto.split()
    vetores = [model.wv[p] for p in palavras if p in model.wv]
    if vetores:
        return np.mean(vetores, axis=0)
    else:
        return np.zeros(model.vector_size)

X_w2v = np.array([texto_para_vetor_w2v(t, w2v_model) for t in X])

# SVM com embeddings Word2Vec (não é pipeline porque os vetores já estão calculados)
from sklearn.svm import SVC
svm_w2v = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=False)
resultados_w2v = cross_validate(svm_w2v, X_w2v, y, cv=kf, scoring=scoring, n_jobs=-1)

print("--- Word2Vec + SVM ---")
print(f"  Accuracy : {resultados_w2v['test_accuracy'].mean():.4f} ± {resultados_w2v['test_accuracy'].std():.4f}")
print(f"  F1       : {resultados_w2v['test_f1'].mean():.4f} ± {resultados_w2v['test_f1'].std():.4f}")
print(f"  Precision: {resultados_w2v['test_precision'].mean():.4f} ± {resultados_w2v['test_precision'].std():.4f}")
print(f"  Recall   : {resultados_w2v['test_recall'].mean():.4f} ± {resultados_w2v['test_recall'].std():.4f}")
print()
resultados_todos.append({
    'modelo': 'Word2Vec + SVM',
    'accuracy': resultados_w2v['test_accuracy'].mean(),
    'f1': resultados_w2v['test_f1'].mean(),
    'precision': resultados_w2v['test_precision'].mean(),
    'recall': resultados_w2v['test_recall'].mean(),
    'accuracy_std': resultados_w2v['test_accuracy'].std(),
    'f1_std': resultados_w2v['test_f1'].std(),
})

# ---------------------------------------------------------------
# 6. MODELO 4: fastText + SVM
# ---------------------------------------------------------------
print("A treinar fastText... (pode demorar 1-2 minutos)")

ft_model = FastText(
    sentences=textos_tokenizados,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    epochs=10
)

def texto_para_vetor_ft(texto, model):
    palavras = texto.split()
    vetores = [model.wv[p] for p in palavras]  # fastText gera vetor para qualquer palavra
    if vetores:
        return np.mean(vetores, axis=0)
    else:
        return np.zeros(model.vector_size)

X_ft = np.array([texto_para_vetor_ft(t, ft_model) for t in X])

svm_ft = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=False)
resultados_ft = cross_validate(svm_ft, X_ft, y, cv=kf, scoring=scoring, n_jobs=-1)

print("--- fastText + SVM ---")
print(f"  Accuracy : {resultados_ft['test_accuracy'].mean():.4f} ± {resultados_ft['test_accuracy'].std():.4f}")
print(f"  F1       : {resultados_ft['test_f1'].mean():.4f} ± {resultados_ft['test_f1'].std():.4f}")
print(f"  Precision: {resultados_ft['test_precision'].mean():.4f} ± {resultados_ft['test_precision'].std():.4f}")
print(f"  Recall   : {resultados_ft['test_recall'].mean():.4f} ± {resultados_ft['test_recall'].std():.4f}")
print()
resultados_todos.append({
    'modelo': 'fastText + SVM',
    'accuracy': resultados_ft['test_accuracy'].mean(),
    'f1': resultados_ft['test_f1'].mean(),
    'precision': resultados_ft['test_precision'].mean(),
    'recall': resultados_ft['test_recall'].mean(),
    'accuracy_std': resultados_ft['test_accuracy'].std(),
    'f1_std': resultados_ft['test_f1'].std(),
})

# ---------------------------------------------------------------
# 7. Guardar tabela de resultados
# ---------------------------------------------------------------
df_resultados = pd.DataFrame(resultados_todos)
df_resultados = df_resultados.round(4)
df_resultados.to_csv('resultados_modelos.csv', index=False, encoding='utf-8-sig')
print("=== TABELA FINAL DE RESULTADOS ===")
print(df_resultados[['modelo','accuracy','f1','precision','recall']].to_string(index=False))
print("\nGuardado: resultados_modelos.csv")
import pandas as pd
import os

print("📁 Arquivos no diretório atual:")
for f in os.listdir('.'):
    print(f"  {f}")

if os.path.exists('dataset_final_projecto10.csv'):
    df = pd.read_csv('dataset_final_projecto10.csv', encoding='utf-8-sig')
    print(f"\n✅ Dataset carregado: {len(df)} linhas")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nPrimeiras 3 linhas:")
    print(df.head(3))
else:
    print("\n❌ dataset_final_projecto10.csv não encontrado!")
    print("Procure o arquivo em outras pastas:")
    for root, dirs, files in os.walk('..'):  # procura na pasta acima também
        for file in files:
            if 'dataset' in file.lower() and file.endswith('.csv'):
                print(f"  - {os.path.join(root, file)}")
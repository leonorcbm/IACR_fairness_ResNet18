import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PASTA_IMAGENS = './img_align_celeba/img_align_celeba/' 
EPOCHS = 2
BATCH_SIZE = 64

# Parâmetros de Mitigação de Viés
LAMBDA_REGULARIZACAO = 0.5  # Peso para o método "Regularização" (Soft)
EPSILON_RESTRICAO = 0.10    # Tolerância máxima de 10% de diferença para "Imposição de Restrições" (Hard)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar: {device}")

# =====================================================================
# 1. CARREGAR E PREPARAR DADOS
# =====================================================================
df_atributos = pd.read_csv("list_attr_celeba.csv").replace(-1, 0)
df_particoes = pd.read_csv("list_eval_partition.csv")
df_completo = pd.merge(df_atributos, df_particoes, on='image_id')

# Vamos usar uma pequena amostra para o código correr rápido no computador
print("A preparar os dados de treino e teste...")
df_treino = df_completo[df_completo['partition'] == 0].sample(n=3000, random_state=42).reset_index(drop=True)
df_teste = df_completo[df_completo['partition'] == 2].sample(n=1000, random_state=42).reset_index(drop=True)

# Atributo Sensível e Variável Alvo
col_A = "Male"
col_Y = "Smiling"

# =====================================================================
# 2. DEFINIR O DATASET PARA O PYTORCH
# =====================================================================
class DatasetSimples(Dataset):
    def __init__(self, df, pasta, col_Y, col_A):
        self.df = df
        self.pasta = pasta
        self.col_Y = col_Y
        self.col_A = col_A
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nome_img = self.df.loc[idx, 'image_id']
        caminho = os.path.join(self.pasta, nome_img)
        imagem = Image.open(caminho).convert('RGB')
        imagem = self.transform(imagem)
            
        y_real = torch.tensor(self.df.loc[idx, self.col_Y], dtype=torch.float32)
        a_real = torch.tensor(self.df.loc[idx, self.col_A], dtype=torch.float32)
        
        return imagem, y_real, a_real

loader_treino = DataLoader(DatasetSimples(df_treino, PASTA_IMAGENS, col_Y, col_A), batch_size=BATCH_SIZE, shuffle=True)
loader_teste = DataLoader(DatasetSimples(df_teste, PASTA_IMAGENS, col_Y, col_A), batch_size=BATCH_SIZE, shuffle=False)

# =====================================================================
# 3. FUNÇÕES DE AVALIAÇÃO (OS 3 PILARES DO FAIRNESS)
# =====================================================================
def metricas_fairness(y_real, y_pred, A_real):
    """
    Calcula as métricas de fairness ensinadas na Aula 7 com código muito legível.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    A_real = np.array(A_real)
    
    # Índices de cada grupo
    g0 = (A_real == 0) # Grupo Mulheres
    g1 = (A_real == 1) # Grupo Homens
    
    # ---------------------------------------------------------
    # PILAR 1: Independência (Paridade Demográfica)
    # Taxa de previsões positivas em cada grupo
    taxa_g0 = np.mean(y_pred[g0] == 1)
    taxa_g1 = np.mean(y_pred[g1] == 1)
    
    spd = abs(taxa_g0 - taxa_g1) # Ideal = 0
    di = taxa_g0 / taxa_g1 if taxa_g1 > 0 else 0 # Regra dos 80% (0.8 a 1.25)
    
    # ---------------------------------------------------------
    # PILAR 2: Separação (Igualdade de Oportunidades - TPR e FPR)
    # TPR: Acertos nos positivos
    tpr_g0 = np.sum((y_pred[g0] == 1) & (y_real[g0] == 1)) / (np.sum(y_real[g0] == 1) + 1e-5)
    tpr_g1 = np.sum((y_pred[g1] == 1) & (y_real[g1] == 1)) / (np.sum(y_real[g1] == 1) + 1e-5)
    diff_tpr = abs(tpr_g0 - tpr_g1)

    # FPR: Erros nos negativos
    fpr_g0 = np.sum((y_pred[g0] == 1) & (y_real[g0] == 0)) / (np.sum(y_real[g0] == 0) + 1e-5)
    fpr_g1 = np.sum((y_pred[g1] == 1) & (y_real[g1] == 0)) / (np.sum(y_real[g1] == 0) + 1e-5)
    diff_fpr = abs(fpr_g0 - fpr_g1)
    
    # ---------------------------------------------------------
    # PILAR 3: Suficiência (Calibração / Paridade Preditiva - PPV)
    # PPV: De todos os que o modelo disse ser '1', quantos eram mesmo '1'?
    ppv_g0 = np.sum((y_pred[g0] == 1) & (y_real[g0] == 1)) / (np.sum(y_pred[g0] == 1) + 1e-5)
    ppv_g1 = np.sum((y_pred[g1] == 1) & (y_real[g1] == 1)) / (np.sum(y_pred[g1] == 1) + 1e-5)
    diff_ppv = abs(ppv_g0 - ppv_g1)
    
    accuracy = np.mean(y_real == y_pred)
    
    return {
        "Accuracy": accuracy, "SPD": spd, "DI": di,
        "Diff_TPR": diff_tpr, "Diff_FPR": diff_fpr, "Diff_PPV": diff_ppv
    }

def testar_modelo(modelo):
    """Passa os dados de teste pelo modelo para recolher as previsões"""
    modelo.eval()
    previsoes = []
    reais = []
    atributos = []
    
    with torch.no_grad():
        for img, y, a in loader_teste:
            img = img.to(device)
            pred = modelo(img).squeeze()
            if pred.dim() == 0: pred = pred.unsqueeze(0)
                
            pred_binaria = (pred.cpu().numpy() > 0.5).astype(int)
            previsoes.extend(pred_binaria)
            reais.extend(y.numpy())
            atributos.extend(a.numpy())
            
    return metricas_fairness(reais, previsoes, atributos)

# =====================================================================
# 4. A FUNÇÃO DE TREINO COM OS MÉTODOS DE MITIGAÇÃO
# =====================================================================
def treinar_modelo_in_processing(metodo):
    print(f"\nA Iniciar Treino: [{metodo.upper()}]")
    
    # Criar um modelo novo para não misturar pesos
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modelo.fc = nn.Sequential(nn.Linear(modelo.fc.in_features, 1), nn.Sigmoid())
    modelo = modelo.to(device)
    
    criterio_erro = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    modelo.train()
    for epoca in range(EPOCHS):
        erro_acumulado = 0.0
        
        for batch_idx, (img, y_real, A_real) in enumerate(loader_treino):
            img, y_real, A_real = img.to(device), y_real.to(device), A_real.to(device)
            
            otimizador.zero_grad()
            y_pred = modelo(img).squeeze()
            if y_pred.dim() == 0: y_pred = y_pred.unsqueeze(0)
            
            # 1. Erro Normal de Classificação
            erro_normal = criterio_erro(y_pred, y_real)
            erro_total = erro_normal
            
            # -----------------------------------------------------
            # IN-PROCESSING: CÁLCULO DE FAIRNESS NO LOTE
            # -----------------------------------------------------
            g0_mask = (A_real == 0)
            g1_mask = (A_real == 1)
            
            # Só podemos calcular viés se o lote tiver pessoas de ambos os géneros
            if g0_mask.sum() > 0 and g1_mask.sum() > 0 and metodo != "baseline":
                pred_bin = (y_pred > 0.5).float()
                
                # Calcular diferença de TPR para penalizar
                tpr0 = (pred_bin[g0_mask] * y_real[g0_mask]).sum() / (y_real[g0_mask].sum() + 1e-5)
                tpr1 = (pred_bin[g1_mask] * y_real[g1_mask]).sum() / (y_real[g1_mask].sum() + 1e-5)
                diferenca_tpr = torch.abs(tpr0 - tpr1)
                
                # --- MÉTODO 1: REGULARIZAÇÃO (Soft Constraint) ---
                if metodo == "regularizacao":
                    erro_fairness = diferenca_tpr
                    erro_total = erro_normal + (LAMBDA_REGULARIZACAO * erro_fairness)
                
                # --- MÉTODO 2: IMPOSIÇÃO DE RESTRIÇÕES (Hard Constraint Simulado) ---
                elif metodo == "restricao":
                    # Simulação: Se violar a restrição máxima permitida (epsilon)
                    if diferenca_tpr > EPSILON_RESTRICAO:
                        # Adiciona uma "barreira infinita" (um valor gigante)
                        # O modelo percebe que esta configuração de pesos é proibida
                        erro_total = erro_normal + 1000.0 
                        
            # Retropropagação
            erro_total.backward()
            otimizador.step()
            erro_acumulado += erro_total.item()
            
        print(f"Época {epoca+1}/{EPOCHS} concluída. Erro Final: {erro_acumulado/(batch_idx+1):.4f}")
        
    return modelo

# =====================================================================
# 5. EXECUÇÃO COMPARATIVA
# =====================================================================
# A) Modelo Original sem mitigação
modelo_base = treinar_modelo_in_processing(metodo="baseline")
res_base = testar_modelo(modelo_base)

# B) Modelo com Regularização
modelo_reg = treinar_modelo_in_processing(metodo="regularizacao")
res_reg = testar_modelo(modelo_reg)

# C) Modelo com Imposição de Restrições
modelo_res = treinar_modelo_in_processing(metodo="restricao")
res_res = testar_modelo(modelo_res)

# =====================================================================
# 6. RELATÓRIO E GRÁFICO (Muito Simples e Limpo)
# =====================================================================
print("\n" + "="*60)
print("             RELATÓRIO DE AUDITORIA             ")
print("="*60)

for nome, resultados in [("1. BASELINE", res_base), ("2. REGULARIZAÇÃO", res_reg), ("3. RESTRIÇÕES", res_res)]:
    print(f"\n{nome}:")
    print(f"  -> Acurácia Global: {resultados['Accuracy']:.2f}")
    print(f"  -> (Independência) DI: {resultados['DI']:.2f} | SPD: {resultados['SPD']:.2f}")
    print(f"  -> (Separação) Diff TPR: {resultados['Diff_TPR']:.2f}")
    print(f"  -> (Suficiência) Diff PPV: {resultados['Diff_PPV']:.2f}")

# Gráfico simples comparando a Acurácia e a Separação (TPR Diff)
nomes = ['Baseline', 'Regularização', 'Restrições']
accs = [res_base['Accuracy'], res_reg['Accuracy'], res_res['Accuracy']]
viés = [res_base['Diff_TPR'], res_reg['Diff_TPR'], res_res['Diff_TPR']]

fig, ax1 = plt.subplots(figsize=(8, 5))

x = np.arange(len(nomes))
ax1.bar(x - 0.2, accs, 0.4, label='Precisão Global (Mais Alto = Melhor)', color='lightgrey', edgecolor='black')
ax1.bar(x + 0.2, viés, 0.4, label='Viés (Δ TPR) (Mais Baixo = Mais Justo)', color='tomato', edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(nomes)
ax1.set_title('Comparação dos Métodos In-Processing (Trade-off)')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
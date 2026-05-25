import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. CARREGAMENTO DOS DADOS LOCAIS (Ficheiros CSV do Kaggle)
# =====================================================================
print("A carregar os metadados do CelebA localmente...")

try:
    # Ler os ficheiros CSV (devem estar na mesma pasta do script)
    df_atributos = pd.read_csv("list_attr_celeba.csv")
    df_particoes = pd.read_csv("list_eval_partition.csv")
except FileNotFoundError as e:
    print(f"\n[ERRO] Ficheiro não encontrado: {e.filename}")
    exit()

# O dataset usa -1 para Falso e 1 para Verdadeiro. Vamos converter os -1 para 0.
df_atributos = df_atributos.replace(-1, 0)

# Juntar os atributos com a informação da partição (0: Treino, 1: Validação, 2: Teste)
df_completo = pd.merge(df_atributos, df_particoes, on='image_id')

# Filtrar apenas as instâncias do conjunto de TESTE (partition == 2) para a nossa auditoria
df_teste = df_completo[df_completo['partition'] == 2].copy()

print(f"Total de instâncias no dataset: {len(df_completo)}")
print(f"Instâncias de teste (para auditoria): {len(df_teste)}")

# =====================================================================
# 2. MENU INTERATIVO (As 10 Opções Sugeridas)
# =====================================================================

opcoes_sugeridas = {
    1: ("Male", "Género (Atributo Sensível)"),
    2: ("Young", "Idade (Atributo Sensível)"),
    3: ("Pale_Skin", "Tom de Pele (Atributo Sensível / Proxy)"),
    4: ("Smiling", "A Sorrir (Variável Alvo / Expressão)"),
    5: ("Eyeglasses", "Óculos (Acessório / Oclusão)"),
    6: ("Attractive", "Atraente (Subjetividade / Viés Humano)"),
    7: ("Heavy_Makeup", "Maquilhagem (Correlação Espúria)"),
    8: ("No_Beard", "Sem Barba (Correlação Espúria)"),
    9: ("Chubby", "Morfologia Facial (Peso)"),
    10: ("Gray_Hair", "Cabelo Grisalho (Idade)")
}

print("\n" + "="*40)
print("     SISTEMA INTERATIVO DE FAIRNESS     ")
print("="*40)
print("Selecione os atributos para a análise de viés:")
for num, (coluna, descricao) in opcoes_sugeridas.items():
    print(f"[{num}] {coluna:<15} -> {descricao}")

try:
    escolha_A = int(input("\nEscolha o número do Atributo Sensível (A): "))
    escolha_Y = int(input("Escolha o número da Variável Alvo a prever (Y): "))
    
    if escolha_A not in opcoes_sugeridas or escolha_Y not in opcoes_sugeridas:
        raise ValueError
except ValueError:
    print("\n[AVISO] Escolha inválida. Predefinições: A = Male, Y = Smiling.")
    escolha_A, escolha_Y = 1, 4

# Obter o nome real das colunas que constam no DataFrame
coluna_A_nome = opcoes_sugeridas[escolha_A][0]
coluna_Y_nome = opcoes_sugeridas[escolha_Y][0]

print(f"\n[CONFIGURAÇÃO] Atributo Sensível (A): {coluna_A_nome}")
print(f"[CONFIGURAÇÃO] Variável Alvo (Y):      {coluna_Y_nome}")

# =====================================================================
# 3. PROCESSAMENTO E EXTRAÇÃO DOS RÓTULOS (Pandas para NumPy)
# =====================================================================
print("\nA extrair rótulos do conjunto de teste para análise...")

# Isolar os vetores de interesse (A e Y reais) extraindo diretamente as colunas do Pandas
A_real = df_teste[coluna_A_nome].values
y_real = df_teste[coluna_Y_nome].values

print(f"Distribuição do Atributo Sensível ({coluna_A_nome}):")
print(f"  -> Grupo 0 (Falso/Não): {np.sum(A_real == 0)} indivíduos")
print(f"  -> Grupo 1 (Verdade/Sim): {np.sum(A_real == 1)} indivíduos")

# =====================================================================
# 4. SIMULAÇÃO DAS PREVISÕES DO MODELO
# =====================================================================
# Nota: Para o script correr de imediato sem treinar a rede convolucional,
# vamos simular o resultado de um classificador imperfeito com viés em NumPy.
# Substitua 'y_pred' pelas previsões reais do seu modelo PyTorch após o treino.
np.random.seed(42)
proporcao_erro = 0.15
ruido = np.random.choice([0, 1], size=len(y_real), p=[1 - proporcao_erro, proporcao_erro])
y_pred = np.abs(y_real - ruido) # Inverte o valor real em 15% dos casos

# Forçar artificialmente uma taxa de aceitação maior no Grupo 0 para criar um viés visível
indices_grupo_0 = np.where(A_real == 0)[0]
substituir_indices = np.random.choice(indices_grupo_0, size=int(len(indices_grupo_0) * 0.1), replace=False)
y_pred[substituir_indices] = 1

# =====================================================================
# 5. IMPLEMENTAÇÃO DAS MÉTRICAS DE FAIRNESS (Sem bibliotecas externas)
# =====================================================================
def calcular_independencia(y_pred, A):
    """
    1. INDEPENDÊNCIA: Calcula a Paridade Demográfica (SPD) e o Disparate Impact (DI).
    Fórmula SPD: P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)
    Fórmula DI:  P(Y_hat=1 | A=0) / P(Y_hat=1 | A=1)
    """
    taxa_grupo0 = np.mean(y_pred[A == 0] == 1)
    taxa_grupo1 = np.mean(y_pred[A == 1] == 1)
    
    spd = taxa_grupo0 - taxa_grupo1
    
    # Prevenir divisões por zero
    di = taxa_grupo0 / taxa_grupo1 if taxa_grupo1 > 0 else np.nan
    
    # Regra dos 80% (0.8 <= DI <= 1.25)
    regra_80 = "Cumpre" if (0.8 <= di <= 1.25) else "Não Cumpre"
    
    return taxa_grupo0, taxa_grupo1, spd, di, regra_80


def calcular_separacao(y_true, y_pred, A):
    """
    2. SEPARAÇÃO: Calcula Igualdade de Oportunidades (Equalized Odds).
    Avalia diferenças nas Taxas de Verdadeiros Positivos (TPR) e Falsos Positivos (FPR).
    """
    tpr_grupos = {}
    fpr_grupos = {}
    
    for grupo in [0, 1]:
        mask = (A == grupo)
        y_t_g = y_true[mask]
        y_p_g = y_pred[mask]
        
        # TPR = VP / (VP + FN) (Protegido contra divisão por zero)
        positivos_reais = np.sum(y_t_g == 1)
        tpr = np.sum((y_p_g == 1) & (y_t_g == 1)) / positivos_reais if positivos_reais > 0 else 0
        
        # FPR = FP / (FP + VN)
        negativos_reais = np.sum(y_t_g == 0)
        fpr = np.sum((y_p_g == 1) & (y_t_g == 0)) / negativos_reais if negativos_reais > 0 else 0
        
        tpr_grupos[grupo] = tpr
        fpr_grupos[grupo] = fpr
        
    diff_tpr = abs(tpr_grupos[0] - tpr_grupos[1])
    diff_fpr = abs(fpr_grupos[0] - fpr_grupos[1])
    
    return tpr_grupos, fpr_grupos, diff_tpr, diff_fpr

def calcular_separacao(y_true, y_pred, A):
    """
    2. SEPARAÇÃO: Calcula Igualdade de Oportunidades (Equalized Odds).
    Avalia diferenças nas Taxas de Verdadeiros Positivos (TPR) e Falsos Positivos (FPR).
    """
    tpr_grupos = {}
    fpr_grupos = {}
    
    for grupo in [0, 1]:
        mask = (A == grupo)
        y_t_g = y_true[mask]
        y_p_g = y_pred[mask]
        
        # TPR = VP / (VP + FN) (Protegido contra divisão por zero)
        positivos_reais = np.sum(y_t_g == 1)
        tpr = np.sum((y_p_g == 1) & (y_t_g == 1)) / positivos_reais if positivos_reais > 0 else 0
        
        # FPR = FP / (FP + VN)
        negativos_reais = np.sum(y_t_g == 0)
        fpr = np.sum((y_p_g == 1) & (y_t_g == 0)) / negativos_reais if negativos_reais > 0 else 0
        
        tpr_grupos[grupo] = tpr
        fpr_grupos[grupo] = fpr
        
    diff_tpr = abs(tpr_grupos[0] - tpr_grupos[1])
    diff_fpr = abs(fpr_grupos[0] - fpr_grupos[1])
    
    return tpr_grupos, fpr_grupos, diff_tpr, diff_fpr


def calcular_suficiencia(y_true, y_pred, A):
    """
    3. SUFICIÊNCIA: Calcula a Paridade Preditiva (Calibração).
    Avalia diferenças nos Valores Preditivos Positivos (PPV) e Negativos (NPV).
    """
    ppv_grupos = {}
    npv_grupos = {}
    
    for grupo in [0, 1]:
        mask = (A == grupo)
        y_t_g = y_true[mask]
        y_p_g = y_pred[mask]
        
        # PPV (Precisão) = VP / (VP + FP) -> De todos os previstos como 1, quantos eram 1?
        previstos_positivos = np.sum(y_p_g == 1)
        ppv = np.sum((y_p_g == 1) & (y_t_g == 1)) / previstos_positivos if previstos_positivos > 0 else 0
        
        # NPV = VN / (VN + FN) -> De todos os previstos como 0, quantos eram 0?
        previstos_negativos = np.sum(y_p_g == 0)
        npv = np.sum((y_p_g == 0) & (y_t_g == 0)) / previstos_negativos if previstos_negativos > 0 else 0
        
        ppv_grupos[grupo] = ppv
        npv_grupos[grupo] = npv
        
    diff_ppv = abs(ppv_grupos[0] - ppv_grupos[1])
    diff_npv = abs(npv_grupos[0] - npv_grupos[1])
    
    return ppv_grupos, npv_grupos, diff_ppv, diff_npv


# Calcular os valores com as funções rigorosas
taxa_g0, taxa_g1, spd, di, regra_80 = calcular_independencia(y_pred, A_real)
tpr_g, fpr_g, diff_tpr, diff_fpr = calcular_separacao(y_real, y_pred, A_real)
ppv_g, npv_g, diff_ppv, diff_npv = calcular_suficiencia(y_real, y_pred, A_real)

# =====================================================================
# 6. EXIBIÇÃO DE RESULTADOS DA AUDITORIA
# =====================================================================
print("\n" + "="*50)
print("       RESULTADOS DA AUDITORIA TEÓRICA        ")
print("="*50)

print(f"\n[Pilar 1] INDEPENDÊNCIA (Paridade Demográfica)")
print(f"  -> Taxa de Seleção Positiva Grupo 0: {taxa_g0:.4f}")
print(f"  -> Taxa de Seleção Positiva Grupo 1: {taxa_g1:.4f}")
print(f"  -> Diferença (SPD):               {spd:.4f}  (Ideal = 0.0)")
print(f"  -> Impacto Disparate (DI):        {di:.4f}  (Ideal = 1.0)")
print(f"  -> Regra dos 80% (0.8 a 1.25):    [{regra_80}]")

print(f"\n[Pilar 2] SEPARAÇÃO (Igualdade de Oportunidades)")
print(f"  -> Grupo 0 | TPR: {tpr_g[0]:.4f} | FPR: {fpr_g[0]:.4f}")
print(f"  -> Grupo 1 | TPR: {tpr_g[1]:.4f} | FPR: {fpr_g[1]:.4f}")
print(f"  -> Diferença na taxa TPR:      {diff_tpr:.4f}  (Ideal = 0.0)")
print(f"  -> Diferença na taxa FPR:      {diff_fpr:.4f}  (Ideal = 0.0)")

print(f"\n[Pilar 3] SUFICIÊNCIA (Paridade Preditiva)")
print(f"  -> Grupo 0 | PPV (Precisão): {ppv_g[0]:.4f} | NPV: {npv_g[0]:.4f}")
print(f"  -> Grupo 1 | PPV (Precisão): {ppv_g[1]:.4f} | NPV: {npv_g[1]:.4f}")
print(f"  -> Diferença no PPV:           {diff_ppv:.4f}  (Ideal = 0.0)")
print(f"  -> Diferença no NPV:           {diff_npv:.4f}  (Ideal = 0.0)")

# =====================================================================
# 7. VISUALIZAÇÃO GRÁFICA DOS RESULTADOS (Matplotlib)
# =====================================================================

grupos_labels = ['Grupo 0 (Não)', 'Grupo 1 (Sim)']
x = np.arange(len(grupos_labels))
width = 0.35

# Expandir para 3 gráficos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Gráfico 1: Independência (Paridade Demográfica)
ax1.bar(grupos_labels, [taxa_g0, taxa_g1], color=['#3498db', '#e74c3c'], edgecolor='black', width=0.4)
ax1.set_ylabel(r'Proporção de Previsões Positivas ($\hat{Y}=1$)')
ax1.set_title(f'1. Independência\nAtributo: {coluna_A_nome}\nSPD = {spd:.3f} | DI = {di:.3f}\nRegra 80%: {regra_80}')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Gráfico 2: Separação (Equalized Odds)
ax2.bar(x - width/2, [tpr_g[0], tpr_g[1]], width, label='TPR (Verdadeiros Positivos)', color='#2ecc71', edgecolor='black')
ax2.bar(x + width/2, [fpr_g[0], fpr_g[1]], width, label='FPR (Falsos Positivos)', color='#f1c40f', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(grupos_labels)
ax2.set_ylabel('Taxa / Proporção')
ax2.set_title(f'2. Separação (Equalized Odds)\nAlvo: {coluna_Y_nome}\nΔ TPR = {diff_tpr:.3f} | Δ FPR = {diff_fpr:.3f}')
ax2.set_ylim(0, 1.0)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Gráfico 3: Suficiência (Paridade Preditiva / Calibração)
ax3.bar(x - width/2, [ppv_g[0], ppv_g[1]], width, label='PPV (Precisão)', color='#9b59b6', edgecolor='black')
ax3.bar(x + width/2, [npv_g[0], npv_g[1]], width, label='NPV', color='#95a5a6', edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(grupos_labels)
ax3.set_ylabel('Valor Preditivo')
ax3.set_title(f'3. Suficiência (Calibração)\nAlvo: {coluna_Y_nome}\nΔ PPV = {diff_ppv:.3f} | Δ NPV = {diff_npv:.3f}')
ax3.set_ylim(0, 1.0)
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
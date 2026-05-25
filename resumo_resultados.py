import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Usar um estilo limpo e académico
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================================================
# 1. DADOS CENTRALIZADOS (EXTRAÍDOS DOS 3 TESTES)
# =====================================================================
modelos = ['Baseline (Sem Mitigação)', 'Regularização (Soft)', 'Restrições (Hard)']
testes_labels = ['Teste 1', 'Teste 2', 'Teste 3']
cores = ['#e74c3c', '#3498db', '#2ecc71'] # Vermelho, Azul, Verde

# --- Dados Organizados por Teste (Para o Gráfico de Dispersão / Trade-off) ---
# Valores: [Baseline, Regularização, Restrições]
acc_t1 = [0.85, 0.89, 0.89]
tpr_t1 = [0.15, 0.01, 0.01]

acc_t2 = [0.89, 0.87, 0.90]
tpr_t2 = [0.05, 0.00, 0.07]

acc_t3 = [0.88, 0.89, 0.86]
tpr_t3 = [0.06, 0.06, 0.09]

dados_tradeoff = [
    (acc_t1, tpr_t1, 'o', 'Teste 1'),
    (acc_t2, tpr_t2, 's', 'Teste 2'),
    (acc_t3, tpr_t3, '^', 'Teste 3')
]

# --- Dados Organizados por Métrica (Para o Gráfico de Barras dos 3 Pilares) ---
# Valores: [Teste 1, Teste 2, Teste 3]
spd_base = [0.14, 0.10, 0.12]
spd_reg  = [0.09, 0.05, 0.13]
spd_res  = [0.09, 0.15, 0.13]

tpr_base = [0.15, 0.05, 0.06]
tpr_reg  = [0.01, 0.00, 0.06]
tpr_res  = [0.01, 0.07, 0.09]

ppv_base = [0.07, 0.07, 0.05]
ppv_reg  = [0.07, 0.13, 0.05]
ppv_res  = [0.06, 0.01, 0.01]

dados_pilares = [
    (spd_base, spd_reg, spd_res, '1. Independência (SPD)', 'Ideal = 0.0'),
    (tpr_base, tpr_reg, tpr_res, '2. Separação ($\Delta$ TPR)', 'Ideal = 0.0'),
    (ppv_base, ppv_reg, ppv_res, '3. Suficiência ($\Delta$ PPV)', 'Ideal = 0.0')
]

# =====================================================================
# GRÁFICO 1: Evolução Multi-Teste (Trade-off Precisão vs Separação)
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))
tamanhos_sobreposicao = [250, 350, 150] # Para não esconder pontos colados (Teste 1)

# Desenhar os pontos e as linhas de cada teste
for acc_teste, tpr_teste, marcador, nome_teste in dados_tradeoff:
    # Linha pontilhada a unir os modelos
    ax1.plot(tpr_teste, acc_teste, linestyle='--', color='grey', alpha=0.5, zorder=4)
    
    # Desenhar os 3 pontos
    for i in range(3):
        tamanho = tamanhos_sobreposicao[i] if nome_teste == 'Teste 1' else 200
        ax1.scatter(tpr_teste[i], acc_teste[i], color=cores[i], marker=marcador, 
                    s=tamanho, edgecolor='black', zorder=5, alpha=0.85)

# Configurar os Eixos e Títulos
ax1.set_title('Evolução: Precisão Global vs. Separação (Trade-off Multi-Teste)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Viés do Modelo ($\Delta$ TPR) $\\rightarrow$ Mais perto de 0 é Melhor', fontsize=12)
ax1.set_ylabel('Precisão Global (Accuracy) $\\rightarrow$ Mais alto é Melhor', fontsize=12)
ax1.margins(0.15)
ax1.invert_xaxis()

# Criar a Legenda Customizada (Modelos e Testes)
elementos_legenda = [
    Line2D([0], [0], marker='o', color='w', label='Baseline', markerfacecolor=cores[0], markersize=12, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label='Regularização', markerfacecolor=cores[1], markersize=12, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label='Restrições', markerfacecolor=cores[2], markersize=12, markeredgecolor='black'),
    Line2D([0], [0], marker='', color='w', label='---'),
    Line2D([0], [0], marker='o', color='w', label='Teste 1', markerfacecolor='grey', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', label='Teste 2', markerfacecolor='grey', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='^', color='w', label='Teste 3', markerfacecolor='grey', markersize=10, markeredgecolor='black')
]
ax1.legend(handles=elementos_legenda, loc='lower left', frameon=True, shadow=True, title="Legenda", title_fontsize='11')

plt.tight_layout()
fig1.savefig('grafico_01_tradeoff.png', dpi=300)
print("Gráfico 1 (Trade-off) guardado como 'grafico_01_tradeoff.png'")

# =====================================================================
# GRÁFICO 2: Painel Multi-Teste (Auditoria aos 3 Pilares)
# =====================================================================
fig2, axs2 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig2.suptitle('Auditoria aos Três Pilares da Imparcialidade', fontsize=16, fontweight='bold', y=1.05)

x = np.arange(len(testes_labels))
largura = 0.25 

# Função auxiliar para colocar números em cima das barras
def adicionar_rotulos(ax, barras):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(f'{altura:.2f}', xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Iterar sobre as métricas para construir os 3 subplots
for i, (base, reg, res, titulo, ideal) in enumerate(dados_pilares):
    ax = axs2[i]
    
    b1 = ax.bar(x - largura, base, largura, label=modelos[0], color=cores[0], edgecolor='black', zorder=3)
    b2 = ax.bar(x,           reg,  largura, label=modelos[1], color=cores[1], edgecolor='black', zorder=3)
    b3 = ax.bar(x + largura, res,  largura, label=modelos[2], color=cores[2], edgecolor='black', zorder=3)
    
    ax.set_title(f'{titulo}\n{ideal}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(testes_labels, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    adicionar_rotulos(ax, b1)
    adicionar_rotulos(ax, b2)
    adicionar_rotulos(ax, b3)

axs2[0].set_ylabel('Valor da Diferença (Viés)', fontsize=12)
axs2[0].legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

plt.tight_layout()
fig2.savefig('grafico_02_pilares.png', dpi=300, bbox_inches='tight')
print("Gráfico 2 (Três Pilares) guardado como 'grafico_02_pilares.png'")

# Mostrar ambos os gráficos no ecrã
plt.show()
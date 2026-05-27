# Projeto de Mitigação de Viés (Fairness) em Reconhecimento Facial

Este repositório contém o código, a metodologia e o relatório final do projeto de IA Responsável.

## Visão Geral
O objetivo deste trabalho é avaliar a eficácia de métodos de mitigação *in-processing* na redução do viés algorítmico, mantendo o desempenho preditivo (*Accuracy*) do modelo. Investigamos  os pilares de imparcialidade **Independência**, **Separação** e **Suficiência**.

## Tecnologias Utilizadas
- **Linguagem:** Python 3.14.3
- **Frameworks:** PyTorch (modelo ResNet-18), NumPy, Matplotlib (visualização de métricas).
- **Dataset:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Métodos de Mitigação:** - Regularização (*Soft Constraints*)
    - Imposição de Restrições (*Hard Constraints* via função de barreira)

## Configuração do ambiente
1. Deve clonar o repositório:

```bash
git clone <url-do-teu-repositorio>
cd <nome-da-pasta>
```

2. Criar um ambiente virtual:

   `python -m venv venv`
3. Ativar o ambiente:

   `source venv/bin/activate` (ou `venv\Scripts\activate` no Windows)
4. Instalar as dependências:

   `pip install -r requirements.txt`

## Dataset e Modelo
Para correr o `project_final.py`, é preciso de colocar o dataset `img_align_celeba` na raiz do projeto. 
*Nota: O dataset e os modelos pré-treinados não estão incluídos no repositório devido ao seu tamanho.*

> O download é fácil através do kaggle $\rightarrow$ <a>https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download</a>

## Estrutura do Projeto

```text
├── project_final.py       # Script principal de treino e mitigação
├── testes/                # Resultados (.txt) das execuções
├── .gitignore             # Definição de ficheiros a ignorar
└── requirements.txt       # Dependências do projeto
```

## Auditoria e Resultados
O projeto utiliza um *pipeline* de auditoria que mapeia a equidade em três pilares fundamentais. Os nossos resultados empíricos validam que a Regularização suave é superior à Imposição de Restrições rígidas, devido à instabilidade numérica que estas últimas introduzem no gradiente descendente.

## Colaboradores
- **João Pinto** M15429
- **Leonor Moreira** M15415
# Projeto de Mitigação de Viés (Fairness) em Reconhecimento Facial

Este repositório contém o código, a metodologia e o relatório final do projeto de IA Responsável.

## Como preparar o ambiente
1. Criar um ambiente virtual:
   `python -m venv venv`
2. Ativar o ambiente:
   `source venv/bin/activate` (ou `venv\Scripts\activate` no Windows)
3. Instalar as dependências.

## Dataset e Modelo
Para correr o `project_final.py`, precisas de colocar o dataset `img_align_celeba` na raiz do projeto. 
*Nota: O dataset e os modelos pré-treinados não estão incluídos no repositório devido ao seu tamanho.*

> O download é fácil através do kaggle!

## Estrutura
- `/testes`: Contém os outputs (.txt) das execuções.
- `project_final.py`: Código principal com a implementação ResNet-18.
- `projeto.py`: Tentativa inicial para testar as métricas de fairness.
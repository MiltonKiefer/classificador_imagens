# Classificador de imagens CIFAR-10

Projeto de classificação de imagens com uma rede neural convolucional em
TensorFlow/Keras. O pipeline carrega o CIFAR-10, normaliza as imagens, treina o
modelo, mede o desempenho no conjunto de teste e mostra gráficos de treinamento
e exemplos de previsões.

## Requisito de ambiente

O projeto usa Python 3.12 e TensorFlow. Para essa combinação, execute o projeto
em Linux, por exemplo no WSL2 com Ubuntu. O TensorFlow oficial não oferece
suporte às versões atuais em Windows nativo. O treinamento pode ser feito em
CPU; não é obrigatório ter GPU.

## Instalação e execução no WSL2/Ubuntu

Abra o terminal do Ubuntu no WSL2 e execute:

```bash
cd "/mnt/c/Users/milto/OneDrive/Documentos/SENAI CIMATEC/Machine Learning/curso-ml-cimatec/classificador_imagem"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m classificador_imagem.run
```

Na primeira execução, o Keras baixa o CIFAR-10 automaticamente. O programa
mostra exemplos do conjunto de dados, o resumo do modelo, curvas de acurácia e
perda, métricas no conjunto de teste e previsões para nove imagens. O
treinamento usa até 50 épocas, lotes de 64 imagens e interrupção/ajuste de taxa
de aprendizagem por callbacks.

Para executar os testes automatizados:

```bash
python -m pytest tests -q
```

Se o projeto estiver na unidade C:, os dados continuam acessíveis pelo WSL no
caminho `/mnt/c/...`. Para melhor desempenho de leitura/escrita, especialmente
em treinamento longo, pode-se trabalhar com uma cópia do projeto no sistema de
arquivos Linux do WSL.

## Estrutura do código

- `classificador_imagem/config.py`: nomes das classes, dimensões e parâmetros.
- `classificador_imagem/dataset.py`: carregamento do CIFAR-10.
- `classificador_imagem/features.py`: normalização das imagens e ajuste dos
  rótulos.
- `classificador_imagem/modeling/model.py`: construção e compilação da CNN.
- `classificador_imagem/modeling/train.py`: treinamento e callbacks.
- `classificador_imagem/modeling/evaluate.py`: avaliação no conjunto de teste.
- `classificador_imagem/modeling/predict.py`: probabilidades e classes previstas.
- `classificador_imagem/plots.py`: gráficos e painéis de imagens.
- `classificador_imagem/run.py`: execução integrada do pipeline.

O notebook original em `../deep_learning/notebooks/Classificador_imagens.ipynb`
é mantido separado e não é alterado por este projeto.

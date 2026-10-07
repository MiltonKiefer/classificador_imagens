# Primeiros passos

Use Ubuntu no WSL2 para executar o projeto com Python 3.12 e TensorFlow. No
terminal do Ubuntu:

```bash
cd "/mnt/c/Users/milto/OneDrive/Documentos/SENAI CIMATEC/Machine Learning/curso-ml-cimatec/classificador_imagem"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m classificador_imagem.run
```

Na primeira execução, o Keras baixa o CIFAR-10. O pipeline treina o modelo,
exibe as métricas de teste e abre os gráficos. Para rodar os testes sem treinar:

```bash
python -m pytest tests -q
```

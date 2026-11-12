
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

from leitor import ler_ocupacao, desenhar

# CONFIGURAÇÕES
ANDARES = 6
LIMITE_ALERTA = 50
CONFIANCA = 0.25
CANTOS = None

# CAMINHOS
BASE = Path(__file__).resolve().parent
ARQ_MODELO = BASE / "models" / "prateleira.pt"
PASTA_IMAGENS = BASE / "imagens"
PASTA_RESULTADOS = BASE / "resultados"

# VERIFICAÇÕES
if not ARQ_MODELO.exists():
    raise FileNotFoundError(
        f"Modelo não encontrado: {ARQ_MODELO}"
    )

if not PASTA_IMAGENS.exists():
    raise FileNotFoundError(
        f"Pasta de imagens não encontrada: {PASTA_IMAGENS}"
    )

PASTA_RESULTADOS.mkdir(exist_ok=True)

# CARREGAR MODELO
modelo = YOLO(str(ARQ_MODELO))
print("Modelo carregado com sucesso!")

# LOCALIZAR IMAGENS
extensoes = {".jpg", ".jpeg", ".png", ".bmp"}

fotos = [
    arquivo for arquivo in PASTA_IMAGENS.iterdir()
    if arquivo.suffix.lower() in extensoes
]

if not fotos:
    raise RuntimeError("Nenhuma imagem encontrada.")

# PROCESSAR IMAGENS
for arquivo in fotos:

    foto = cv2.imread(str(arquivo))

    if foto is None:
        print(f"Não foi possível abrir: {arquivo.name}")
        continue

    resultado = ler_ocupacao(
        foto,
        modelo,
        andares=ANDARES,
        cantos=CANTOS,
        conf=CONFIANCA,
        limite=LIMITE_ALERTA
    )

    imagem_final = desenhar(
        foto,
        resultado,
        LIMITE_ALERTA
    )

    # SALVAR RESULTADO
    destino = PASTA_RESULTADOS / f"{arquivo.stem}_resultado.jpg"
    cv2.imwrite(str(destino), imagem_final)

    # EXIBIR INFORMAÇÕES
    print(f"\nFoto: {arquivo.name}")
    print(f"Produtos encontrados: {len(resultado['caixas'])}")

    for i, ocupacao in enumerate(resultado["por_andar"]):
        status = (
            "ALERTA DE REABASTECIMENTO"
            if ocupacao < LIMITE_ALERTA
            else "OK"
        )
        print(f"Andar {i+1}: {ocupacao:.0f}% - {status}")

    print(f"Ocupação total: {resultado['total']:.0f}%")
    print(f"Alerta: {resultado['alerta']}")

    # EXIBIR IMAGEM
    plt.figure(figsize=(12, 9))
    plt.imshow(cv2.cvtColor(imagem_final, cv2.COLOR_BGR2RGB))
    plt.title(arquivo.name)
    plt.axis("off")
    plt.show()

print("\nProcessamento concluído!")

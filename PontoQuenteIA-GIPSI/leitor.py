
import cv2
import numpy as np


# ============================================================
# 1. LEITURA E CALCULO DA OCUPACAO
# ============================================================

def ler_ocupacao(
    foto,
    modelo,
    andares=1,
    cantos=None,
    conf=0.35,
    limite=50
):
    """
    Detecta produtos com YOLO e calcula a ocupacao horizontal
    de cada andar da prateleira.

    Parametros:
        foto: imagem BGR carregada pelo OpenCV.
        modelo: modelo YOLO previamente carregado.
        andares: quantidade de divisoes horizontais.
        cantos: quatro pontos da prateleira (opcional).
        conf: confianca minima das deteccoes.
        limite: porcentagem minima de ocupacao.

    Retorno:
        Dicionario contendo caixas, faixas, ocupacao e alertas.
    """

    # -------------------- VALIDACOES --------------------

    if foto is None or not isinstance(foto, np.ndarray):
        raise ValueError("Imagem invalida ou nao carregada.")

    if foto.ndim != 3 or foto.shape[2] != 3:
        raise ValueError("A imagem deve possuir tres canais BGR.")

    if not isinstance(andares, int) or isinstance(andares, bool) or andares < 1:
        raise ValueError("A quantidade de andares deve ser um inteiro positivo.")

    if not 0 < conf <= 1:
        raise ValueError("A confianca deve estar entre 0 e 1.")

    if not 0 <= limite <= 100:
        raise ValueError("O limite deve estar entre 0 e 100.")

    h, w = foto.shape[:2]

    # -------------------- AREA DA PRATELEIRA --------------------

    if cantos is None:
        cantos = [
            (0, 0),
            (w, 0),
            (w, h),
            (0, h)
        ]

    cantos = np.asarray(cantos, dtype=np.float32)

    if cantos.shape != (4, 2) or not np.isfinite(cantos).all():
        raise ValueError(
            "Informe exatamente quatro cantos validos da prateleira."
        )

    if not cv2.isContourConvex(cantos.reshape(-1, 1, 2)):
        raise ValueError(
            "Os quatro cantos devem formar um quadrilatero convexo."
        )

    # -------------------- DETECCAO YOLO --------------------

    deteccao = modelo.predict(
        foto,
        conf=conf,
        imgsz=960,
        max_det=300,
        verbose=False
    )[0]

    if deteccao.boxes is not None and len(deteccao.boxes) > 0:
        caixas = deteccao.boxes.xyxy.cpu().numpy()
    else:
        caixas = np.empty((0, 4), dtype=np.float32)

    # -------------------- CORRECAO DE PERSPECTIVA --------------------

    destino = np.float32([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    matriz = cv2.getPerspectiveTransform(cantos, destino)

    # Cada andar recebe intervalos horizontais ocupados.
    faixas = [[] for _ in range(andares)]

    for x0, y0, x1, y1 in caixas:

        # Quatro vertices da caixa de deteccao.
        pontos = np.float32([[
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1]
        ]])

        # Transformacao para coordenadas normalizadas.
        pontos = cv2.perspectiveTransform(
            pontos,
            matriz
        )[0]

        u0 = np.clip(pontos[:, 0].min(), 0, 1)
        u1 = np.clip(pontos[:, 0].max(), 0, 1)

        vc = (
            pontos[:, 1].min() +
            pontos[:, 1].max()
        ) / 2

        if u1 <= u0 or not 0 <= vc < 1:
            continue

        indice = min(
            int(vc * andares),
            andares - 1
        )

        faixas[indice].append((u0, u1))

    # -------------------- CALCULO DA OCUPACAO --------------------

    ocupacao = []

    for faixa in faixas:

        faixa.sort()

        total = 0
        inicio = fim = None

        # Uniao de intervalos:
        # evita contar duas vezes produtos sobrepostos.

        for a, b in faixa:

            if inicio is None or a > fim:

                if inicio is not None:
                    total += fim - inicio

                inicio, fim = a, b

            else:
                fim = max(fim, b)

        if inicio is not None:
            total += fim - inicio

        ocupacao.append(float(100 * total))

    # -------------------- RESULTADO FINAL --------------------

    media = float(np.mean(ocupacao))

    alerta_andares = [
        i + 1
        for i, valor in enumerate(ocupacao)
        if valor < limite
    ]

    return {
        "caixas": caixas,
        "faixas": faixas,
        "por_andar": ocupacao,
        "total": media,
        "alerta": media < limite,
        "alerta_andares": alerta_andares,
        "cantos": cantos
    }


# ============================================================
# 2. DESENHO DOS RESULTADOS NA IMAGEM
# ============================================================

def desenhar(foto, r, limite=50):
    """
    Desenha deteccoes, divisoes dos andares, porcentagens
    de ocupacao e alerta geral na imagem original.

    Parametros:
        foto: imagem original em BGR.
        r: resultado retornado por ler_ocupacao().
        limite: porcentagem minima de ocupacao.

    Retorno:
        Imagem processada em BGR.
    """

    # -------------------- VALIDACOES --------------------

    if foto is None or not isinstance(foto, np.ndarray):
        raise ValueError("Imagem invalida.")

    if not isinstance(r, dict):
        raise ValueError("Resultado de ocupacao invalido.")

    chaves = {
        "caixas",
        "por_andar",
        "total",
        "alerta",
        "cantos"
    }

    if not chaves.issubset(r.keys()):
        raise ValueError("O resultado nao possui todos os dados necessarios.")

    if not 0 <= limite <= 100:
        raise ValueError("O limite deve estar entre 0 e 100.")

    out = foto.copy()

    h, w = out.shape[:2]

    andares = len(r["por_andar"])

    if andares < 1:
        raise ValueError("Nenhum andar foi informado.")

    espessura = max(2, w // 300)

    # -------------------- CAIXAS DE DETECCAO --------------------

    for x0, y0, x1, y1 in np.asarray(r["caixas"]).reshape(-1, 4).astype(int):

        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            (60, 200, 60),
            espessura
        )

    # -------------------- PERSPECTIVA INVERSA --------------------

    cantos = np.asarray(r["cantos"], dtype=np.float32)

    origem = np.float32([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    matriz_inversa = cv2.getPerspectiveTransform(
        origem,
        cantos
    )

    # -------------------- DIVISAO DOS ANDARES --------------------

    for i in range(1, andares):

        pontos = np.float32([[
            [0, i / andares],
            [1, i / andares]
        ]])

        pontos = cv2.perspectiveTransform(
            pontos,
            matriz_inversa
        )[0].astype(int)

        cv2.line(
            out,
            tuple(pontos[0]),
            tuple(pontos[1]),
            (255, 200, 0),
            espessura
        )

    # Contorno da prateleira.
    cv2.polylines(
        out,
        [cantos.astype(int)],
        True,
        (255, 200, 0),
        espessura
    )

    # -------------------- ETIQUETAS DOS ANDARES --------------------

    escala = max(0.5, w / 1000)

    fonte_espessura = max(1, int(2 * escala))

    for i, ocupacao in enumerate(r["por_andar"]):

        ponto = np.float32([[
            [0.01, (i + 0.12) / andares]
        ]])

        ponto = cv2.perspectiveTransform(
            ponto,
            matriz_inversa
        )[0, 0].astype(int)

        cor = (
            (40, 40, 220)
            if ocupacao < limite
            else (60, 170, 60)
        )

        texto = f"Andar {i + 1}: {ocupacao:.0f}%"

        (tw, th), _ = cv2.getTextSize(
            texto,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            fonte_espessura
        )

        px, py = ponto

        # Mantem a etiqueta dentro da imagem.
        px = int(np.clip(px, 0, max(0, w - tw - 16)))
        py = int(np.clip(py, th + 10, max(th + 10, h - 5)))

        cv2.rectangle(
            out,
            (px, py - th - 10),
            (px + tw + 16, py + 10),
            cor,
            -1
        )

        cv2.putText(
            out,
            texto,
            (px + 8, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            (255, 255, 255),
            fonte_espessura
        )

    # -------------------- ALERTA GERAL --------------------

    faixa = max(35, int(70 * escala))

    cor_alerta = (
        (40, 40, 220)
        if r["alerta"]
        else (60, 150, 60)
    )

    out = cv2.copyMakeBorder(
        out,
        0,
        faixa,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=cor_alerta
    )

    mensagem = (
        f"Ocupacao {r['total']:.0f}% - "
        + (
            "ALERTA DE REABASTECIMENTO"
            if r["alerta"]
            else "Prateleira OK"
        )
    )

    cv2.putText(
        out,
        mensagem,
        (int(20 * escala), h + int(48 * escala)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1 * escala,
        (255, 255, 255),
        max(2, int(3 * escala))
    )

    return out

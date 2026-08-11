"""Núcleo do pipeline: dados COTAHIST -> features -> modelo -> ranking.

Assume arquivo já processado pelo b3_cotahist.py (colunas: ticker, data_pregao,
tipo_mercado, preco_abertura/maximo/minimo/fechamento, volume_financeiro,
numero_negocios).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DB_PADRAO = "decisoes.db"

HORIZONTE_PADRAO = 5  # pregões à frente que o modelo tenta prever
MERCADO_VISTA = 10    # tipo_mercado da B3 para ações à vista

SSA_JANELA_PADRAO = 60          # pregões olhados para trás para extrair a tendência (SSA)
SSA_JANELAS_DISPONIVEIS = (20, 60, 126)

FOURIER_JANELA_PADRAO = 60      # pregões da janela causal da decomposição de Fourier
FOURIER_JANELAS_DISPONIVEIS = (20, 60, 126)
FOURIER_HARMONICOS_PADRAO = 3   # nº de harmônicos de menor frequência mantidos no filtro
FOURIER_HARMONICOS_DISPONIVEIS = (1, 2, 3, 5, 8)

# Todos os parâmetros abaixo têm default 0 (custo/imposto desligado). Digite os
# valores reais na interface do app se quiser considerá-los na simulação.
# Referências de mercado (não são aplicadas automaticamente, é só consulta):
#   - Taxa B3 (negociação+CCP+TTA, não day trade, ADTV até R$3mi/mês): 0,0300%
#     Fonte: b3.com.br/pt_br/produtos-e-servicos/tarifas/.../a-vista (jul/2026)
#   - ISS sobre corretagem: 2% a 5% conforme município (LC 116/03); SP cobra 5%
#   - IRRF swing trade ("dedo-duro"): 0,005% sobre valor da venda, não retido
#     se o valor calculado for menor que R$1,00
#     Fonte: gov.br/receitafederal, seção Renda Variável > Isenções (jul/2026)
#   - IR swing trade: 15% sobre o ganho líquido mensal, isento se as vendas do
#     mês somarem até R$20.000 (Lei 11.033/2004, Art. 3º, II)
TAXA_B3_PADRAO = 0.0
CORRETAGEM_PADRAO = 0.0
CORRETAGEM_PERCENTUAL_PADRAO = 0.0
ISS_PADRAO = 0.0
IRRF_SWING_PCT = 0.0
IRRF_MINIMO = 1.0              # abaixo disso (em R$), a corretora não retém o IRRF
IR_SWING_ALIQUOTA = 0.0
ISENCAO_VENDAS_MENSAL = 0.0    # 0 = isenção nunca se aplica

FEATURE_COLS = [
    "retorno_1d", "retorno_5d", "retorno_20d",
    "volatilidade_20d", "media_5_sobre_20", "media_20_sobre_60",
    "volume_rel_20d", "amplitude_dia", "rsi_14", "momentum_10d",
    "ssa_tendencia", "fourier_tendencia", "fourier_energia_baixa",
]


def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_parquet(caminho) if caminho.endswith(".parquet") else pd.read_csv(
        caminho, parse_dates=["data_pregao"]
    )
    df = df[df["tipo_mercado"] == MERCADO_VISTA].copy()
    df = df.sort_values(["ticker", "data_pregao"])
    return df


def carregar_ibovespa(caminho: str) -> pd.DataFrame:
    """Lê CSV do Ibovespa (colunas: simbolo, data, abertura, maxima, minima,
    fechamento, fechamento_ajustado, volume, fonte — formato Yahoo Finance).

    Usa `fechamento_ajustado`. Descarta linhas com abertura/máxima/mínima
    zeradas (pregão do dia corrente ainda incompleto no momento do download,
    não um fechamento real).
    """
    ibov = pd.read_csv(caminho, parse_dates=["data"])
    incompleta = (ibov["abertura"] == 0) & (ibov["maxima"] == 0) & (ibov["minima"] == 0)
    if incompleta.any():
        ibov = ibov[~incompleta]
    return ibov[["data", "fechamento_ajustado"]].rename(
        columns={"fechamento_ajustado": "fechamento"}
    ).sort_values("data").reset_index(drop=True)


def carregar_selic(caminho: str) -> pd.DataFrame:
    """Lê o JSON da série SGS/BCB 11 (Selic diária: [{"data":"dd/mm/aaaa","valor":"x.xxxxxx"}]).

    `valor` já vem em percentual ao dia (ex.: "0.024620" = 0,02462%/dia);
    aqui é convertido para fração decimal (0.0002462).
    """
    selic = pd.read_json(caminho)
    selic["data"] = pd.to_datetime(selic["data"], format="%d/%m/%Y")
    selic["taxa_diaria"] = selic["valor"].astype(float) / 100
    return selic[["data", "taxa_diaria"]].sort_values("data").reset_index(drop=True)


def preparar_benchmark(resumo: pd.DataFrame, ibovespa: pd.DataFrame, selic: pd.DataFrame) -> pd.DataFrame:
    """Alinha Ibovespa e Selic às janelas [data_decisao, data_venda_prevista) de cada
    período do backtest: um retorno de benchmark e um retorno livre de risco por
    período, na mesma ordem/tamanho de `resumo` (uso direto em `calcular_metricas`).

    O período final, se não tiver `data_venda_prevista` (a simulação terminou antes
    do horizonte se completar), fica com NaN nas duas colunas.
    """
    linhas = resumo[["data_decisao", "data_venda_prevista"]].copy()
    linhas["data_decisao"] = pd.to_datetime(linhas["data_decisao"]).astype("datetime64[ns]")
    linhas["data_venda_prevista"] = pd.to_datetime(linhas["data_venda_prevista"]).astype("datetime64[ns]")

    ibov = ibovespa.sort_values("data").copy()
    ibov["data"] = pd.to_datetime(ibov["data"]).astype("datetime64[ns]")
    preco_decisao = pd.merge_asof(
        linhas[["data_decisao"]].rename(columns={"data_decisao": "data"}), ibov, on="data", direction="backward",
    )["fechamento"]
    preco_venda = pd.merge_asof(
        linhas[["data_venda_prevista"]].rename(columns={"data_venda_prevista": "data"}), ibov, on="data", direction="backward",
    )["fechamento"]
    retorno_benchmark = (preco_venda.to_numpy() / preco_decisao.to_numpy() - 1)

    taxas = selic.set_index("data")["taxa_diaria"]
    retorno_livre_risco = []
    for _, row in linhas.iterrows():
        if pd.isna(row["data_venda_prevista"]):
            retorno_livre_risco.append(np.nan)
            continue
        janela = taxas[(taxas.index >= row["data_decisao"]) & (taxas.index < row["data_venda_prevista"])]
        retorno_livre_risco.append(float(np.prod(1 + janela.to_numpy()) - 1) if len(janela) > 0 else np.nan)

    return pd.DataFrame({"retorno_benchmark": retorno_benchmark, "retorno_livre_risco": retorno_livre_risco})


def _ssa_ultimo_ponto(precos: np.ndarray) -> float:
    """Tendência (1º componente do SSA) no último pregão da janela recebida.

    Reconstrução via SVD da matriz trajetória (Hankel) usando só o
    autovetor de maior autovalor. O último ponto da janela corresponde a
    uma única célula da matriz reconstruída (não precisa de diagonal
    averaging), então o cálculo é direto: S[0] * U[-1,0] * Vt[0,-1].
    Não olha nada fora da janela recebida (uso causal via `.rolling`).
    """
    l = len(precos) // 2
    trajetoria = np.lib.stride_tricks.sliding_window_view(precos, l).T  # (l, k)
    U, S, Vt = np.linalg.svd(trajetoria, full_matrices=False)
    return S[0] * U[-1, 0] * Vt[0, -1]


def _fourier_ultimo_ponto(precos: np.ndarray, n_harmonicos: int = FOURIER_HARMONICOS_PADRAO) -> float:
    """Valor filtrado (passa-baixa de Fourier) no último pregão da janela recebida.

    Diferença conceitual para o SSA: o SSA extrai componentes cujas frequências
    emergem dos próprios dados (base adaptativa via SVD); Fourier projeta a série
    numa base FIXA de senos/cossenos de frequências pré-determinadas pela janela.
    São métodos espectrais distintos, e a discordância entre os dois é informativa.

    Detalhe de implementação importante: a FFT assume sinal periódico, então uma
    janela que começa em 20 e termina em 30 é lida como se houvesse um salto
    abrupto de 30 para 20 na "emenda" — isso gera artefato de borda (Gibbs)
    justamente no último ponto, que é o que interessa aqui. Por isso a tendência
    linear é removida antes da FFT e recomposta depois: sem esse passo, a feature
    seria dominada pelo artefato em séries com tendência forte.

    Trade-off assumido (verificado numericamente): esse detrend linear é a escolha
    certa para preços de ações, que têm tendência; mas em uma oscilação pura o
    ajuste de mínimos quadrados capta uma inclinação espúria e espalha energia
    entre harmônicos vizinhos (numa senoide exata de 2 ciclos, ~10% da energia
    vaza para o harmônico 1). Não existe filtro de FFT sem artefato em janela
    finita não periódica; a alternativa (sem detrend) troca esse vazamento por um
    artefato de borda bem pior no caso que de fato importa aqui.

    Não olha nada fora da janela recebida (uso causal via `.rolling`).
    """
    n = len(precos)
    t = np.arange(n)
    coef = np.polyfit(t, precos, 1)
    tendencia_linear = np.polyval(coef, t)
    residuo = precos - tendencia_linear

    espectro = np.fft.rfft(residuo)
    espectro[n_harmonicos + 1:] = 0  # mantém DC + os n harmônicos de menor frequência
    reconstruido = np.fft.irfft(espectro, n=n)
    return float(tendencia_linear[-1] + reconstruido[-1])


def _fourier_energia_baixa(precos: np.ndarray, n_harmonicos: int = FOURIER_HARMONICOS_PADRAO) -> float:
    """Fração da energia espectral concentrada nas baixas frequências da janela.

    Interpretação: perto de 1 = movimento dominado por poucos ciclos longos
    (comportamento tendencial/suave); perto de 0 = energia espalhada nas altas
    frequências (comportamento ruidoso, serrilhado). Mede o REGIME do ativo, não
    a direção — complementa `fourier_tendencia`, que mede o desvio de preço.

    Usa o mesmo detrend linear de `_fourier_ultimo_ponto` para que a tendência
    determinística não infle artificialmente a energia de baixa frequência.
    """
    n = len(precos)
    t = np.arange(n)
    coef = np.polyfit(t, precos, 1)
    residuo = precos - np.polyval(coef, t)

    espectro = np.abs(np.fft.rfft(residuo)) ** 2
    energia_total = espectro[1:].sum()  # ignora DC (é ~0 após detrend)
    if energia_total <= 0:
        return np.nan
    return float(espectro[1:n_harmonicos + 1].sum() / energia_total)


def _rsi(precos: pd.Series, periodo: int = 14) -> pd.Series:
    delta = precos.diff()
    ganho = delta.clip(lower=0).rolling(periodo).mean()
    perda = (-delta.clip(upper=0)).rolling(periodo).mean()
    rs = ganho / perda.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def construir_features(
    df: pd.DataFrame, horizonte: int = HORIZONTE_PADRAO, ssa_janela: int = SSA_JANELA_PADRAO,
    fourier_janela: int = FOURIER_JANELA_PADRAO, fourier_harmonicos: int = FOURIER_HARMONICOS_PADRAO,
) -> pd.DataFrame:
    g = df.groupby("ticker", group_keys=False)
    fechamento = g["preco_fechamento"]

    df["retorno_1d"] = fechamento.pct_change(1)
    df["retorno_5d"] = fechamento.pct_change(5)
    df["retorno_20d"] = fechamento.pct_change(20)
    df["volatilidade_20d"] = g["preco_fechamento"].transform(lambda s: s.pct_change().rolling(20).std())
    media_5 = g["preco_fechamento"].transform(lambda s: s.rolling(5).mean())
    media_20 = g["preco_fechamento"].transform(lambda s: s.rolling(20).mean())
    df["media_5_sobre_20"] = media_5 / media_20 - 1
    media_60 = g["preco_fechamento"].transform(lambda s: s.rolling(60).mean())
    df["media_20_sobre_60"] = media_20 / media_60 - 1
    vol_medio_20 = g["volume_financeiro"].transform(lambda s: s.rolling(20).mean())
    df["volume_rel_20d"] = df["volume_financeiro"] / vol_medio_20 - 1
    df["amplitude_dia"] = (df["preco_maximo"] - df["preco_minimo"]) / df["preco_fechamento"]
    df["rsi_14"] = g["preco_fechamento"].transform(_rsi)
    df["momentum_10d"] = fechamento.pct_change(10)
    # preço atual sobre a tendência SSA da janela: >0 acima da tendência, <0 abaixo
    ssa_tendencia = g["preco_fechamento"].transform(
        lambda s: s.rolling(ssa_janela).apply(_ssa_ultimo_ponto, raw=True)
    )
    df["ssa_tendencia"] = df["preco_fechamento"] / ssa_tendencia - 1

    # Fourier: preço atual sobre a tendência filtrada (passa-baixa), mesma leitura
    # do ssa_tendencia (>0 acima da tendência), mas com base espectral fixa.
    fourier_tendencia = g["preco_fechamento"].transform(
        lambda s: s.rolling(fourier_janela).apply(
            _fourier_ultimo_ponto, raw=True, args=(fourier_harmonicos,)
        )
    )
    df["fourier_tendencia"] = df["preco_fechamento"] / fourier_tendencia - 1
    df["fourier_energia_baixa"] = g["preco_fechamento"].transform(
        lambda s: s.rolling(fourier_janela).apply(
            _fourier_energia_baixa, raw=True, args=(fourier_harmonicos,)
        )
    )

    # alvo: retorno acumulado nos próximos `horizonte` pregões (shift negativo = futuro)
    df["retorno_futuro"] = g["preco_fechamento"].transform(
        lambda s: s.shift(-horizonte) / s - 1
    )
    return df


def _splits_walk_forward(datas_unicas: np.ndarray, n_splits: int = 5):
    """Gera cortes temporais crescentes (treino = passado, teste = bloco seguinte)."""
    blocos = np.array_split(datas_unicas, n_splits + 1)
    treino_ate = blocos[0]
    for bloco_teste in blocos[1:]:
        yield treino_ate, bloco_teste
        treino_ate = np.concatenate([treino_ate, bloco_teste])


def validar_walk_forward(
    dados_treino: pd.DataFrame, n_splits: int = 5, horizonte: int = HORIZONTE_PADRAO
) -> pd.DataFrame:
    """Validação honesta: treina só com o passado, testa no bloco futuro seguinte.

    Descarta as últimas `horizonte` datas de cada bloco de treino, porque o
    alvo (`retorno_futuro`) dessas linhas só se resolve depois do início do
    bloco de teste — sem esse corte, o treino "veria" indiretamente preços
    do período de teste através do próprio alvo.
    """
    datas = np.sort(dados_treino["data_pregao"].unique())
    resultados = []

    for datas_treino, datas_teste in _splits_walk_forward(datas, n_splits):
        datas_treino_seguras = datas_treino[:-horizonte] if horizonte > 0 else datas_treino
        treino = dados_treino[dados_treino["data_pregao"].isin(datas_treino_seguras)]
        teste = dados_treino[dados_treino["data_pregao"].isin(datas_teste)]
        if len(treino) < 200 or len(teste) < 20:
            continue

        modelo = HistGradientBoostingRegressor(random_state=42)
        modelo.fit(treino[FEATURE_COLS], treino["retorno_futuro"])
        previsto = modelo.predict(teste[FEATURE_COLS])
        real = teste["retorno_futuro"].to_numpy()

        acerto_direcao = float(np.mean(np.sign(previsto) == np.sign(real)))
        r2 = float(1 - np.sum((real - previsto) ** 2) / np.sum((real - real.mean()) ** 2))
        resultados.append({
            "treino_de": pd.Timestamp(datas_treino_seguras.min()).date(),
            "treino_ate": pd.Timestamp(datas_treino_seguras.max()).date(),
            "n_linhas_treino": len(treino),
            "periodo_teste_inicio": pd.Timestamp(datas_teste.min()).date(),
            "periodo_teste_fim": pd.Timestamp(datas_teste.max()).date(),
            "n_amostras": len(teste),
            "acerto_direcao": round(acerto_direcao, 3),
            "r2": round(r2, 3),
        })

    return pd.DataFrame(resultados)


def treinar_modelo_final(dados_treino: pd.DataFrame) -> HistGradientBoostingRegressor:
    modelo = HistGradientBoostingRegressor(random_state=42)
    modelo.fit(dados_treino[FEATURE_COLS], dados_treino["retorno_futuro"])
    return modelo


def prever_ultimo_pregao(df_features: pd.DataFrame, modelo: HistGradientBoostingRegressor) -> pd.DataFrame:
    """Usa a linha mais recente de cada ticker (sem alvo, pois é o futuro real) para prever."""
    ultima_data = df_features["data_pregao"].max()
    atual = df_features[df_features["data_pregao"] == ultima_data].dropna(subset=FEATURE_COLS).copy()
    atual["retorno_previsto"] = modelo.predict(atual[FEATURE_COLS])
    return atual[["ticker", "data_pregao", "preco_fechamento", "retorno_previsto"]].sort_values(
        "retorno_previsto", ascending=False
    )


def _custo_operacional(
    valor_financeiro: float,
    taxa_b3: float,
    corretagem_fixa: float,
    corretagem_percentual: float,
    iss_pct: float,
) -> tuple[float, float]:
    """Custo de UMA ponta (compra OU venda), decomposto em (emolumentos, corretagem+ISS)."""
    emolumentos = valor_financeiro * taxa_b3
    corretagem = corretagem_fixa + valor_financeiro * corretagem_percentual
    corretagem_com_iss = corretagem + corretagem * iss_pct
    return emolumentos, corretagem_com_iss


def _alocar_greedy(candidatos: pd.DataFrame, capital_disponivel: float, capital_max_por_acao: float) -> pd.DataFrame:
    """Percorre candidatos (já ordenados) comprando até `capital_max_por_acao` de cada,
    respeitando o capital disponível. Compra em lotes de ações inteiras."""
    linhas = []
    capital_restante = capital_disponivel
    for _, ativo in candidatos.iterrows():
        if capital_restante < ativo["preco_fechamento"]:
            continue
        valor_alvo = min(capital_max_por_acao, capital_restante)
        quantidade = int(valor_alvo // ativo["preco_fechamento"])
        if quantidade <= 0:
            continue
        valor_efetivo = quantidade * ativo["preco_fechamento"]
        capital_restante -= valor_efetivo
        linha = ativo.to_dict()
        linha["quantidade_acoes"] = quantidade
        linha["valor_alocado"] = round(valor_efetivo, 2)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def alocar_capital(
    ranking: pd.DataFrame,
    capital: float,
    top_n: int = 10,
    taxa_b3: float = TAXA_B3_PADRAO,
    corretagem_fixa: float = CORRETAGEM_PADRAO,
    corretagem_percentual: float = CORRETAGEM_PERCENTUAL_PADRAO,
    iss_pct: float = ISS_PADRAO,
    capital_max_por_acao: float | None = None,
) -> pd.DataFrame:
    """Aloca capital entre os top_n com retorno previsto positivo, líquido de custos.

    Compra até `capital_max_por_acao` por ativo (padrão: sem limite, usa todo o
    capital se necessário), na ordem do ranking, até esgotar o capital total.
    """
    custo_pct_ida_volta = 2 * (taxa_b3 + corretagem_percentual * (1 + iss_pct))
    selecionados = ranking.head(top_n).copy()
    selecionados["retorno_liquido_estimado"] = selecionados["retorno_previsto"] - custo_pct_ida_volta
    selecionados = selecionados[selecionados["retorno_liquido_estimado"] > 0]
    if selecionados.empty:
        return selecionados.assign(valor_alocado=[], quantidade_acoes=[], custo_estimado=[])

    limite = capital_max_por_acao if capital_max_por_acao is not None else capital
    carteira = _alocar_greedy(selecionados, capital, limite)
    if carteira.empty:
        return carteira.assign(custo_estimado=[])

    custo_por_ponta = carteira["valor_alocado"].apply(
        lambda v: sum(_custo_operacional(v, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct))
    )
    carteira["custo_estimado"] = (2 * custo_por_ponta).round(2)  # compra + venda
    return carteira


def _conectar(caminho_db: str = DB_PADRAO) -> sqlite3.Connection:
    con = sqlite3.connect(caminho_db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS decisoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_decisao TEXT NOT NULL,
            ticker TEXT NOT NULL,
            preco_na_decisao REAL NOT NULL,
            retorno_previsto REAL NOT NULL,
            horizonte_pregoes INTEGER NOT NULL,
            valor_alocado REAL NOT NULL,
            quantidade_acoes INTEGER NOT NULL,
            preco_realizado REAL,
            retorno_realizado REAL,
            UNIQUE(data_decisao, ticker)
        )
    """)
    return con


def registrar_decisoes(carteira: pd.DataFrame, horizonte: int, caminho_db: str = DB_PADRAO) -> None:
    """Grava a carteira sugerida no histórico local, para conferir o acerto depois."""
    if carteira.empty:
        return
    with _conectar(caminho_db) as con:
        for _, linha in carteira.iterrows():
            con.execute(
                """INSERT OR IGNORE INTO decisoes
                   (data_decisao, ticker, preco_na_decisao, retorno_previsto,
                    horizonte_pregoes, valor_alocado, quantidade_acoes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(linha["data_pregao"].date()), linha["ticker"], float(linha["preco_fechamento"]),
                    float(linha["retorno_previsto"]), int(horizonte),
                    float(linha["valor_alocado"]), int(linha["quantidade_acoes"]),
                ),
            )


def reconciliar_historico(df_precos_atualizado: pd.DataFrame, caminho_db: str = DB_PADRAO) -> pd.DataFrame:
    """Para decisões antigas ainda sem resultado, busca o preço `horizonte` pregões à
    frente nos dados atualizados (se já disponível) e calcula o retorno realizado."""
    with _conectar(caminho_db) as con:
        pendentes = pd.read_sql(
            "SELECT * FROM decisoes WHERE retorno_realizado IS NULL", con,
            parse_dates=["data_decisao"],
        )
        if pendentes.empty:
            return pd.read_sql("SELECT * FROM decisoes", con, parse_dates=["data_decisao"])

        precos = df_precos_atualizado.sort_values(["ticker", "data_pregao"])
        for _, linha in pendentes.iterrows():
            serie = precos[
                (precos["ticker"] == linha["ticker"]) & (precos["data_pregao"] > linha["data_decisao"])
            ]
            if len(serie) < linha["horizonte_pregoes"]:
                continue  # ainda não passou tempo suficiente nos dados disponíveis
            preco_futuro = serie.iloc[int(linha["horizonte_pregoes"]) - 1]["preco_fechamento"]
            retorno = preco_futuro / linha["preco_na_decisao"] - 1
            con.execute(
                "UPDATE decisoes SET preco_realizado = ?, retorno_realizado = ? WHERE id = ?",
                (float(preco_futuro), float(retorno), int(linha["id"])),
            )

        return pd.read_sql("SELECT * FROM decisoes", con, parse_dates=["data_decisao"])


def backtest(
    df_features: pd.DataFrame,
    capital_total: float,
    capital_max_por_acao: float,
    horizonte: int = HORIZONTE_PADRAO,
    top_n: int = 10,
    taxa_b3: float = TAXA_B3_PADRAO,
    corretagem_fixa: float = CORRETAGEM_PADRAO,
    corretagem_percentual: float = CORRETAGEM_PERCENTUAL_PADRAO,
    iss_pct: float = ISS_PADRAO,
    irrf_pct: float = IRRF_SWING_PCT,
    min_dias_treino: int = 120,
    data_inicio: str | pd.Timestamp | None = None,
    data_fim: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula compra/venda ao longo do histórico disponível (ou de uma janela dele).

    A cada `horizonte` pregões, treina o modelo só com dados anteriores (sem
    vazamento), escolhe as ações com retorno previsto líquido positivo, compra
    limitado a `capital_max_por_acao` por ação até esgotar `capital_total`
    disponível naquele momento, e realiza o resultado usando o retorno real
    ocorrido (já calculado em `retorno_futuro`). Capital e lucro OPERACIONAL
    (antes de IR) são compostos entre os períodos. O IR/IRRF é calculado à
    parte por `calcular_ir_mensal`, pois no Brasil ele é apurado por mês, não
    por operação.

    Importante sobre o treino: só entram linhas cujo alvo (`retorno_futuro`,
    calculado com `horizonte` pregões de folga) já esteja totalmente resolvido
    ANTES da data de decisão. Sem esse corte, linhas de treino muito recentes
    carregariam informação de preço do próprio período de teste através do
    alvo, um vazamento sutil de dados do futuro. `treino_ate`, no resumo
    retornado, mostra exatamente até que data cada decisão usou dados.

    `data_inicio`/`data_fim` restringem as datas de DECISÃO (compra) simuladas:
    - Só `data_inicio` (== `data_fim`): simula um único dia de decisão.
    - Ambos preenchidos: simula todas as decisões dentro do intervalo (ainda
      espaçadas por `horizonte` pregões).
    - Nenhum: comportamento padrão, usa todo o histórico disponível.
    Em qualquer caso, o treino do modelo em cada decisão usa só dados
    anteriores à própria data de decisão (walk-forward), mesmo que
    `data_inicio` esteja no meio do histórico.

    Retorna (operacoes, resumo_por_periodo).
    """
    base = df_features.dropna(subset=FEATURE_COLS).copy()
    datas = np.sort(base["data_pregao"].unique())
    custo_pct_ida_volta = 2 * (taxa_b3 + corretagem_percentual * (1 + iss_pct))

    capital_disponivel = capital_total
    operacoes = []
    resumo = []

    i = min_dias_treino
    if data_inicio is not None:
        i = max(i, int(np.searchsorted(datas, np.datetime64(pd.Timestamp(data_inicio)))))
    limite_fim = np.datetime64(pd.Timestamp(data_fim)) if data_fim is not None else None

    while i < len(datas):
        data_decisao = datas[i]
        if limite_fim is not None and data_decisao > limite_fim:
            break

        corte_idx = i - horizonte  # última data cujo alvo (retorno_futuro) já está resolvido
        if corte_idx < 0:
            i += horizonte
            continue
        data_treino_ate = datas[corte_idx]
        treino = base[(base["data_pregao"] <= data_treino_ate) & base["retorno_futuro"].notna()]
        candidatos = base[base["data_pregao"] == data_decisao].dropna(subset=["retorno_futuro"])

        if len(treino) < 200 or candidatos.empty:
            i += horizonte
            continue

        modelo = HistGradientBoostingRegressor(random_state=42)
        modelo.fit(treino[FEATURE_COLS], treino["retorno_futuro"])
        candidatos = candidatos.copy()
        candidatos["retorno_previsto"] = modelo.predict(candidatos[FEATURE_COLS])
        candidatos["retorno_liquido_estimado"] = candidatos["retorno_previsto"] - custo_pct_ida_volta
        candidatos = candidatos[candidatos["retorno_liquido_estimado"] > 0].sort_values(
            "retorno_previsto", ascending=False
        ).head(top_n)

        selecionados = _alocar_greedy(candidatos, capital_disponivel, capital_max_por_acao)
        data_venda = pd.Timestamp(datas[i + horizonte]) if i + horizonte < len(datas) else pd.NaT

        lucro_periodo = 0.0
        for _, ativo in selecionados.iterrows():
            preco_compra = ativo["preco_fechamento"]
            valor_compra = ativo["valor_alocado"]
            preco_venda_real = preco_compra * (1 + ativo["retorno_futuro"])
            valor_venda = ativo["quantidade_acoes"] * preco_venda_real

            emol_compra, corr_compra = _custo_operacional(valor_compra, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct)
            emol_venda, corr_venda = _custo_operacional(valor_venda, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct)
            emolumentos = emol_compra + emol_venda
            corretagem = corr_compra + corr_venda
            custo = emolumentos + corretagem
            irrf = valor_venda * irrf_pct
            irrf = irrf if irrf >= IRRF_MINIMO else 0.0

            lucro = valor_venda - valor_compra - custo
            lucro_periodo += lucro
            operacoes.append({
                "data_compra": pd.Timestamp(data_decisao).date(),
                "data_venda": data_venda.date() if pd.notna(data_venda) else None,
                "ticker": ativo["ticker"],
                "preco_compra": round(preco_compra, 2),
                "quantidade": int(ativo["quantidade_acoes"]),
                "preco_venda": round(preco_venda_real, 2),
                "valor_venda": round(valor_venda, 2),
                "retorno_previsto": round(ativo["retorno_previsto"], 4),
                "retorno_realizado": round(float(ativo["retorno_futuro"]), 4),
                "emolumentos": round(emolumentos, 2),
                "corretagem": round(corretagem, 2),
                "custo": round(custo, 2),
                "irrf": round(irrf, 2),
                "lucro": round(lucro, 2),
            })

        capital_disponivel += lucro_periodo
        resumo.append({
            "treino_de": pd.Timestamp(datas[0]).date(),
            "treino_ate": pd.Timestamp(data_treino_ate).date(),
            "n_linhas_treino": len(treino),
            "data_decisao": pd.Timestamp(data_decisao).date(),
            "data_venda_prevista": data_venda.date() if pd.notna(data_venda) else None,
            "n_operacoes": len(selecionados),
            "lucro_periodo": round(lucro_periodo, 2),
            "capital_apos_periodo": round(capital_disponivel, 2),
        })
        i += horizonte

    return pd.DataFrame(operacoes), pd.DataFrame(resumo)


def calcular_ir_mensal(
    operacoes: pd.DataFrame,
    ir_aliquota: float = IR_SWING_ALIQUOTA,
    isencao_vendas_mensal: float = ISENCAO_VENDAS_MENSAL,
) -> pd.DataFrame:
    """Apura IR mês a mês (regra brasileira de swing trade em ações).

    Regras aplicadas (Lei 11.033/2004 art. 3º, II + regulamentação da Receita
    Federal): lucro/prejuízo de cada mês é somado ao saldo de prejuízo acumulado
    de meses anteriores (compensação sem prazo de validade); se o total vendido
    no mês for menor ou igual a `isencao_vendas_mensal`, o IR do mês fica
    isento (mas o resultado ainda é usado para compensação de prejuízo);
    caso contrário, aplica-se `ir_aliquota` sobre o ganho líquido do mês, e o
    IRRF já retido nas vendas é abatido do valor a pagar (DARF).

    Simplificação assumida: eventual IRRF que exceda o IR devido no mês não é
    modelado como restituição (na prática, seria compensado/restituído na
    declaração anual). Retorna DataFrame vazio se `operacoes` estiver vazio.
    """
    if operacoes.empty:
        return pd.DataFrame()

    op = operacoes.dropna(subset=["data_venda"]).copy()
    op["data_venda"] = pd.to_datetime(op["data_venda"])
    op["mes"] = op["data_venda"].dt.to_period("M")

    mensal = op.groupby("mes").agg(vendas_mes=("valor_venda", "sum"), lucro_mes=("lucro", "sum"),
                                    irrf_retido_mes=("irrf", "sum")).reset_index().sort_values("mes")

    linhas = []
    prejuizo_acumulado = 0.0
    for _, row in mensal.iterrows():
        resultado = row["lucro_mes"] - prejuizo_acumulado
        if resultado <= 0:
            prejuizo_acumulado = -resultado
            ganho_tributavel = 0.0
        else:
            prejuizo_acumulado = 0.0
            ganho_tributavel = resultado

        isento = row["vendas_mes"] <= isencao_vendas_mensal
        ir_devido = 0.0 if isento else ganho_tributavel * ir_aliquota
        darf = max(0.0, ir_devido - row["irrf_retido_mes"])

        linhas.append({
            "mes": str(row["mes"]),
            "vendas_mes": round(row["vendas_mes"], 2),
            "lucro_mes": round(row["lucro_mes"], 2),
            "isento": isento,
            "prejuizo_acumulado_apos": round(prejuizo_acumulado, 2),
            "irrf_retido_mes": round(row["irrf_retido_mes"], 2),
            "ir_devido": round(ir_devido, 2),
            "darf_a_pagar": round(darf, 2),
        })

    return pd.DataFrame(linhas)


def _intervalo_datas(resumo: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Primeira data_decisao e última data_venda_prevista (ou data_decisao, se a
    última não tiver venda prevista) do backtest — usado para CAGR e para
    comparar com buy & hold no mesmo intervalo de calendário."""
    data_inicio = pd.Timestamp(resumo["data_decisao"].min())
    datas_venda = pd.to_datetime(resumo["data_venda_prevista"]).dropna()
    data_fim = datas_venda.max() if not datas_venda.empty else pd.Timestamp(resumo["data_decisao"].max())
    return data_inicio, data_fim


def comparar_buy_and_hold(
    resumo: pd.DataFrame,
    capital_inicial: float,
    ibovespa: pd.DataFrame | None = None,
    selic: pd.DataFrame | None = None,
) -> dict:
    """Quanto o mesmo `capital_inicial` teria virado comprando e segurando o
    Ibovespa, e aplicando na Selic, no MESMO intervalo de calendário do backtest
    (primeira decisão até a última venda prevista).

    Valores BRUTOS (sem IR): a regra de IR de swing trade em ações usada em
    `calcular_ir_mensal` não se aplica a fundos/ETF de índice nem a Tesouro
    Selic (têm regras próprias, tabela regressiva por prazo), então aplicar o
    mesmo modelo tributário aqui seria inventar um número. Por isso a
    comparação abaixo fica no bruto para os três lados: quem quiser o líquido
    da estratégia de ações, veja `patrimonio_final` em `calcular_metricas`
    (esse sim líquido de IR) e desconte o IR do Ibovespa/Selic à parte.
    """
    if resumo.empty:
        return {}
    data_inicio, data_fim = _intervalo_datas(resumo)
    resultado = {}

    if ibovespa is not None and not ibovespa.empty:
        ibov = ibovespa.sort_values("data")
        antes_inicio = ibov.loc[ibov["data"] <= data_inicio, "fechamento"]
        antes_fim = ibov.loc[ibov["data"] <= data_fim, "fechamento"]
        if not antes_inicio.empty and not antes_fim.empty:
            preco_inicio, preco_fim = float(antes_inicio.iloc[-1]), float(antes_fim.iloc[-1])
            patrimonio = capital_inicial * preco_fim / preco_inicio
            resultado["ibovespa"] = {
                "patrimonio_final_bruto": round(patrimonio, 2),
                "resultado_bruto": round(patrimonio - capital_inicial, 2),
                "retorno_pct": round(preco_fim / preco_inicio - 1, 4),
            }

    if selic is not None and not selic.empty:
        taxas = selic.set_index("data")["taxa_diaria"]
        janela = taxas[(taxas.index >= data_inicio) & (taxas.index <= data_fim)]
        if len(janela) > 0:
            fator = float(np.prod(1 + janela.to_numpy()))
            patrimonio = capital_inicial * fator
            resultado["selic"] = {
                "patrimonio_final_bruto": round(patrimonio, 2),
                "resultado_bruto": round(patrimonio - capital_inicial, 2),
                "retorno_pct": round(fator - 1, 4),
            }

    return resultado


def calcular_drawdown_maximo_percentual(curva: np.ndarray) -> float:
    if len(curva) < 2:
        return 0.0

    maior_variacao_pct = 0.0  # Guardará o drawdown acumulado em decimal negativo (ex: -0.0462)
    pico_global = curva[0]    # Maior valor já visto até o momento (running max)

    i = 0
    while i < len(curva) - 1:
        # Atualiza o pico global sempre que a curva sobe acima do máximo anterior
        if curva[i] > pico_global:
            pico_global = curva[i]

        # Verifica se começou uma perna de baixa
        if curva[i + 1] < curva[i]:
            vale = curva[i + 1]  # Captura o 1º dia de queda como o vale inicial
            i = i + 1
            # Enquanto continuar caindo nos dias seguintes...
            while i < len(curva) - 1 and curva[i + 1] < curva[i]:
                i = i + 1
                vale = curva[i]  # Atualiza o vale apenas se o próximo dia for de queda

            # Calcula a variação percentual do PICO GLOBAL até o fundo
            var_pct = (vale - pico_global) / pico_global
            if var_pct < maior_variacao_pct:
                maior_variacao_pct = var_pct
        else:
            i = i + 1

    # Garante que o último ponto também seja considerado no pico global
    if curva[-1] > pico_global:
        pico_global = curva[-1]

    # Retorna o valor percentual positivo (ex: 4.62 para 4.62%)
    return -maior_variacao_pct


def calcular_drawdown_maximo(curva: np.ndarray):
    if len(curva) < 2:
        return 0.0

    atual = curva[1]
    ant = curva[0]
    maior_variacao = 0

    i = 1
    while i < len(curva):
        taxa_variacao = atual - ant
        if taxa_variacao < 0:
            if taxa_variacao < maior_variacao:
                maior_variacao = taxa_variacao
                i = i + 1
                if i < len(curva):
                    ant = atual
                    atual = curva[i]

                # Adicionada a verificação (i < len(curva)) ANTES de acessar curva[i]
                while i < len(curva) and (atual - ant) < 0:
                    taxa_variacao = taxa_variacao + (atual - ant)
                    maior_variacao = taxa_variacao
                    i = i + 1
                    if i < len(curva):
                        ant = atual
                        atual = curva[i]
            else:
                # Isola a nova sequência de quedas para não misturar com o 'taxa_variacao' antigo
                acumulado_temp = taxa_variacao

                i = i + 1
                if i < len(curva):
                    ant = atual
                    atual = curva[i]

                    # Enquanto continuar caindo nos dias seguintes...
                    while i < len(curva) and (atual - ant) < 0:
                        acumulado_temp = acumulado_temp + (atual - ant)
                        if acumulado_temp < maior_variacao:
                            maior_variacao = acumulado_temp
                        i = i + 1
                        if i < len(curva):
                            ant = atual
                            atual = curva[i]
        else:
            i = i + 1
            if i < len(curva):
                ant = atual
                atual = curva[i]

    return -(maior_variacao)
            




def calcular_metricas(
    resumo: pd.DataFrame,
    operacoes: pd.DataFrame,
    capital_inicial: float,
    horizonte: int,
    ir_mensal: pd.DataFrame | None = None,
    risco_livre_anual: float = 0.0,
    retorno_livre_risco_periodo: pd.Series | None = None,
    benchmark_retornos: pd.Series | None = None,
) -> dict:
    """Métricas de desempenho do backtest completo.

    `resumo`/`operacoes` são as saídas de `backtest()`; `ir_mensal` é a saída
    de `calcular_ir_mensal()` (opcional; sem ela, `impostos` fica 0).

    A curva de patrimônio usada em drawdown/Sharpe/Sortino/Calmar é a
    OPERACIONAL (bruta, antes de IR), a mesma de `resumo["capital_apos_periodo"]`,
    porque o IR é apurado por mês (não por período de `horizonte`) — misturar
    os dois exigiria redistribuir o IR entre períodos de forma arbitrária.
    `patrimonio_final` e `resultado_liquido` já descontam o IR total, à parte.

    Sharpe/Sortino anualizam usando 252/`horizonte` períodos por ano (dias úteis
    padrão da B3). Taxa livre de risco: se `retorno_livre_risco_periodo` for
    passado (ex.: saída de `preparar_benchmark`, com a Selic real de cada
    janela), ela é usada período a período; qualquer período faltante (ou se o
    parâmetro não for passado) cai no escalar `risco_livre_anual` (padrão 0%).

    `benchmark_retornos`: Series com um retorno por linha de `resumo` (mesmo
    tamanho, mesma ordem), retorno do período de um índice de referência (ex.
    Ibovespa) nas mesmas janelas de `data_decisao`/`data_venda_prevista`. Sem
    isso, `beta` e `alfa_jensen_anualizado` ficam None: não existe cálculo de
    beta/alfa sem uma série de mercado para comparar.
    """
    if resumo.empty:
        return {}

    patrimonio_bruto_final = float(resumo["capital_apos_periodo"].iloc[-1])
    total_darf = float(ir_mensal["darf_a_pagar"].sum()) if ir_mensal is not None and not ir_mensal.empty else 0.0
    patrimonio_liquido_final = patrimonio_bruto_final - total_darf

    # curva de patrimônio (bruta): capital inicial + capital_apos_periodo de cada período
    curva = np.concatenate([[capital_inicial], resumo["capital_apos_periodo"].to_numpy(dtype=float)])
    pico = np.maximum.accumulate(curva)
    dd_serie = curva - pico
    idx_min = int(np.argmin(dd_serie))
    drawdown_maximo_rs = float(-dd_serie[idx_min])
    drawdown_maximo_rs = calcular_drawdown_maximo(curva)
    pico_no_ponto = float(pico[idx_min])
    drawdown_relativo_pct = drawdown_maximo_rs / pico_no_ponto if pico_no_ponto > 0 else np.nan
    drawdown_relativo_pct = calcular_drawdown_maximo_percentual(curva)
    drawdown_absoluto_rs = float(max(0.0, capital_inicial - curva.min()))

    capital_inicio_periodo = curva[:-1]
    retornos_periodo = resumo["lucro_periodo"].to_numpy(dtype=float) / capital_inicio_periodo
    periodos_por_ano = 252 / horizonte if horizonte > 0 else np.nan
    rf_escalar = (1 + risco_livre_anual) ** (1 / periodos_por_ano) - 1 if periodos_por_ano > 0 else 0.0
    if retorno_livre_risco_periodo is not None and len(retorno_livre_risco_periodo) == len(retornos_periodo):
        rf_periodo = np.asarray(retorno_livre_risco_periodo, dtype=float)
        rf_periodo = np.where(np.isnan(rf_periodo), rf_escalar, rf_periodo)
    else:
        rf_periodo = np.full(len(retornos_periodo), rf_escalar)

    excesso = retornos_periodo - rf_periodo
    desvio = excesso.std(ddof=1) if len(excesso) > 1 else np.nan
    sharpe = excesso.mean() / desvio if desvio and desvio > 0 else np.nan

    downside = np.clip(excesso, a_min=None, a_max=0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else np.nan
    sortino = excesso.mean() / downside_dev  if downside_dev and downside_dev > 0 else np.nan

    data_inicio_bt, data_fim_bt = _intervalo_datas(resumo)
    anos = max((data_fim_bt - data_inicio_bt).days / 365.25, horizonte / 252)
    cagr = (patrimonio_bruto_final / capital_inicial) ** (1 / anos) - 1
    calmar = cagr / drawdown_relativo_pct if drawdown_relativo_pct and drawdown_relativo_pct > 0 else np.nan

    ops_realizadas = operacoes.dropna(subset=["retorno_realizado"]) if not operacoes.empty else operacoes
    taxa_acerto = float((ops_realizadas["lucro"] > 0).mean()) if not ops_realizadas.empty else np.nan
    lucro_bruto_ops = ops_realizadas.loc[ops_realizadas["lucro"] > 0, "lucro"].sum() if not ops_realizadas.empty else 0.0
    prejuizo_bruto_ops = ops_realizadas.loc[ops_realizadas["lucro"] < 0, "lucro"].sum() if not ops_realizadas.empty else 0.0
    profit_factor = lucro_bruto_ops / abs(prejuizo_bruto_ops) if prejuizo_bruto_ops < 0 else np.nan

    beta = alfa_jensen = np.nan
    if benchmark_retornos is not None and len(benchmark_retornos) == len(retornos_periodo):
        bench = np.asarray(benchmark_retornos, dtype=float)
        valido = ~np.isnan(bench)
        if valido.sum() > 1:
            variancia_bench = (bench[valido] - rf_periodo[valido]).var(ddof=1)
            if variancia_bench > 0:
                covariancia = np.cov((retornos_periodo[valido] - rf_periodo[valido]), (bench[valido] - rf_periodo[valido]), ddof=1)[0, 1]
                beta = covariancia / variancia_bench
                alfa_por_periodo = (retornos_periodo[valido] - rf_periodo[valido]) - beta * (bench[valido] - rf_periodo[valido])
                alfa_jensen = alfa_por_periodo.mean()

    def _ou_none(x: float) -> float | None:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

    return {
        "capital_inicial": round(capital_inicial, 2),
        "patrimonio_final": round(patrimonio_liquido_final, 2),
        "resultado_liquido": round(patrimonio_liquido_final - capital_inicial, 2),
        "taxa_acerto": _ou_none(taxa_acerto),
        "alfa_jensen": _ou_none(alfa_jensen),
        "beta": _ou_none(beta),
        "numero_ativos_negociados": int(operacoes["ticker"].nunique()) if not operacoes.empty else 0,
        "profit_factor": _ou_none(profit_factor),
        "taxas_e_corretagem": round(float(operacoes["corretagem"].sum()), 2) if "corretagem" in operacoes else 0.0,
        "emolumentos": round(float(operacoes["emolumentos"].sum()), 2) if "emolumentos" in operacoes else 0.0,
        "impostos": round(total_darf, 2),
        "sharpe": _ou_none(sharpe),
        "sortino": _ou_none(sortino),
        "calmar": _ou_none(calmar),
        "drawdown_absoluto_rs": round(drawdown_absoluto_rs, 2),
        "drawdown_maximo_rs": round(drawdown_maximo_rs, 2),
        "drawdown_relativo_pct": _ou_none(drawdown_relativo_pct),
    }
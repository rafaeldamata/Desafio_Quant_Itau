#!/usr/bin/env python3
"""
Pipeline para baixar e converter as cotações históricas COTAHIST da B3.

Exemplos:
    python b3_cotahist.py --inicio 2020 --fim 2026 --formato parquet
    python b3_cotahist.py --inicio 1986 --fim 2026 --formato parquet --consolidar
    python b3_cotahist.py --inicio 2025 --fim 2026 --tickers PETR4 VALE3 ITUB4

Dependências:
    pip install pandas pyarrow requests
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# Endpoint histórico tradicional da B3/BM&FBovespa.
URLS = [
    "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP",
    "http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP",
]

COLSPECS = [
    (0, 2),    # TIPREG
    (2, 10),   # DATA
    (10, 12),  # CODBDI
    (12, 24),  # CODNEG
    (24, 27),  # TPMERC
    (27, 39),  # NOMRES
    (39, 49),  # ESPECI
    (49, 52),  # PRAZOT
    (52, 56),  # MODREF
    (56, 69),  # PREABE
    (69, 82),  # PREMAX
    (82, 95),  # PREMIN
    (95, 108), # PREMED
    (108, 121),# PREULT
    (121, 134),# PREOFC
    (134, 147),# PREOFV
    (147, 152),# TOTNEG
    (152, 170),# QUATOT
    (170, 188),# VOLTOT
    (188, 201),# PREEXE
    (201, 202),# INDOPC
    (202, 210),# DATVEN
    (210, 217),# FATCOT
    (217, 230),# PTOEXE
    (230, 242),# CODISI
    (242, 245),# DISMES
]

NAMES = [
    "tipo_registro",
    "data_pregao",
    "codigo_bdi",
    "ticker",
    "tipo_mercado",
    "nome_resumido",
    "especificacao",
    "prazo_termo",
    "moeda",
    "preco_abertura",
    "preco_maximo",
    "preco_minimo",
    "preco_medio",
    "preco_fechamento",
    "melhor_oferta_compra",
    "melhor_oferta_venda",
    "numero_negocios",
    "quantidade_negociada",
    "volume_financeiro",
    "preco_exercicio",
    "indicador_correcao",
    "data_vencimento",
    "fator_cotacao",
    "pontos_exercicio",
    "codigo_isin",
    "distribuicao",
]

PRICE_COLUMNS = [
    "preco_abertura",
    "preco_maximo",
    "preco_minimo",
    "preco_medio",
    "preco_fechamento",
    "melhor_oferta_compra",
    "melhor_oferta_venda",
    "preco_exercicio",
    "pontos_exercicio",
]

INTEGER_COLUMNS = [
    "tipo_registro",
    "tipo_mercado",
    "numero_negocios",
    "quantidade_negociada",
    "fator_cotacao",
    "distribuicao",
]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; B3-COTAHIST-Pipeline/1.0; "
                "+https://www.b3.com.br/)"
            )
        }
    )
    return session


def download_year(
    session: requests.Session,
    ano: int,
    destino: Path,
    sobrescrever: bool = False,
    tentativas: int = 3,
) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"COTAHIST_A{ano}.ZIP"

    if arquivo.exists() and arquivo.stat().st_size > 0 and not sobrescrever:
        if zipfile.is_zipfile(arquivo):
            logging.info("%s já existe e é um ZIP válido.", arquivo.name)
            return arquivo
        logging.warning("%s existe, mas está inválido; será baixado novamente.", arquivo.name)
        arquivo.unlink()

    erros: list[str] = []

    for template in URLS:
        url = template.format(ano=ano)

        for tentativa in range(1, tentativas + 1):
            parcial = arquivo.with_suffix(".ZIP.part")
            try:
                logging.info("Baixando %d: %s", ano, url)
                with session.get(url, stream=True, timeout=(20, 180)) as resposta:
                    resposta.raise_for_status()

                    content_type = resposta.headers.get("content-type", "").lower()
                    if "text/html" in content_type:
                        raise RuntimeError(
                            "O servidor retornou HTML em vez do arquivo ZIP."
                        )

                    with parcial.open("wb") as handle:
                        for bloco in resposta.iter_content(chunk_size=1024 * 1024):
                            if bloco:
                                handle.write(bloco)

                if not zipfile.is_zipfile(parcial):
                    raise RuntimeError("O conteúdo baixado não é um ZIP válido.")

                parcial.replace(arquivo)
                logging.info("Download concluído: %s", arquivo)
                return arquivo

            except Exception as exc:
                erros.append(f"{url} (tentativa {tentativa}): {exc}")
                logging.warning("Falha: %s", exc)
                parcial.unlink(missing_ok=True)
                time.sleep(min(2 ** tentativa, 10))

    raise RuntimeError(
        f"Não foi possível baixar o COTAHIST de {ano}.\n" + "\n".join(erros)
    )


def find_txt_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        txts = [name for name in zf.namelist() if name.upper().endswith(".TXT")]
        if not txts:
            raise RuntimeError(f"Nenhum TXT encontrado em {zip_path.name}.")
        return txts[0]


def parse_cotahist(zip_path: Path, tickers: set[str] | None = None) -> pd.DataFrame:
    txt_name = find_txt_in_zip(zip_path)

    with zipfile.ZipFile(zip_path) as zf, zf.open(txt_name) as arquivo:
        df = pd.read_fwf(
            arquivo,
            colspecs=COLSPECS,
            names=NAMES,
            dtype=str,
            encoding="latin-1",
            skiprows=1,
            skipfooter=1,
            engine="python",
        )

    # Somente registros de cotação.
    df = df[df["tipo_registro"].eq("01")].copy()

    for col in [
        "codigo_bdi",
        "ticker",
        "nome_resumido",
        "especificacao",
        "prazo_termo",
        "moeda",
        "codigo_isin",
    ]:
        df[col] = df[col].astype("string").str.strip()

    if tickers:
        df = df[df["ticker"].isin(tickers)].copy()

    df["data_pregao"] = pd.to_datetime(
        df["data_pregao"], format="%Y%m%d", errors="coerce"
    )

    vencimento = df["data_vencimento"].replace({"99991231": pd.NA, "00000000": pd.NA})
    df["data_vencimento"] = pd.to_datetime(
        vencimento, format="%Y%m%d", errors="coerce"
    )

    # No COTAHIST, preços e volumes monetários vêm sem separador decimal.
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    for col in INTEGER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # O volume financeiro também possui duas casas decimais implícitas.
    df["volume_financeiro"] = (
        pd.to_numeric(df["volume_financeiro"], errors="coerce") / 100
    )

    df["codigo_bdi"] = pd.to_numeric(
        df["codigo_bdi"], errors="coerce"
    ).astype("Int64")
    df["indicador_correcao"] = pd.to_numeric(
        df["indicador_correcao"], errors="coerce"
    ).astype("Int64")

    df["ano"] = df["data_pregao"].dt.year.astype("Int64")

    return df.reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output: Path, formato: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    if formato == "parquet":
        df.to_parquet(output, index=False, compression="snappy")
    elif formato == "csv":
        df.to_csv(output, index=False, encoding="utf-8")
    else:
        raise ValueError(f"Formato não suportado: {formato}")

    logging.info("Arquivo salvo: %s (%d linhas)", output, len(df))
    return output


def process_year(
    session: requests.Session,
    ano: int,
    raw_dir: Path,
    processed_dir: Path,
    formato: str,
    tickers: set[str] | None,
    sobrescrever: bool,
) -> Path:
    ext = "parquet" if formato == "parquet" else "csv"
    sufixo = ""
    if tickers:
        sufixo = "_" + "-".join(sorted(tickers))

    output = processed_dir / f"cotahist_{ano}{sufixo}.{ext}"

    if output.exists() and output.stat().st_size > 0 and not sobrescrever:
        logging.info("%s já processado.", output.name)
        return output

    zip_path = download_year(
        session=session,
        ano=ano,
        destino=raw_dir,
        sobrescrever=sobrescrever,
    )
    df = parse_cotahist(zip_path, tickers=tickers)
    return save_dataframe(df, output, formato)


def consolidate(files: Iterable[Path], output: Path, formato: str) -> Path:
    frames = []

    for file in files:
        if formato == "parquet":
            frames.append(pd.read_parquet(file))
        else:
            frames.append(pd.read_csv(file, parse_dates=["data_pregao", "data_vencimento"]))

    if not frames:
        raise RuntimeError("Não há arquivos para consolidar.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["data_pregao", "ticker", "tipo_mercado", "distribuicao"],
        keep="last",
    )
    df = df.sort_values(["ticker", "data_pregao", "tipo_mercado"])

    return save_dataframe(df, output, formato)


def parse_args() -> argparse.Namespace:
    current_year = date.today().year

    parser = argparse.ArgumentParser(
        description="Baixa e converte arquivos COTAHIST anuais da B3."
    )
    parser.add_argument("--inicio", type=int, default=current_year)
    parser.add_argument("--fim", type=int, default=current_year)
    parser.add_argument(
        "--formato",
        choices=["parquet", "csv"],
        default="parquet",
    )
    parser.add_argument(
        "--diretorio",
        type=Path,
        default=Path("dados_b3"),
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Filtra tickers específicos, por exemplo: PETR4 VALE3 ITUB4.",
    )
    parser.add_argument(
        "--consolidar",
        action="store_true",
        help="Cria também um arquivo único com todos os anos.",
    )
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="Baixa e processa novamente arquivos existentes.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.inicio < 1986:
        logging.error("A série histórica oficial começa em 1986.")
        return 2

    if args.fim < args.inicio:
        logging.error("--fim deve ser maior ou igual a --inicio.")
        return 2

    tickers = {ticker.strip().upper() for ticker in args.tickers or []} or None
    raw_dir = args.diretorio / "raw"
    processed_dir = args.diretorio / "processed"

    session = build_session()
    outputs: list[Path] = []
    failures: list[tuple[int, str]] = []

    for ano in range(args.inicio, args.fim + 1):
        try:
            output = process_year(
                session=session,
                ano=ano,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                formato=args.formato,
                tickers=tickers,
                sobrescrever=args.sobrescrever,
            )
            outputs.append(output)
        except Exception as exc:
            failures.append((ano, str(exc)))
            logging.error("Ano %d não processado: %s", ano, exc)

    if args.consolidar and outputs:
        ext = "parquet" if args.formato == "parquet" else "csv"
        suffix = ""
        if tickers:
            suffix = "_" + "-".join(sorted(tickers))
        consolidated = args.diretorio / f"cotahist_{args.inicio}_{args.fim}{suffix}.{ext}"
        consolidate(outputs, consolidated, args.formato)

    if failures:
        logging.warning("Anos com falha:")
        for ano, erro in failures:
            logging.warning("  %d: %s", ano, erro)
        return 1

    logging.info("Pipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

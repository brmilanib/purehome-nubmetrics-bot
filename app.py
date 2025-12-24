import os
import re
import sqlite3
from datetime import datetime, date
from typing import Optional

import pandas as pd
import streamlit as st

# =========================
# CONFIG / DB
# =========================
# ✅ NÃO usa /tmp (no Streamlit Cloud pode sumir). Usa arquivo local.
DB_PATH = os.environ.get("DB_PATH", "snapshots.db")

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics (um por concorrente) → compara com o último upload anterior + BI clean de ATAQUE.")

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

# =========================
# DB helpers
# =========================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competitor TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
        """)
init_db()


def save_snapshot(competitor: str, snapshot_date: str, filename: str, df: pd.DataFrame) -> int:
    payload = df.to_json(orient="records", force_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO snapshots (competitor, snapshot_date, uploaded_at, filename, data_json) VALUES (?, ?, ?, ?, ?)",
            (competitor, snapshot_date, datetime.now().isoformat(timespec="seconds"), filename, payload)
        )
        return int(cur.lastrowid)


def load_last_snapshot(competitor: str):
    """Pega o ÚLTIMO snapshot salvo desse concorrente (por id desc)."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT id, snapshot_date, uploaded_at, filename, data_json
            FROM snapshots
            WHERE competitor = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (competitor,)
        ).fetchone()
    if not row:
        return None
    sid, d, uploaded_at, filename, data_json = row
    df = pd.read_json(data_json, orient="records")
    return {"id": sid, "snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}


def list_snapshots():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT competitor, snapshot_date, uploaded_at, filename, id
            FROM snapshots
            ORDER BY id DESC
            """,
            conn
        )
    return df


# =========================
# Normalização
# =========================
CANON_MAP = {
    "título": "titulo",
    "marca": "marca",
    "vendas em $": "vendas_valor",
    "vendas em unid.": "vendas_unid",
    "preço médio": "preco_medio",
    "tipo de publicação": "tipo_publicacao",
    "fulfillment": "full",
    "catálogo.": "catalogo",
    "com frete grátis": "frete_gratis",
    "com mercado envios": "mercado_envios",
    "com desconto": "desconto",
    "sku": "sku",
    "oem": "oem",
    "gtin": "gtin",
    "n° peça": "n_peca",
    "condição": "condicao",
    "estoque": "estoque",
}


def detect_snapshot_date_from_filename(filename: str) -> Optional[str]:
    """Tenta extrair uma data do nome do arquivo."""
    name = str(filename)

    # YYYY-MM-DD
    iso = re.findall(r"(\d{4}-\d{2}-\d{2})", name)
    if iso:
        return iso[-1]

    # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
    br = re.findall(r"(\d{2})[\/\-\._](\d{2})[\/\-\._](\d{4})", name)
    if br:
        dd, mm, yyyy = br[-1]
        return f"{yyyy}-{mm}-{dd}"

    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ Corrige "anúncios irmãos":
      - Estoque NÃO soma (estoque é compartilhado) -> MAX
      - Preço ref = preço mínimo (preço de guerra)
      - Cria anuncios_irmaos
      - Guarda preco_min/preco_max/preco_wavg
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: CANON_MAP.get(c, c) for c in df.columns})

    needed = [
        "sku", "gtin", "n_peca", "titulo", "marca",
        "estoque", "preco_medio", "vendas_unid",
        "desconto", "frete_gratis", "full", "tipo_publicacao"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")

    def norm_id(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return None
        # 123.0 -> 123
        if re.fullmatch(r"\d+\.0", s):
            s = s[:-2]
        # scientific notation
        if "e+" in s.lower():
            try:
                s = str(int(float(s)))
            except:
                pass
        return s

    def make_key(row):
        for k in ["sku", "gtin", "n_peca"]:
            v = norm_id(row.get(k))
            if v:
                return v
        t = row.get("titulo")
        if pd.notna(t) and str(t).strip() != "":
            return "TIT_" + str(t).strip()[:80]
        return None

    df["key"] = df.apply(make_key, axis=1)
    df = df[df["key"].notna()].copy()
    df["key"] = df["key"].astype(str)

    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else pd.NA

    def mode_or_na(s: pd.Series):
        s2 = s.dropna()
        if len(s2) == 0:
            return pd.NA
        m = s2.mode()
        return m.iloc[0] if len(m) else s2.iloc[0]

    def wavg_price(g: pd.DataFrame):
        p = g["preco_medio"]
        w = g["vendas_unid"].fillna(1).clip(lower=1)
        mask = p.notna()
        if mask.sum() == 0:
            return pd.NA
        return (p[mask] * w[mask]).sum() / w[mask].sum()

    out = (
        df.groupby("key", as_index=False)
        .agg(
            titulo=("titulo", first_non_null),
            marca=("marca", first_non_null),
            estoque=("estoque", "max"),         # ✅ NÃO soma
            vendas_unid=("vendas_unid", "sum"), # soma demanda
            preco_min=("preco_medio", "min"),
            preco_max=("preco_medio", "max"),
            desconto=("desconto", mode_or_na),
            frete_gratis=("frete_gratis", mode_or_na),
            full=("full", mode_or_na),
            tipo_publicacao=("tipo_publicacao", mode_or_na),
            anuncios_irmaos=("key", "size"),
        )
    )

    price_wavg_map = df.groupby("key").apply(wavg_price)
    out["preco_wavg"] = out["key"].map(price_wavg_map)

    # ✅ Preço de referência para competir: o menor preço (guerra)
    out["preco_ref"] = out["preco_min"]

    out = out.sort_values(["vendas_unid", "estoque"], ascending=[False, True], na_position="last")
    return out


# =========================
# Diff / Ataque
# =========================
def compute_diff(prev: pd.DataFrame, curr: pd.DataFrame,
                 stock_crit_max: int,
                 stock_drop_pct_min: float,
                 price_up_min: float):
    merged = curr.merge(prev, on="key", how="inner", suffixes=("_new", "_old"))

    # preço ref (%)
    denom_p = merged["preco_ref_old"].where(merged["preco_ref_old"].notna() & (merged["preco_ref_old"] != 0), pd.NA)
    merged["pct_change_preco_ref"] = (merged["preco_ref_new"] - merged["preco_ref_old"]) / denom_p * 100.0

    # estoque (queda %)
    merged["delta_estoque"] = merged["estoque_new"] - merged["estoque_old"]
    denom_e = merged["estoque_old"].where(merged["estoque_old"].notna() & (merged["estoque_old"] > 0), pd.NA)
    pct_change_estoque = (merged["estoque_new"] - merged["estoque_old"]) / denom_e * 100.0
    merged["pct_drop_estoque"] = pct_change_estoque.apply(lambda v: (-v) if pd.notna(v) and v < 0 else 0.0)

    # ✅ regra de ataque (MESMO concorrente, comparando com upload anterior)
    merged["flag_ataque"] = (
        (merged["delta_estoque"].fillna(0) < 0) &
        (merged["pct_drop_estoque"].fillna(0) >= stock_drop_pct_min) &
        (merged["estoque_new"].fillna(10**9) <= stock_crit_max) &
        (merged["pct_change_preco_ref"].fillna(-10**9) >= price_up_min)
    )

    out = merged[[
        "key",
        "titulo_new", "marca_new",
        "anuncios_irmaos_old", "anuncios_irmaos_new",
        "estoque_old", "estoque_new", "delta_estoque", "pct_drop_estoque",
        "preco_ref_old", "preco_ref_new", "pct_change_preco_ref",
        "preco_min_old", "preco_min_new",
        "preco_max_old", "preco_max_new",
        "preco_wavg_old", "preco_wavg_new",
        "flag_ataque"
    ]].copy()

    out = out.rename(columns={"titulo_new": "titulo", "marca_new": "marca"})
    return out


def build_score_row(row, stock_crit_max, stock_drop_pct_min, price_up_min):
    pct_up = float(row.get("alta_preco_ref_pct_max")) if pd.notna(row.get("alta_preco_ref_pct_max")) else 0.0
    drop = float(row.get("queda_estoque_pct_max")) if pd.notna(row.get("queda_estoque_pct_max")) else 0.0
    conc = int(row.get("conc_ataque")) if pd.notna(row.get("conc_ataque")) else 0
    est = float(row.get("estoque_atual")) if pd.notna(row.get("estoque_atual")) else stock_crit_max

    s1 = (pct_up / max(price_up_min, 1e-9)) if price_up_min > 0 else 0.0
    s2 = (drop / max(stock_drop_pct_min, 1e-9)) if stock_drop_pct_min > 0 else 0.0
    s3 = 0.6 * max(0, conc - 1)
    s4 = 0.8 * (1.0 - min(est / max(stock_crit_max, 1e-9), 1.0)) if stock_crit_max > 0 else 0.0
    return round(s1 + s2 + s3 + s4, 3)


def style_aggressive(df: pd.DataFrame):
    def row_style(row):
        score = float(row.get("score")) if pd.notna(row.get("score")) else 0.0
        # começa a pintar a partir de ~1.2
        alpha = min(0.75, max(0.0, (score - 1.2) * 0.22))
        if alpha <= 0:
            return [""] * len(row)
        return [f"background-color: rgba(0, 200, 0, {alpha})"] * len(row)
    return df.style.apply(row_style, axis=1)


# =========================
# UI
# =========================
tab_bi, tab_ctrl = st.tabs(["📌 BI (Clean)", "🧾 Controle (histórico)"])

with tab_bi:
    st.subheader("Regras (você controla aqui)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stock_crit_max = st.number_input("Estoque crítico (≤)", 0, 10000, 20, 1)
    with c2:
        stock_drop_pct_min = st.number_input("Queda mínima do estoque (%)", 0.0, 100.0, 25.0, 1.0)
    with c3:
        price_up_min = st.number_input("Alta mínima do preço (%)", 0.0, 200.0, 10.0, 1.0)
    with c4:
        min_concorrentes = st.number_input("Exigir sinal em quantos concorrentes?", 1, 3, 2, 1)
    with c5:
        top_n = st.number_input("Mostrar TOP N", 5, 200, 30, 5)

    st.caption("ATAQUE = estoque caiu forte + ficou crítico + preço (ref = preço mínimo) subiu forte, comparando SEMPRE com o último upload anterior do mesmo concorrente.")

    st.divider()
    st.subheader("Upload (3 arquivos do Nubmetrics)")

    auto_date = st.checkbox("Tentar pegar data automaticamente do nome do arquivo", value=True)
    fallback_date = st.date_input("Se não achar no nome, usa esta data do snapshot", value=date.today())
    fallback_date_str = fallback_date.isoformat()

    u = {}
    cols = st.columns(3)
    for i, comp in enumerate(COMPETITORS):
        with cols[i]:
            u[comp] = st.file_uploader(f"{comp} (Export Nubmetrics)", type=["xlsx", "xls"], key=f"up_{comp}")

    run = st.button("Processar e gerar BI", type="primary")

    if run:
        missing = [c for c, f in u.items() if f is None]
        if missing:
            st.error(f"Faltou upload: {', '.join(missing)}")
            st.stop()

        per_comp = {}
        for comp, f in u.items():
            df_raw = pd.read_excel(f)
            df = normalize_df(df_raw)

            snap = detect_snapshot_date_from_filename(f.name) if auto_date else None
            if not snap:
                snap = fallback_date_str

            # ✅ pega o último snapshot ANTES de salvar o atual
            prev_pack = load_last_snapshot(comp)
            prev_df = prev_pack["df"] if prev_pack else None

            # salva o atual
            save_snapshot(comp, snap, f.name, df)

            if prev_df is None:
                per_comp[comp] = {"prev_exists": False, "diff": pd.DataFrame()}
            else:
                prev_df = normalize_df(prev_df)
                diff = compute_diff(prev_df, df, stock_crit_max, stock_drop_pct_min, price_up_min)
                diff["concorrente"] = comp
                diff["data"] = snap
                per_comp[comp] = {"prev_exists": True, "diff": diff}

        diffs = []
        for comp in COMPETITORS:
            d = per_comp.get(comp)
            if d and d.get("prev_exists"):
                diffs.append(d["diff"][d["diff"]["flag_ataque"] == True].copy())

        st.divider()
        st.subheader("🎯 Produtos para ATACAR (BI CLEAN)")

        if not diffs:
            st.info("Agora você já criou o baseline. Sobe os 3 arquivos de novo (pode ser AGORA mesmo) que ele compara com o upload anterior e mostra o BI.")
            st.stop()

        ataques_all = pd.concat(diffs, ignore_index=True)

        resumo = (
            ataques_all.groupby(["key", "titulo", "marca"], as_index=False)
            .agg(
                conc_ataque=("concorrente", "nunique"),
                concorrentes=("concorrente", lambda x: ", ".join(sorted(set(x)))),
                anuncios_irmaos=("anuncios_irmaos_new", "max"),
                estoque_atual=("estoque_new", "min"),
                queda_estoque_pct_max=("pct_drop_estoque", "max"),
                preco_ref_antes=("preco_ref_old", "min"),
                preco_ref_agora=("preco_ref_new", "max"),
                alta_preco_ref_pct_max=("pct_change_preco_ref", "max"),
                preco_min_agora=("preco_min_new", "min"),
                preco_max_agora=("preco_max_new", "max"),
                preco_wavg_agora=("preco_wavg_new", "max"),
            )
        )

        if min_concorrentes > 1:
            resumo = resumo[resumo["conc_ataque"] >= int(min_concorrentes)]

        if resumo.empty:
            st.warning("Com as regras atuais, não apareceu nada forte em concorrentes suficientes. Abaixa as % ou exige menos concorrentes.")
            st.stop()

        resumo["score"] = resumo.apply(
            lambda r: build_score_row(r, stock_crit_max, stock_drop_pct_min, price_up_min),
            axis=1
        )
        resumo = resumo.sort_values(["score", "conc_ataque", "estoque_atual"], ascending=[False, False, True]).head(int(top_n))

        st.dataframe(style_aggressive(resumo), use_container_width=True, height=560)

        st.download_button(
            "Baixar TOP ATAQUES (CSV)",
            resumo.to_csv(index=False).encode("utf-8"),
            file_name=f"top_ataques_{fallback_date_str}.csv",
            mime="text/csv"
        )

with tab_ctrl:
    st.subheader("Histórico do que já foi enviado")
    snaps = list_snapshots()
    st.dataframe(snaps, use_container_width=True, height=600)

    st.markdown("### Diagnóstico rápido")
    if snaps.empty:
        st.error("Seu banco está vazio. Sobe os 3 arquivos 1x pra criar baseline, depois sobe de novo pra comparar.")
    else:
        st.success(f"Banco OK: {len(snaps)} snapshots salvos.")
        byc = snaps.groupby("competitor").size().reset_index(name="qtd")
        st.dataframe(byc, use_container_width=True)

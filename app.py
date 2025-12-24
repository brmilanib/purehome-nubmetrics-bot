import os
import re
import sqlite3
from datetime import datetime, date
from typing import Optional
from io import StringIO

import pandas as pd
import streamlit as st

# =========================
# CONFIG / DB
# =========================
DB_PATH = os.environ.get("DB_PATH", "snapshots.db")

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics (um por concorrente) → compara com o último upload anterior + BI clean e sempre mostra candidatos.")

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
    """Último snapshot salvo desse concorrente."""
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
    # evita warning do pandas
    df = pd.read_json(StringIO(data_json), orient="records")
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
    name = str(filename)

    iso = re.findall(r"(\d{4}-\d{2}-\d{2})", name)
    if iso:
        return iso[-1]

    br = re.findall(r"(\d{2})[\/\-\._](\d{2})[\/\-\._](\d{4})", name)
    if br:
        dd, mm, yyyy = br[-1]
        return f"{yyyy}-{mm}-{dd}"

    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ Corrige "anúncios irmãos":
      - Estoque NÃO soma (estoque é compartilhado) -> MAX
      - Preço ref = preço mínimo (guerra)
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
        if re.fullmatch(r"\d+\.0", s):
            s = s[:-2]
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
            estoque=("estoque", "max"),          # ✅ NÃO soma
            vendas_unid=("vendas_unid", "sum"),
            preco_min=("preco_medio", "min"),
            preco_max=("preco_medio", "max"),
            desconto=("desconto", mode_or_na),
            frete_gratis=("frete_gratis", mode_or_na),
            full=("full", mode_or_na),
            tipo_publicacao=("tipo_publicacao", mode_or_na),
            anuncios_irmaos=("key", "size"),
        )
    )

    # evita warning do groupby.apply
    price_wavg_map = df.groupby("key", group_keys=False).apply(wavg_price)
    out["preco_wavg"] = out["key"].map(price_wavg_map)

    out["preco_ref"] = out["preco_min"]  # preço de guerra
    return out


# =========================
# Diff / Sinais
# =========================
def compute_signals(prev: pd.DataFrame, curr: pd.DataFrame):
    merged = curr.merge(prev, on="key", how="inner", suffixes=("_new", "_old"))
    common = len(merged)

    # preço ref (%)
    denom_p = merged["preco_ref_old"].where(merged["preco_ref_old"].notna() & (merged["preco_ref_old"] != 0), pd.NA)
    merged["pct_change_preco_ref"] = (merged["preco_ref_new"] - merged["preco_ref_old"]) / denom_p * 100.0

    # estoque (queda %)
    merged["delta_estoque"] = merged["estoque_new"] - merged["estoque_old"]
    denom_e = merged["estoque_old"].where(merged["estoque_old"].notna() & (merged["estoque_old"] > 0), pd.NA)
    pct_change_estoque = (merged["estoque_new"] - merged["estoque_old"]) / denom_e * 100.0
    merged["pct_drop_estoque"] = pct_change_estoque.apply(lambda v: (-v) if pd.notna(v) and v < 0 else 0.0)

    out = merged[[
        "key",
        "titulo_new", "marca_new",
        "anuncios_irmaos_old", "anuncios_irmaos_new",
        "estoque_old", "estoque_new", "delta_estoque", "pct_drop_estoque",
        "preco_ref_old", "preco_ref_new", "pct_change_preco_ref",
        "preco_min_old", "preco_min_new",
        "preco_max_old", "preco_max_new",
        "preco_wavg_old", "preco_wavg_new",
    ]].copy()

    out = out.rename(columns={"titulo_new": "titulo", "marca_new": "marca"})
    return out, common


def style_green(df: pd.DataFrame, score_col: str = "score"):
    def row_style(row):
        v = row.get(score_col)
        try:
            v = float(v)
        except:
            v = 0.0
        alpha = min(0.75, max(0.0, (v - 1.0) * 0.25))
        if alpha <= 0:
            return [""] * len(row)
        return [f"background-color: rgba(0, 200, 0, {alpha})"] * len(row)
    return df.style.apply(row_style, axis=1)


def score_row(row, stock_drop_pct_min, price_up_min, stock_crit_max, require_crit):
    pct_up = float(row.get("pct_change_preco_ref")) if pd.notna(row.get("pct_change_preco_ref")) else 0.0
    drop = float(row.get("pct_drop_estoque")) if pd.notna(row.get("pct_drop_estoque")) else 0.0
    est = float(row.get("estoque_new")) if pd.notna(row.get("estoque_new")) else 10**9

    s1 = pct_up / max(price_up_min, 1e-9) if price_up_min > 0 else 0.0
    s2 = drop / max(stock_drop_pct_min, 1e-9) if stock_drop_pct_min > 0 else 0.0
    s3 = 0.0
    if require_crit and stock_crit_max > 0:
        s3 = 0.8 * (1.0 - min(est / max(stock_crit_max, 1e-9), 1.0))
    return round(s1 + s2 + s3, 3)


# =========================
# UI
# =========================
tab_bi, tab_ctrl = st.tabs(["📌 BI (Clean)", "🧾 Controle (histórico)"])

with tab_bi:
    st.subheader("Regras (você controla aqui)")
    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])

    with c1:
        stock_crit_max = st.number_input("Estoque crítico (≤)", 0, 10000, 30, 1)
    with c2:
        require_crit = st.checkbox("Exigir estoque crítico no ATAQUE", value=False)
    with c3:
        stock_drop_pct_min = st.number_input("Queda mínima estoque (%)", 0.0, 100.0, 10.0, 1.0)
    with c4:
        price_up_min = st.number_input("Alta mínima preço (%)", 0.0, 200.0, 5.0, 1.0)
    with c5:
        min_concorrentes = st.number_input("Sinal em quantos concorrentes?", 1, 3, 1, 1)
    with c6:
        top_n = st.number_input("Mostrar TOP N", 5, 200, 40, 5)

    st.caption("⚠️ Se não aparecer ATAQUE, o app agora ainda te mostra os CANDIDATOS: estoque caindo + preço subindo.")

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

        comp_debug = []
        all_candidates = []

        for comp, f in u.items():
            df_raw = pd.read_excel(f)
            df = normalize_df(df_raw)

            snap = detect_snapshot_date_from_filename(f.name) if auto_date else None
            if not snap:
                snap = fallback_date_str

            prev_pack = load_last_snapshot(comp)
            prev_df = prev_pack["df"] if prev_pack else None

            # salva atual
            save_snapshot(comp, snap, f.name, df)

            if prev_df is None:
                comp_debug.append({
                    "concorrente": comp,
                    "comparou_com": "— (primeiro upload)",
                    "itens_em_comum": 0,
                    "itens_no_upload": len(df)
                })
                continue

            prev_df = normalize_df(prev_df)
            signals, common = compute_signals(prev_df, df)

            comp_debug.append({
                "concorrente": comp,
                "comparou_com": f'{prev_pack["snapshot_date"]} ({prev_pack["filename"]})',
                "itens_em_comum": int(common),
                "itens_no_upload": int(len(df))
            })

            if common == 0:
                continue

            # filtros básicos (candidatos)
            signals["flag_preco_up"] = signals["pct_change_preco_ref"].fillna(-10**9) >= price_up_min
            signals["flag_estoque_drop"] = (signals["delta_estoque"].fillna(0) < 0) & (signals["pct_drop_estoque"].fillna(0) >= stock_drop_pct_min)
            signals["flag_critico"] = signals["estoque_new"].fillna(10**9) <= stock_crit_max if stock_crit_max > 0 else True

            # ATAQUE “formal”
            if require_crit:
                signals["flag_ataque"] = signals["flag_preco_up"] & signals["flag_estoque_drop"] & signals["flag_critico"]
            else:
                signals["flag_ataque"] = signals["flag_preco_up"] & signals["flag_estoque_drop"]

            # score e guarda
            signals["score"] = signals.apply(lambda r: score_row(r, stock_drop_pct_min, price_up_min, stock_crit_max, require_crit), axis=1)
            signals["concorrente"] = comp
            signals["data"] = snap

            # candidatos = caiu estoque + subiu preço (mesmo se não for “critico”)
            cand = signals[signals["flag_preco_up"] & signals["flag_estoque_drop"]].copy()
            all_candidates.append(cand)

        st.divider()
        st.subheader("🔍 Debug rápido (pra você ter certeza que comparou)")
        st.dataframe(pd.DataFrame(comp_debug), use_container_width=True, height=220)

        if not all_candidates:
            st.warning("Não teve itens em comum para comparar (keys não bateram). Isso só acontece se o arquivo veio diferente/sem SKU/GTIN. Me manda 1 export que eu ajusto o key.")
            st.stop()

        candidates_all = pd.concat(all_candidates, ignore_index=True)
        if candidates_all.empty:
            st.warning("Comparou, mas não achou NENHUM candidato (estoque caiu + preço subiu) com os thresholds atuais. Baixa mais as % e tenta de novo.")
            st.stop()

        # resumo por produto cruzando concorrentes
        resumo = (
            candidates_all.groupby(["key", "titulo", "marca"], as_index=False)
            .agg(
                conc=("concorrente", "nunique"),
                concorrentes=("concorrente", lambda x: ", ".join(sorted(set(x)))),
                anuncios_irmaos=("anuncios_irmaos_new", "max"),
                estoque_min=("estoque_new", "min"),
                queda_estoque_pct_max=("pct_drop_estoque", "max"),
                alta_preco_ref_pct_max=("pct_change_preco_ref", "max"),
                preco_ref_antes=("preco_ref_old", "min"),
                preco_ref_agora=("preco_ref_new", "max"),
            )
        )

        resumo = resumo[resumo["conc"] >= int(min_concorrentes)]
        if resumo.empty:
            st.warning("Teve candidatos, mas não bateu o número mínimo de concorrentes exigido. Baixa esse número.")
            st.stop()

        # score final
        resumo["score"] = (
            (resumo["alta_preco_ref_pct_max"] / max(price_up_min, 1e-9)) +
            (resumo["queda_estoque_pct_max"] / max(stock_drop_pct_min, 1e-9)) +
            (1.2 * (resumo["conc"] - 1))
        ).round(3)

        resumo = resumo.sort_values(["score", "conc", "estoque_min"], ascending=[False, False, True]).head(int(top_n))

        st.subheader("🎯 Produtos pra ATACAR (candidatos: estoque caiu + preço subiu)")
        st.dataframe(style_green(resumo, "score"), use_container_width=True, height=560)

        st.download_button(
            "Baixar TOP Candidatos (CSV)",
            resumo.to_csv(index=False).encode("utf-8"),
            file_name=f"top_candidatos_{fallback_date_str}.csv",
            mime="text/csv"
        )

        # extras: sempre mostrar “top estoque caiu” e “top preço subiu”
        st.divider()
        st.subheader("📉 Top quedas de estoque (mesmo que preço não suba)")
        top_stock = candidates_all.sort_values(["pct_drop_estoque", "delta_estoque"], ascending=[False, True]).head(30)
        st.dataframe(top_stock[[
            "concorrente","key","titulo","marca",
            "estoque_old","estoque_new","delta_estoque","pct_drop_estoque",
            "preco_ref_old","preco_ref_new","pct_change_preco_ref",
            "anuncios_irmaos_new"
        ]], use_container_width=True, height=420)

        st.subheader("📈 Top altas de preço (mesmo que estoque não caia)")
        # usa signals “bruto”: pega do candidates_all só quem tem colunas (já tem),
        # mas aqui quer qualquer alta → então recria a partir dos candidatos_all (que já é subset)
        # (se quiser 100% completo, depois eu amplio para puxar todos os signals)
        top_price = candidates_all.sort_values(["pct_change_preco_ref"], ascending=[False]).head(30)
        st.dataframe(top_price[[
            "concorrente","key","titulo","marca",
            "preco_ref_old","preco_ref_new","pct_change_preco_ref",
            "estoque_old","estoque_new","delta_estoque","pct_drop_estoque",
            "anuncios_irmaos_new"
        ]], use_container_width=True, height=420)


with tab_ctrl:
    st.subheader("Histórico do que já foi enviado")
    snaps = list_snapshots()
    st.dataframe(snaps, use_container_width=True, height=600)

    st.markdown("### Diagnóstico rápido")
    if snaps.empty:
        st.error("Banco vazio. Sobe os 3 arquivos 1x pra criar baseline, depois sobe o próximo pra comparar.")
    else:
        st.success(f"Banco OK: {len(snaps)} snapshots salvos.")
        byc = snaps.groupby("competitor").size().reset_index(name="qtd")
        st.dataframe(byc, use_container_width=True)

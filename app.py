import os
import re
import io
import sqlite3
from datetime import datetime, date

import pandas as pd
import streamlit as st

# ======================
# CONFIG / DB
# ======================
DB_PATH = os.environ.get("DB_PATH", "/tmp/purehome_snapshots.db")

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

CANON_MAP = {
    "título": "titulo",
    "titulo": "titulo",
    "marca": "marca",
    "vendas em $": "vendas_valor",
    "vendas em unid.": "vendas_unid",
    "vendas em unid": "vendas_unid",
    "preço médio": "preco_medio",
    "preco médio": "preco_medio",
    "tipo de publicação": "tipo_publicacao",
    "fulfillment": "full",
    "catálogo.": "catalogo",
    "catálogo": "catalogo",
    "com frete grátis": "frete_gratis",
    "com frete gratis": "frete_gratis",
    "com mercado envios": "mercado_envios",
    "com desconto": "desconto",
    "sku": "sku",
    "oem": "oem",
    "gtin": "gtin",
    "n° peça": "n_peca",
    "n° peca": "n_peca",
    "nº peça": "n_peca",
    "nº peca": "n_peca",
    "estado": "estado",
    "mercadopago": "mercadopago",
    "republicada": "republicada",
    "condição": "condicao",
    "condicao": "condicao",
    "estoque": "estoque",
}

# ======================
# DB helpers
# ======================
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


def save_snapshot(competitor: str, snapshot_date: str, filename: str, df_norm: pd.DataFrame):
    payload = df_norm.to_json(orient="records", force_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO snapshots (competitor, snapshot_date, uploaded_at, filename, data_json) VALUES (?, ?, ?, ?, ?)",
            (competitor, snapshot_date, datetime.now().isoformat(timespec="seconds"), filename, payload)
        )


def load_last_snapshot_before(competitor: str, snapshot_date: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT snapshot_date, uploaded_at, filename, data_json
            FROM snapshots
            WHERE competitor = ? AND snapshot_date < ?
            ORDER BY snapshot_date DESC, id DESC
            LIMIT 1
            """,
            (competitor, snapshot_date)
        ).fetchone()
    if not row:
        return None
    d, uploaded_at, filename, data_json = row
    df = pd.read_json(io.StringIO(data_json), orient="records")
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}


def list_snapshots(limit: int = 200):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT competitor, snapshot_date, uploaded_at, filename, LENGTH(data_json) as bytes
            FROM snapshots
            ORDER BY snapshot_date DESC, id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    return pd.DataFrame(rows, columns=["competitor", "snapshot_date", "uploaded_at", "filename", "bytes_json"])


def wipe_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM snapshots")
        conn.commit()

# ======================
# Utils
# ======================
def parse_date_from_filename(name: str):
    if not name:
        return None

    s = name.lower()

    # YYYY-MM-DD
    m = re.search(r"(20\d{2})[-_\.](\d{2})[-_\.](\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    # YYYYMMDD
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    # DD-MM-YYYY
    m = re.search(r"(\d{2})[-_\.](\d{2})[-_\.](20\d{2})", s)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    return None


def to_bool_sim_na(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"sim", "yes", "true", "1"}


def safe_pct_change(new, old):
    try:
        if pd.isna(new) or pd.isna(old):
            return pd.NA
        old = float(old)
        new = float(new)
        if old == 0:
            return pd.NA
        return (new - old) / old * 100.0
    except Exception:
        return pd.NA


def safe_pct_drop(old, new):
    """% de queda: (old - new)/old*100 (positivo quando cai)"""
    try:
        if pd.isna(new) or pd.isna(old):
            return pd.NA
        old = float(old)
        new = float(new)
        if old == 0:
            return pd.NA
        return (old - new) / old * 100.0
    except Exception:
        return pd.NA


def fmt_brl_money(v):
    if pd.isna(v):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    # 2 casas, mas remove zeros desnecessários
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # pt-BR
    if s.endswith(",00"):
        s = s[:-3]
    return f"R$ {s}"


def fmt_pct_signed(v, force_minus=False):
    """
    v em número (ex 8.24) => +8,24%
    se force_minus=True => -8,24% (para quedas)
    """
    if pd.isna(v):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    sign = "-" if force_minus else ("+" if v > 0 else ("-" if v < 0 else ""))
    vv = abs(v) if force_minus else abs(v)
    s = f"{vv:.2f}".replace(".", ",")
    if s.endswith(",00"):
        s = s[:-3]
    return f"{sign}{s}%"


def fmt_int(v):
    if pd.isna(v):
        return ""
    try:
        return str(int(round(float(v))))
    except Exception:
        return ""

# ======================
# Normalize + agregação por SKU (anúncios irmãos)
# ======================
def normalize_and_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: CANON_MAP.get(c, c) for c in df.columns})

    for c in ["sku", "gtin", "n_peca", "titulo", "marca", "preco_medio", "estoque", "vendas_unid",
              "desconto", "frete_gratis", "full", "tipo_publicacao"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")

    def make_key(row):
        for k in ["sku", "gtin", "n_peca"]:
            v = row.get(k)
            if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() != "nan":
                return str(v).strip()
        t = row.get("titulo")
        if pd.notna(t) and str(t).strip() != "":
            return "TIT_" + str(t).strip()[:80]
        return None

    df["key"] = df.apply(make_key, axis=1)
    df = df[df["key"].notna()].copy()

    for b in ["desconto", "frete_gratis", "full"]:
        df[b] = df[b].apply(to_bool_sim_na)

    def pick_mode_str(series: pd.Series):
        s = series.dropna().astype(str).str.strip()
        if s.empty:
            return pd.NA
        return s.mode().iloc[0]

    # IMPORTANTE:
    # - NÃO somar estoque (anúncios irmãos compartilham estoque)
    # - preço referência = preço MIN (mais barato)
    agg = df.groupby("key", as_index=False).agg(
        titulo=("titulo", lambda s: s.dropna().iloc[0] if len(s.dropna()) else pd.NA),
        marca=("marca", lambda s: s.dropna().iloc[0] if len(s.dropna()) else pd.NA),
        preco_ref=("preco_medio", "min"),
        estoque_ref=("estoque", "max"),
        vendas_unid=("vendas_unid", "sum"),
        desconto=("desconto", "max"),
        frete_gratis=("frete_gratis", "max"),
        full=("full", "max"),
        tipo_publicacao=("tipo_publicacao", pick_mode_str),
        anuncios_irmaos=("key", "size"),
    )

    cheap_title = (
        df.sort_values(["key", "preco_medio"], ascending=[True, True], na_position="last")
          .drop_duplicates("key")[["key", "titulo"]]
          .rename(columns={"titulo": "titulo_barato"})
    )
    agg = agg.merge(cheap_title, on="key", how="left")
    agg["titulo"] = agg["titulo_barato"].combine_first(agg["titulo"])
    agg = agg.drop(columns=["titulo_barato"])

    agg = agg.sort_values(["marca", "titulo"], na_position="last")
    return agg

# ======================
# Comparação (D-1 -> D0)
# ======================
def compare(prev_norm: pd.DataFrame, curr_norm: pd.DataFrame):
    merged = curr_norm.merge(prev_norm, on="key", how="inner", suffixes=("_new", "_old"))

    merged["preco_pct"] = merged.apply(
        lambda r: safe_pct_change(r["preco_ref_new"], r["preco_ref_old"]), axis=1
    )
    merged["estoque_drop_pct"] = merged.apply(
        lambda r: safe_pct_drop(r["estoque_ref_old"], r["estoque_ref_new"]), axis=1
    )

    out = merged[[
        "key",
        "titulo_new", "marca_new",
        "anuncios_irmaos_new",
        "preco_ref_old", "preco_ref_new", "preco_pct",
        "estoque_ref_old", "estoque_ref_new", "estoque_drop_pct",
    ]].rename(columns={
        "titulo_new": "titulo",
        "marca_new": "marca",
        "anuncios_irmaos_new": "anuncios_irmaos",
        "preco_ref_old": "preco_old",
        "preco_ref_new": "preco_new",
        "estoque_ref_old": "estoque_old",
        "estoque_ref_new": "estoque_new",
    })

    return out, len(merged), len(curr_norm), len(prev_norm)

# ======================
# UI
# ======================
st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics (um por concorrente) → compara com o último upload anterior + BI clean de ATAQUE.")

tab_bi, tab_hist = st.tabs(["📌 BI (Clean)", "🗂️ Controle (histórico)"])

with tab_bi:
    st.markdown("### Regras (você controla aqui)")
    r1, r2, r3, r4, r5 = st.columns([1.2, 1.6, 1.6, 1.6, 1.2])

    with r1:
        estoque_critico = st.number_input("Estoque crítico (≤)", min_value=0, max_value=9999, value=30, step=1)
        exigir_critico = st.checkbox("Exigir estoque crítico no ATAQUE", value=True)
    with r2:
        queda_min_estoque_pct = st.number_input("Queda mínima de estoque (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with r3:
        alta_min_preco_pct = st.number_input("Alta mínima do preço (%)", min_value=0.0, max_value=300.0, value=5.0, step=1.0)
    with r4:
        min_concorrentes = st.number_input("Sinal em quantos concorrentes?", min_value=1, max_value=3, value=1, step=1)
    with r5:
        top_n = st.number_input("Mostrar TOP N", min_value=5, max_value=300, value=40, step=5)

    st.caption(
        "ATAQUE = estoque caiu forte + (opcional) ficou crítico + preço (ref = preço mínimo entre anúncios irmãos) subiu forte, "
        "comparando SEMPRE com o último upload anterior do MESMO concorrente."
    )

    st.divider()
    st.markdown("### Upload (3 arquivos do Nubmetrics)")
    auto_date = st.checkbox("Tentar pegar a data automaticamente do nome do arquivo", value=True)
    snap_date_manual = st.date_input("Se não achar no nome, usa esta data do snapshot", value=date.today())
    snap_date_str_default = snap_date_manual.isoformat()

    ucols = st.columns(3)
    uploads = {}
    for i, comp in enumerate(COMPETITORS):
        with ucols[i]:
            uploads[comp] = st.file_uploader(f"{comp} (Export Nubmetrics)", type=["xlsx", "xls"], key=f"up_{comp}")

    process = st.button("Processar e gerar BI", type="primary")

    st.divider()

    if process:
        missing = [c for c, f in uploads.items() if f is None]
        if missing:
            st.error(f"Faltou upload: {', '.join(missing)}")
            st.stop()

        per_comp = {}
        debug_rows = []

        for comp, f in uploads.items():
            snap_from_name = parse_date_from_filename(getattr(f, "name", "")) if auto_date else None
            snap_date_str = snap_from_name or snap_date_str_default

            raw = pd.read_excel(f)
            curr_norm = normalize_and_aggregate(raw)

            prev_pack = load_last_snapshot_before(comp, snap_date_str)

            # salva o atual (já agregado)
            save_snapshot(comp, snap_date_str, f.name, curr_norm)

            if not prev_pack:
                per_comp[comp] = {"prev": None, "curr": curr_norm, "cmp": None, "common": 0}
                debug_rows.append({
                    "Concorrente": comp,
                    "Comparou com": "— (primeiro upload)",
                    "Itens em comum": 0,
                    "Itens no upload": len(curr_norm)
                })
                continue

            prev_norm = prev_pack["df"]  # BUG FIX: não re-normaliza
            cmp_df, common, n_curr, _n_prev = compare(prev_norm, curr_norm)

            per_comp[comp] = {
                "prev": prev_norm,
                "curr": curr_norm,
                "cmp": cmp_df,
                "common": common,
                "prev_date": prev_pack["snapshot_date"],
                "prev_file": prev_pack["filename"],
            }

            debug_rows.append({
                "Concorrente": comp,
                "Comparou com": f'{prev_pack["snapshot_date"]} ({prev_pack["filename"]})',
                "Itens em comum": common,
                "Itens no upload": n_curr
            })

        st.markdown("#### 🔎 Debug rápido (pra você ter certeza que comparou)")
        dbg = pd.DataFrame(debug_rows)
        st.dataframe(dbg, use_container_width=True, height=160)

        # BI CLEAN
        frames = []
        for comp, pack in per_comp.items():
            if pack["cmp"] is None or pack["common"] == 0:
                continue

            d = pack["cmp"].copy()
            d["concorrente"] = comp

            d["flag_estoque_critico"] = d["estoque_new"].fillna(10**9) <= float(estoque_critico)
            d["flag_queda_estoque"] = d["estoque_drop_pct"].fillna(-10**9) >= float(queda_min_estoque_pct)
            d["flag_alta_preco"] = d["preco_pct"].fillna(-10**9) >= float(alta_min_preco_pct)

            if exigir_critico:
                d["flag_ataque"] = d["flag_estoque_critico"] & d["flag_queda_estoque"] & d["flag_alta_preco"]
            else:
                d["flag_ataque"] = d["flag_queda_estoque"] & d["flag_alta_preco"]

            frames.append(d)

        if not frames:
            st.info("Ainda não deu para comparar (provável 1º upload de todos). No próximo upload já aparece.")
            st.stop()

        all_cmp = pd.concat(frames, ignore_index=True)

        # Estoque por concorrente (pra você bater o olho)
        stock_by_comp = (
            all_cmp.groupby(["key", "concorrente"], as_index=False)
                  .agg(estoque=("estoque_new", "min"))
        )
        stock_pivot = stock_by_comp.pivot(index="key", columns="concorrente", values="estoque").reset_index()

        def join_stocks(row):
            parts = []
            for c in COMPETITORS:
                v = row.get(c)
                if pd.isna(v):
                    parts.append(f"{c}:-")
                else:
                    parts.append(f"{c}:{fmt_int(v)}")
            return " | ".join(parts)

        stock_pivot["estoques_concorrentes"] = stock_pivot.apply(join_stocks, axis=1)
        stock_pivot = stock_pivot[["key", "estoques_concorrentes"]]

        g = (
            all_cmp.groupby("key", as_index=False)
                  .agg(
                      titulo=("titulo", "first"),
                      marca=("marca", "first"),
                      concorrentes_com_sinal=("flag_ataque", "sum"),
                      estoque_atual_min=("estoque_new", "min"),
                      queda_estoque_pct_max=("estoque_drop_pct", "max"),
                      preco_anterior_min=("preco_old", "min"),
                      preco_atual_min=("preco_new", "min"),
                      variacao_preco_pct_max=("preco_pct", "max"),
                      anuncios_irmaos_max=("anuncios_irmaos", "max"),
                  )
        )

        g = g.merge(stock_pivot, on="key", how="left")

        atacar = g[g["concorrentes_com_sinal"] >= int(min_concorrentes)].copy()

        if atacar.empty:
            st.warning("Nada bateu ATAQUE com as regras atuais. Vou te mostrar CANDIDATOS (estoque caindo + preço subindo).")
            atacar = g[
                (g["queda_estoque_pct_max"].fillna(-10**9) >= float(queda_min_estoque_pct)) &
                (g["variacao_preco_pct_max"].fillna(-10**9) >= float(alta_min_preco_pct))
            ].copy()

        atacar["score"] = (
            atacar["concorrentes_com_sinal"].fillna(0) * 100
            + atacar["queda_estoque_pct_max"].fillna(0) * 2
            + atacar["variacao_preco_pct_max"].fillna(0) * 1
            - atacar["estoque_atual_min"].fillna(0) * 0.5
        )

        atacar = (
            atacar.sort_values(["score"], ascending=False)
                  .head(int(top_n))
                  .drop(columns=["score"])
        )

        # ======================
        # APRESENTAÇÃO CLEAN
        # ======================
        st.markdown("### 🎯 Produtos para ATACAR (BI CLEAN)")

        show = atacar[[
            "key",
            "titulo",
            "marca",
            "concorrentes_com_sinal",
            "estoque_atual_min",
            "estoques_concorrentes",
            "queda_estoque_pct_max",
            "preco_anterior_min",
            "preco_atual_min",
            "variacao_preco_pct_max",
            "anuncios_irmaos_max",
        ]].copy()

        # renomeia colunas (claras)
        col_rename = {
            "key": "SKU / ID",
            "titulo": "Produto",
            "marca": "Marca",
            "concorrentes_com_sinal": "Concorrentes c/ sinal",
            "estoque_atual_min": "Estoque atual (mín)",
            "estoques_concorrentes": "Estoques por concorrente",
            "queda_estoque_pct_max": "Queda estoque",
            "preco_anterior_min": "Preço anterior",
            "preco_atual_min": "Preço atual",
            "variacao_preco_pct_max": "Variação preço",
            "anuncios_irmaos_max": "Anúncios irmãos",
        }
        show = show.rename(columns=col_rename)

        # formata textos (percent e R$) mantendo colunas numéricas separadas pra estilo
        # vamos criar colunas "display" e manter as originais só pra style
        show["_drop_num"] = atacar["queda_estoque_pct_max"].values
        show["_price_num"] = atacar["variacao_preco_pct_max"].values

        show["Queda estoque"] = show["_drop_num"].apply(lambda v: fmt_pct_signed(v, force_minus=True))
        show["Variação preço"] = show["_price_num"].apply(lambda v: fmt_pct_signed(v, force_minus=False))

        show["Preço anterior"] = show["Preço anterior"].apply(fmt_brl_money)
        show["Preço atual"] = show["Preço atual"].apply(fmt_brl_money)

        show["Estoque atual (mín)"] = show["Estoque atual (mín)"].apply(fmt_int)
        show["Concorrentes c/ sinal"] = show["Concorrentes c/ sinal"].apply(fmt_int)
        show["Anúncios irmãos"] = show["Anúncios irmãos"].apply(fmt_int)

        # === estilos: queda vermelho negrito, alta verde negrito
        def style_pct(row):
            styles = [""] * len(row)
            cols = list(row.index)

            # queda (vermelho)
            if "Queda estoque" in cols:
                i = cols.index("Queda estoque")
                if pd.notna(row.get("_drop_num")) and row.get("_drop_num") > 0:
                    styles[i] = "color:#b00020;font-weight:700;"
            # variação preço (verde ou vermelho)
            if "Variação preço" in cols:
                i = cols.index("Variação preço")
                v = row.get("_price_num")
                if pd.isna(v):
                    pass
                else:
                    if float(v) > 0:
                        styles[i] = "color:#0a7a0a;font-weight:700;"
                    elif float(v) < 0:
                        styles[i] = "color:#b00020;font-weight:700;"
            return styles

        def highlight_strong(row):
            # verde clarinho nas mudanças mais agressivas
            # (top 25% de queda de estoque + top 25% de alta de preço)
            if len(show) < 4:
                return [""] * len(row)

            q_drop = float(atacar["queda_estoque_pct_max"].quantile(0.75))
            q_price = float(atacar["variacao_preco_pct_max"].quantile(0.75))

            strong = (
                pd.notna(row.get("_drop_num")) and float(row.get("_drop_num")) >= q_drop
                and pd.notna(row.get("_price_num")) and float(row.get("_price_num")) >= q_price
            )
            if strong:
                return ["background-color:#d8f5d8;"] * len(row)
            return [""] * len(row)

        # remove colunas auxiliares da visão final, mas mantém pra style via row.get
        # (o pandas styler ainda "vê" elas se estiverem no df; então deixamos e escondemos)
        styled = (
            show.style
                .apply(highlight_strong, axis=1)
                .apply(style_pct, axis=1)
                .hide(axis="columns", subset=["_drop_num", "_price_num"])
        )

        st.dataframe(styled, use_container_width=True, height=520)

        st.download_button(
            "Baixar BI (CSV)",
            show.drop(columns=["_drop_num", "_price_num"]).to_csv(index=False).encode("utf-8"),
            file_name=f"bi_clean_{snap_date_str_default}.csv",
            mime="text/csv"
        )

with tab_hist:
    st.markdown("### Histórico de uploads")
    hist = list_snapshots(limit=200)
    if hist.empty:
        st.info("Ainda não tem histórico salvo.")
    else:
        st.dataframe(hist, use_container_width=True, height=380)

    st.divider()
    st.markdown("### Reset (se bagunçar ou quiser começar do zero)")
    colx, coly = st.columns([1, 2])
    with colx:
        confirm = st.checkbox("Confirmo que quero APAGAR tudo", value=False)
    with coly:
        if st.button("🧨 Resetar histórico (apagar banco)", type="secondary", disabled=not confirm):
            wipe_db()
            st.success("Apagado. Agora é começar o histórico de novo com seus uploads.")

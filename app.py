import os
import re
import json
import sqlite3
from datetime import datetime, date

import pandas as pd
import streamlit as st

# =========================
# CONFIG / DB (NÃO USE /tmp)
# =========================
DB_PATH = os.environ.get("DB_PATH", "data/snapshots.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

# Mapeamento Nubmetrics -> canônico
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
    "n° peça ": "n_peca",
    "estado": "estado",
    "mercadopago": "mercadopago",
    "republicada": "republicada",
    "condição": "condicao",
    "estoque": "estoque",
}

WATCH_COLS = [
    "preco_medio",
    "estoque",
    "desconto",
    "frete_gratis",
    "full",
    "tipo_publicacao",
    "vendas_unid",
]


# =========================
# DB
# =========================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competitor TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
            """
        )


def save_snapshot(competitor: str, snapshot_date: str, filename: str, df_key: pd.DataFrame):
    payload = json.dumps(df_key.to_dict(orient="records"), ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (competitor, snapshot_date, uploaded_at, filename, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (competitor, snapshot_date, datetime.now().isoformat(timespec="seconds"), filename, payload),
        )


def load_latest_snapshot(competitor: str):
    """✅ SEMPRE pega o último upload (independente de data)"""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT snapshot_date, uploaded_at, filename, data_json, id
            FROM snapshots
            WHERE competitor = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (competitor,),
        ).fetchone()

    if not row:
        return None

    d, uploaded_at, filename, data_json, sid = row
    df = pd.DataFrame(json.loads(data_json))
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df, "id": sid}


def list_snapshots(limit=300):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT competitor, snapshot_date, uploaded_at, filename, id
            FROM snapshots
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame(rows, columns=["Concorrente", "Data snapshot", "Upload em", "Arquivo", "id"])


# =========================
# UTIL
# =========================
def try_extract_date_from_filename(name: str):
    if not name:
        return None

    # yyyy-mm-dd
    m = re.search(r"(20\d{2})[-_\.](\d{2})[-_\.](\d{2})", name)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    # dd-mm-yyyy
    m = re.search(r"(\d{2})[-_\.](\d{2})[-_\.](20\d{2})", name)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: CANON_MAP.get(c, c) for c in df.columns})

    needed = WATCH_COLS + ["sku", "gtin", "n_peca", "titulo", "marca"]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")

    for c in ["sku", "gtin", "n_peca"]:
        df[c] = df[c].astype(str).replace({"nan": "", "None": ""}).str.strip()

    df["titulo"] = df["titulo"].astype(str).replace({"nan": "", "None": ""}).str.strip()
    df["marca"] = df["marca"].astype(str).replace({"nan": "", "None": ""}).str.strip()

    def make_key(row):
        for k in ["sku", "gtin", "n_peca"]:
            v = row.get(k, "")
            if v and v.strip():
                return v.strip()
        if row.get("titulo", ""):
            return "TIT_" + row["titulo"][:80]
        return None

    df["key"] = df.apply(make_key, axis=1)
    df = df[df["key"].notna()].copy()
    df["key"] = df["key"].astype(str)

    return df


def aggregate_per_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ Anúncios irmãos:
    - NÃO soma estoque (usa MAX como estoque “compartilhado”)
    - preço usa MIN (captura anúncio mais agressivo)
    - cria contagem anúncios irmãos
    """
    return df.groupby("key", as_index=False).agg(
        titulo=("titulo", "first"),
        marca=("marca", "first"),
        preco_min=("preco_medio", "min"),
        preco_max=("preco_medio", "max"),
        preco_avg=("preco_medio", "mean"),
        estoque_shared=("estoque", "max"),
        anuncios_irmaos=("key", "size"),
    )


def pct_drop(old, new):
    if pd.isna(old) or old is None or old <= 0:
        return pd.NA
    if pd.isna(new) or new is None:
        return pd.NA
    return (old - new) / old * 100.0


def pct_change(old, new):
    if pd.isna(old) or old is None or old <= 0:
        return pd.NA
    if pd.isna(new) or new is None:
        return pd.NA
    return (new - old) / old * 100.0


def fmt_price(v):
    if pd.isna(v):
        return "—"
    try:
        s = f"{float(v):.2f}".rstrip("0").rstrip(".")
        s = s.replace(".", ",")
        return f"R$ {s}"
    except Exception:
        return "—"


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    try:
        x = float(v)
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.2f}%".replace(".", ",")
    except Exception:
        return "—"


def compute_prev_curr(prev_key: pd.DataFrame, curr_key: pd.DataFrame) -> pd.DataFrame:
    m = curr_key.merge(prev_key, on="key", how="inner", suffixes=("_new", "_old"))

    m["drop_estoque_pct"] = m.apply(
        lambda r: pct_drop(r.get("estoque_shared_old"), r.get("estoque_shared_new")), axis=1
    )
    m["preco_change_pct"] = m.apply(
        lambda r: pct_change(r.get("preco_min_old"), r.get("preco_min_new")), axis=1
    )
    return m


# =========================
# APP
# =========================
init_db()

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics → compara SEMPRE com o último upload do mesmo concorrente + BI clean.")

tab_bi, tab_precos, tab_ctrl = st.tabs(["📌 BI (Clean)", "💸 Alterações de Preço (Clean)", "🧾 Controle (histórico)"])

# -------------------------
# BI
# -------------------------
with tab_bi:
    st.markdown("### Regras (você controla aqui)")

    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

    with c1:
        stock_crit = st.number_input("Estoque crítico (≤)", min_value=0, max_value=99999, value=30, step=1)
    with c2:
        require_stock_crit = st.checkbox("Exigir estoque crítico no ATAQUE", value=False)
    with c3:
        min_drop_pct = st.number_input("Queda mínima de estoque (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with c4:
        min_price_up_pct = st.number_input("Alta mínima de preço (%)", min_value=0.0, max_value=999.0, value=5.0, step=1.0)
    with c5:
        need_comp = st.number_input("Sinal em quantos concorrentes?", min_value=1, max_value=3, value=1, step=1)
    with c6:
        top_n = st.number_input("Mostrar TOP N", min_value=5, max_value=500, value=40, step=5)

    st.divider()
    st.markdown("### Upload (3 arquivos do Nubmetrics)")

    auto_date_from_name = st.checkbox("Tentar pegar data automaticamente do nome do arquivo", value=True)

    snap_date = st.date_input("Se não achar no nome, usa esta data do snapshot", value=date.today())
    snap_date_str = snap_date.isoformat()

    u1, u2, u3 = st.columns(3)
    uploaded_files = {}

    with u1:
        uploaded_files["AUMA"] = st.file_uploader("AUMA (Export Nubmetrics)", type=["xlsx", "xls"], key="up_AUMA")
    with u2:
        uploaded_files["BAGATELLE"] = st.file_uploader("BAGATELLE (Export Nubmetrics)", type=["xlsx", "xls"], key="up_BAGATELLE")
    with u3:
        uploaded_files["PERFUMES_BHZ"] = st.file_uploader("PERFUMES_BHZ (Export Nubmetrics)", type=["xlsx", "xls"], key="up_PERFUMES_BHZ")

    process = st.button("Processar e gerar BI", type="primary")

    st.divider()

    if process:
        missing = [c for c, f in uploaded_files.items() if f is None]
        if missing:
            st.error(f"Faltou upload: {', '.join(missing)}")
            st.stop()

        per_comp = {}
        debug_rows = []
        stock_map_all = []

        # ✅ pega PREV = último upload, e só depois salva o atual
        for comp, f in uploaded_files.items():
            df_raw = pd.read_excel(f)
            df_norm = normalize_df(df_raw)
            curr_key = aggregate_per_key(df_norm)

            effective_date = snap_date_str
            if auto_date_from_name:
                dname = try_extract_date_from_filename(f.name)
                if dname:
                    effective_date = dname

            prev = load_latest_snapshot(comp)  # ✅ sempre o último

            # salva snapshot atual
            save_snapshot(comp, effective_date, f.name, curr_key)

            # estoque atual por concorrente
            stock_tmp = curr_key[["key", "estoque_shared", "anuncios_irmaos"]].copy()
            stock_tmp["concorrente"] = comp
            stock_map_all.append(stock_tmp)

            if prev is None:
                per_comp[comp] = {"has_prev": False, "prev": None, "curr_key": curr_key, "comp_df": pd.DataFrame()}
                debug_rows.append([comp, "— (primeiro upload)", 0, len(curr_key)])
            else:
                prev_key = prev["df"]
                comp_df = compute_prev_curr(prev_key, curr_key)
                per_comp[comp] = {"has_prev": True, "prev": prev, "curr_key": curr_key, "comp_df": comp_df}
                debug_rows.append([comp, f'{prev["snapshot_date"]} ({prev["filename"]})', int(len(comp_df)), len(curr_key)])

        st.session_state["per_comp"] = per_comp

        st.markdown("### 🔍 Debug rápido (pra ter certeza que comparou)")
        dbg = pd.DataFrame(debug_rows, columns=["concorrente", "comparou_com", "itens_em_comum", "itens_no_upload"])
        st.dataframe(dbg, use_container_width=True, height=160)

        signals = []
        stock_map_all = pd.concat(stock_map_all, ignore_index=True) if stock_map_all else pd.DataFrame()

        # sinais por concorrente
        for comp, pack in per_comp.items():
            if not pack["has_prev"]:
                continue

            d = pack["comp_df"].copy()

            d["flag_drop"] = d["drop_estoque_pct"].apply(lambda x: (not pd.isna(x)) and (float(x) >= float(min_drop_pct)))
            d["flag_price_up"] = d["preco_change_pct"].apply(lambda x: (not pd.isna(x)) and (float(x) >= float(min_price_up_pct)))
            d["flag_stock_crit"] = d["estoque_shared_new"].fillna(10**9) <= float(stock_crit)

            if require_stock_crit:
                d["flag_attack"] = d["flag_drop"] & d["flag_price_up"] & d["flag_stock_crit"]
            else:
                d["flag_attack"] = d["flag_drop"] & d["flag_price_up"]

            d_attack = d[d["flag_attack"]].copy()
            if not d_attack.empty:
                d_attack["concorrente"] = comp
                signals.append(
                    d_attack[
                        [
                            "key",
                            "titulo_new",
                            "marca_new",
                            "drop_estoque_pct",
                            "preco_min_old",
                            "preco_min_new",
                            "preco_change_pct",
                            "anuncios_irmaos_new",
                            "concorrente",
                        ]
                    ]
                )

        st.markdown("### 🎯 Produtos para ATACAR (BI CLEAN)")

        if not signals:
            st.info("Não apareceu ATAQUE com as regras atuais. Baixa as % ou exige sinal em menos concorrentes.")
        else:
            signals = pd.concat(signals, ignore_index=True)

            agg = signals.groupby("key", as_index=False).agg(
                produto=("titulo_new", "first"),
                marca=("marca_new", "first"),
                concorrentes_com_sinal=("concorrente", "nunique"),
                queda_estoque_pct_max=("drop_estoque_pct", "max"),
                preco_old_min=("preco_min_old", "min"),
                preco_new_min=("preco_min_new", "min"),
                preco_pct_max=("preco_change_pct", "max"),
                anuncios_irmaos_max=("anuncios_irmaos_new", "max"),
            )

            agg = agg[agg["concorrentes_com_sinal"] >= int(need_comp)].copy()

            if agg.empty:
                st.warning("Teve sinal, mas não bateu o mínimo de concorrentes. Abaixa 'Sinal em quantos concorrentes?'.")
            else:
                pivot_stock = stock_map_all.pivot_table(index="key", columns="concorrente", values="estoque_shared", aggfunc="max").reset_index()
                for c in COMPETITORS:
                    if c not in pivot_stock.columns:
                        pivot_stock[c] = pd.NA

                pivot_stock["estoque_atual_min"] = pivot_stock[COMPETITORS].min(axis=1, skipna=True)

                def stock_str(row):
                    parts = []
                    for c in COMPETITORS:
                        v = row.get(c)
                        parts.append(f"{c}:{'—' if pd.isna(v) else int(v)}")
                    return " | ".join(parts)

                pivot_stock["estoques_por_concorrente"] = pivot_stock.apply(stock_str, axis=1)

                agg = agg.merge(pivot_stock[["key", "estoque_atual_min", "estoques_por_concorrente"]], on="key", how="left")

                show = pd.DataFrame(
                    {
                        "SKU / ID": agg["key"],
                        "Produto": agg["produto"],
                        "Marca": agg["marca"],
                        "Concorrentes c/ sinal": agg["concorrentes_com_sinal"],
                        "Estoque atual (mín)": agg["estoque_atual_min"],
                        "Estoques por concorrente": agg["estoques_por_concorrente"],
                        "Queda estoque": agg["queda_estoque_pct_max"].apply(lambda x: -abs(x) if not pd.isna(x) else pd.NA),
                        "Preço anterior": agg["preco_old_min"],
                        "Preço atual": agg["preco_new_min"],
                        "Variação preço": agg["preco_pct_max"],
                        "Anúncios irmãos": agg["anuncios_irmaos_max"],
                    }
                )

                show["_drop_num"] = agg["queda_estoque_pct_max"].fillna(0.0).astype(float)
                show["_price_num"] = agg["preco_pct_max"].fillna(0.0).astype(float)

                show = show.sort_values(["_drop_num", "_price_num"], ascending=[False, False]).head(int(top_n)).copy()

                show["Queda estoque"] = show["Queda estoque"].apply(fmt_pct)
                show["Variação preço"] = show["Variação preço"].apply(fmt_pct)
                show["Preço anterior"] = show["Preço anterior"].apply(fmt_price)
                show["Preço atual"] = show["Preço atual"].apply(fmt_price)

                def style_row(row):
                    cols = list(row.index)
                    styles = [""] * len(cols)

                    def idx(c): return cols.index(c)

                    q = row.get("Queda estoque", "—")
                    if isinstance(q, str) and q != "—":
                        if q.strip().startswith("-"):
                            styles[idx("Queda estoque")] = "color:#b91c1c;font-weight:700;"

                    p = row.get("Variação preço", "—")
                    if isinstance(p, str) and p != "—":
                        if p.strip().startswith("-"):
                            styles[idx("Variação preço")] = "color:#b91c1c;font-weight:700;"
                        elif p.strip().startswith("+"):
                            styles[idx("Variação preço")] = "color:#15803d;font-weight:700;"

                    return styles

                show_display = show.drop(columns=["_drop_num", "_price_num"], errors="ignore")

                st.dataframe(show_display.style.apply(style_row, axis=1), use_container_width=True, height=420)

                st.download_button(
                    "Baixar BI (CSV)",
                    show_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"bi_clean_{date.today().isoformat()}.csv",
                    mime="text/csv",
                )

# -------------------------
# ALTERAÇÕES DE PREÇO (CLEAN)
# -------------------------
with tab_precos:
    st.markdown("## 💸 Alterações de Preço (Clean)")
    st.caption("Tudo que mudou de preço (subiu/baixou) comparando com o último upload do mesmo concorrente.")

    per_comp = st.session_state.get("per_comp")

    if not per_comp:
        st.info("Primeiro rode o **Processar e gerar BI** na aba BI (Clean). Depois volte aqui.")
    else:
        price_rows = []
        for comp, pack in per_comp.items():
            if not pack.get("has_prev"):
                continue

            d = pack["comp_df"].copy()
            d = d[(d["preco_min_old"].fillna(-1) != d["preco_min_new"].fillna(-1))].copy()
            if d.empty:
                continue

            d["concorrente"] = comp
            price_rows.append(
                d[
                    [
                        "key",
                        "titulo_new",
                        "marca_new",
                        "concorrente",
                        "preco_min_old",
                        "preco_min_new",
                        "preco_change_pct",
                        "estoque_shared_new",
                        "anuncios_irmaos_new",
                    ]
                ]
            )

        if not price_rows:
            st.info("Não achei mudanças de preço entre o upload atual e o anterior.")
        else:
            price_all = pd.concat(price_rows, ignore_index=True)

            price_all = price_all.rename(
                columns={
                    "key": "SKU / ID",
                    "titulo_new": "Produto",
                    "marca_new": "Marca",
                    "concorrente": "Concorrente",
                    "preco_min_old": "Preço anterior",
                    "preco_min_new": "Preço atual",
                    "preco_change_pct": "Variação preço",
                    "estoque_shared_new": "Estoque atual",
                    "anuncios_irmaos_new": "Anúncios irmãos",
                }
            )

            price_all["_pct_num"] = pd.to_numeric(price_all["Variação preço"], errors="coerce")
            price_all["Variação preço"] = price_all["Variação preço"].apply(fmt_pct)
            price_all["Preço anterior"] = price_all["Preço anterior"].apply(fmt_price)
            price_all["Preço atual"] = price_all["Preço atual"].apply(fmt_price)

            up = price_all[price_all["_pct_num"].fillna(0) > 0].copy().sort_values("_pct_num", ascending=False).drop(columns=["_pct_num"])
            down = price_all[price_all["_pct_num"].fillna(0) < 0].copy().sort_values("_pct_num", ascending=True).drop(columns=["_pct_num"])

            def style_price_row(row):
                cols = list(row.index)
                styles = [""] * len(cols)
                if "Variação preço" in cols:
                    v = row["Variação preço"]
                    if isinstance(v, str) and v != "—":
                        if v.strip().startswith("-"):
                            styles[cols.index("Variação preço")] = "color:#b91c1c;font-weight:700;"
                        elif v.strip().startswith("+"):
                            styles[cols.index("Variação preço")] = "color:#15803d;font-weight:700;"
                return styles

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ✅ Preços que subiram")
                st.dataframe(up.style.apply(style_price_row, axis=1), use_container_width=True, height=520)
            with c2:
                st.markdown("### 🔻 Preços que caíram")
                st.dataframe(down.style.apply(style_price_row, axis=1), use_container_width=True, height=520)

# -------------------------
# CONTROLE / HISTÓRICO
# -------------------------
with tab_ctrl:
    st.markdown("## 🧾 Controle (histórico)")
    hist = list_snapshots(limit=300)
    st.dataframe(hist.drop(columns=["id"], errors="ignore"), use_container_width=True, height=520)

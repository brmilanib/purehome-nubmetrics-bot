import os
import re
import json
import sqlite3
from datetime import datetime, date

import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
DB_PATH = os.environ.get("DB_PATH", "/tmp/snapshots.db")

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

# Mapeamento de colunas do Nubmetrics -> nomes canônicos
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
    "n° peça": "n_peca",
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
            (competitor, snapshot_date),
        ).fetchone()

    if not row:
        return None

    d, uploaded_at, filename, data_json = row
    data = json.loads(data_json)
    df = pd.DataFrame(data)
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}


def list_snapshots(limit=200):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT competitor, snapshot_date, uploaded_at, filename, id
            FROM snapshots
            ORDER BY snapshot_date DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = pd.DataFrame(rows, columns=["Concorrente", "Data snapshot", "Upload em", "Arquivo", "id"])
    return out


# =========================
# UTIL
# =========================
def try_extract_date_from_filename(name: str):
    """
    Tenta achar uma data no nome do arquivo.
    Aceita padrões como:
    - 2025-12-21
    - 21-12-2025 / 21_12_2025 / 21.12.2025
    """
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

    # garante colunas básicas
    needed = WATCH_COLS + ["sku", "gtin", "n_peca", "titulo", "marca"]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    # tipos numéricos
    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")

    # limpa ids como string
    for c in ["sku", "gtin", "n_peca"]:
        df[c] = df[c].astype(str).replace({"nan": "", "None": ""}).str.strip()

    df["titulo"] = df["titulo"].astype(str).replace({"nan": "", "None": ""}).str.strip()
    df["marca"] = df["marca"].astype(str).replace({"nan": "", "None": ""}).str.strip()

    # cria key robusta: sku > gtin > n_peca > titulo
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

    # normaliza booleanos (Sim/Não)
    for b in ["desconto", "frete_gratis", "full", "mercado_envios", "catalogo", "mercadopago"]:
        if b in df.columns:
            df[b] = df[b].astype(str).replace({"nan": "", "None": ""}).str.strip()

    return df


def aggregate_per_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Junta anúncios irmãos (mesmo key):
    - NÃO soma estoque (usa MAX, porque é o estoque "compartilhado" na prática)
    - preço usa MIN (pra capturar o anúncio mais agressivo)
    - cria contagem de anúncios irmãos
    """
    df = df.copy()

    # preço min / max / avg
    agg = df.groupby("key", as_index=False).agg(
        titulo=("titulo", "first"),
        marca=("marca", "first"),
        preco_min=("preco_medio", "min"),
        preco_max=("preco_medio", "max"),
        preco_avg=("preco_medio", "mean"),
        estoque_shared=("estoque", "max"),
        anuncios_irmaos=("key", "size"),
    )

    return agg


def pct_drop(old, new):
    # queda de estoque em %
    if pd.isna(old) or old is None or old <= 0:
        return pd.NA
    if pd.isna(new) or new is None:
        return pd.NA
    return (old - new) / old * 100.0


def pct_change(old, new):
    # variação de preço em %
    if pd.isna(old) or old is None or old <= 0:
        return pd.NA
    if pd.isna(new) or new is None:
        return pd.NA
    return (new - old) / old * 100.0


def fmt_price(v):
    if pd.isna(v):
        return "—"
    try:
        # 2 casas, mas remove zeros desnecessários
        s = f"{float(v):.2f}"
        s = s.rstrip("0").rstrip(".")
        # troca ponto por vírgula
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


# =========================
# CORE (comparação)
# =========================
def compute_prev_curr(prev_key: pd.DataFrame, curr_key: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe tabelas por KEY (já agregadas), e devolve tabela comparativa.
    """
    prev_key = prev_key.copy()
    curr_key = curr_key.copy()

    m = curr_key.merge(prev_key, on="key", how="inner", suffixes=("_new", "_old"))

    # estoque drop
    m["drop_estoque_pct"] = m.apply(
        lambda r: pct_drop(r.get("estoque_shared_old"), r.get("estoque_shared_new")), axis=1
    )

    # preço change (usa preço mínimo por key)
    m["preco_change_pct"] = m.apply(
        lambda r: pct_change(r.get("preco_min_old"), r.get("preco_min_new")), axis=1
    )

    return m


# =========================
# UI / APP
# =========================
init_db()

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics (um por concorrente) → compara com o upload anterior + BI clean de ATAQUE.")

tab_bi, tab_precos, tab_ctrl = st.tabs(["📌 BI (Clean)", "💸 Alterações de Preço (Clean)", "🧾 Controle (histórico)"])

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

    st.caption(
        "ATAQUE = estoque caiu forte + (opcional) ficou crítico + preço (min) subiu, comparando SEMPRE com o upload anterior do mesmo concorrente."
    )

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
        # valida uploads
        missing = [c for c, f in uploaded_files.items() if f is None]
        if missing:
            st.error(f"Faltou upload: {', '.join(missing)}")
            st.stop()

        per_comp = {}
        debug_rows = []

        # 1) carrega e salva snapshot atual (AGREGADO POR KEY)
        for comp, f in uploaded_files.items():
            df_raw = pd.read_excel(f)
            df_norm = normalize_df(df_raw)
            df_key = aggregate_per_key(df_norm)

            # define snapshot_date
            effective_date = snap_date_str
            if auto_date_from_name:
                dname = try_extract_date_from_filename(f.name)
                if dname:
                    effective_date = dname

            prev = load_last_snapshot_before(comp, effective_date)

            # salva snapshot atual
            save_snapshot(comp, effective_date, f.name, df_key)

            if prev is None:
                per_comp[comp] = {
                    "has_prev": False,
                    "prev": None,
                    "curr_date": effective_date,
                    "curr_key": df_key,
                    "comp_df": pd.DataFrame(),
                }
                debug_rows.append([comp, "—", 0, len(df_key)])
            else:
                prev_key = prev["df"]
                curr_key = df_key

                comp_df = compute_prev_curr(prev_key, curr_key)

                # debug: itens em comum
                debug_rows.append([comp, prev["snapshot_date"], int(len(comp_df)), len(curr_key)])

                per_comp[comp] = {
                    "has_prev": True,
                    "prev": prev,
                    "curr_date": effective_date,
                    "curr_key": curr_key,
                    "comp_df": comp_df,
                }

        # 2) Debug rápido (pra você ter certeza que comparou)
        st.markdown("### 🔍 Debug rápido (pra você ter certeza que comparou)")
        dbg = pd.DataFrame(debug_rows, columns=["concorrente", "comparou_com", "itens_em_comum", "itens_no_upload"])
        st.dataframe(dbg, use_container_width=True, height=160)

        if (dbg["itens_em_comum"] == 0).all():
            st.warning(
                "Não teve itens em comum para comparar (keys não bateram). "
                "Isso acontece se algum export veio diferente/sem SKU/GTIN. "
                "Mas pelo seu print já está comparando — então daqui pra frente você está ok."
            )

        # 3) monta BI clean (ATAQUE)
        # sinal por concorrente
        signals = []
        stock_map_all = []  # para mostrar estoque por concorrente no último upload

        for comp, pack in per_comp.items():
            # estoque atual por key (do último upload)
            currk = pack["curr_key"][["key", "estoque_shared", "anuncios_irmaos"]].copy()
            currk["concorrente"] = comp
            stock_map_all.append(currk)

            if not pack["has_prev"]:
                continue

            d = pack["comp_df"].copy()

            # sinal: queda estoque >= min_drop_pct e preço subiu >= min_price_up_pct
            d["flag_drop"] = d["drop_estoque_pct"].apply(lambda x: (not pd.isna(x)) and (float(x) >= float(min_drop_pct)))
            d["flag_price_up"] = d["preco_change_pct"].apply(lambda x: (not pd.isna(x)) and (float(x) >= float(min_price_up_pct)))

            # estoque crítico (novo)
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
                            "estoque_shared_old",
                            "estoque_shared_new",
                            "drop_estoque_pct",
                            "preco_min_old",
                            "preco_min_new",
                            "preco_change_pct",
                            "anuncios_irmaos_new",
                            "concorrente",
                        ]
                    ]
                )

        stock_map_all = pd.concat(stock_map_all, ignore_index=True) if stock_map_all else pd.DataFrame()
        signals = pd.concat(signals, ignore_index=True) if signals else pd.DataFrame()

        st.markdown("### 🎯 Produtos para ATACAR (BI CLEAN)")

        if signals.empty:
            st.info(
                "Com as regras atuais, não apareceu nada forte em concorrentes suficientes.\n\n"
                "Dica: baixa a % de queda ou a % de alta, ou exige sinal em menos concorrentes."
            )
        else:
            # agrega por key (quantos concorrentes deram sinal)
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

            # filtra pelo número mínimo de concorrentes
            agg = agg[agg["concorrentes_com_sinal"] >= int(need_comp)].copy()

            if agg.empty:
                st.warning(
                    "Teve sinais individuais, mas não bateu o mínimo de concorrentes exigidos.\n"
                    "Abaixa 'Sinal em quantos concorrentes?' pra 1 ou 2."
                )
            else:
                # cria estoque por concorrente (do último upload atual)
                # e estoque mínimo atual
                pivot_stock = stock_map_all.pivot_table(index="key", columns="concorrente", values="estoque_shared", aggfunc="max")
                pivot_stock = pivot_stock.reset_index()

                for c in COMPETITORS:
                    if c not in pivot_stock.columns:
                        pivot_stock[c] = pd.NA

                # estoque mínimo
                pivot_stock["estoque_atual_min"] = pivot_stock[COMPETITORS].min(axis=1, skipna=True)

                # string "AUMA:14 | BAGATELLE:— | PERFUMES_BHZ:12"
                def stock_str(row):
                    parts = []
                    for c in COMPETITORS:
                        v = row.get(c)
                        if pd.isna(v):
                            parts.append(f"{c}:—")
                        else:
                            try:
                                parts.append(f"{c}:{int(v)}")
                            except Exception:
                                parts.append(f"{c}:{v}")
                    return " | ".join(parts)

                pivot_stock["estoques_por_concorrente"] = pivot_stock.apply(stock_str, axis=1)

                # junta no agg
                agg = agg.merge(
                    pivot_stock[["key", "estoque_atual_min", "estoques_por_concorrente"]],
                    on="key",
                    how="left",
                )

                # monta tabela final com nomes humanos
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

                # ordena: mais queda + mais alta
                show["_drop_num"] = agg["queda_estoque_pct_max"].fillna(0.0).astype(float)
                show["_price_num"] = agg["preco_pct_max"].fillna(0.0).astype(float)
                show = show.sort_values(["_drop_num", "_price_num"], ascending=[False, False]).head(int(top_n)).copy()

                # formata exibição
                show["Queda estoque"] = show["Queda estoque"].apply(fmt_pct)
                show["Variação preço"] = show["Variação preço"].apply(fmt_pct)
                show["Preço anterior"] = show["Preço anterior"].apply(fmt_price)
                show["Preço atual"] = show["Preço atual"].apply(fmt_price)

                # estilo (pinta e negrita)
                def style_row(row):
                    styles = [""] * len(row)

                    # encontra índices
                    cols = list(row.index)
                    def idx(c): return cols.index(c)

                    # Queda estoque (sempre negativo)
                    q = row.get("Queda estoque", "—")
                    if isinstance(q, str) and q != "—":
                        if q.strip().startswith("-"):
                            styles[idx("Queda estoque")] = "color:#b91c1c;font-weight:700;"
                        elif q.strip().startswith("+"):
                            styles[idx("Queda estoque")] = "color:#15803d;font-weight:700;"

                    # Variação preço
                    p = row.get("Variação preço", "—")
                    if isinstance(p, str) and p != "—":
                        if p.strip().startswith("-"):
                            styles[idx("Variação preço")] = "color:#b91c1c;font-weight:700;"
                        elif p.strip().startswith("+"):
                            styles[idx("Variação preço")] = "color:#15803d;font-weight:700;"

                    return styles

                # remove colunas internas antes de mostrar
                show_display = show.drop(columns=["_drop_num", "_price_num"], errors="ignore")

                st.dataframe(
                    show_display.style.apply(style_row, axis=1),
                    use_container_width=True,
                    height=420,
                )

                st.download_button(
                    "Baixar BI (CSV)",
                    show_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"bi_clean_{snap_date_str}.csv",
                    mime="text/csv",
                )

with tab_precos:
    st.markdown("## 💸 Alterações de Preço (Clean)")
    st.caption("Aqui entra TUDO que mudou de preço (subiu ou caiu), sem depender de % mínimo.")

    # Para esta aba funcionar, o usuário precisa ter processado no tab BI (porque depende do per_comp)
    if "per_comp" not in st.session_state:
        st.info("Primeiro rode o **Processar e gerar BI** na aba BI (Clean). Depois volte aqui.")
    else:
        per_comp = st.session_state["per_comp"]

    # Se não existir, tenta reconstruir do que estiver no session_state
    try:
        per_comp = st.session_state.get("per_comp", None)
    except Exception:
        per_comp = None

    if per_comp is None:
        st.info("Primeiro rode o **Processar e gerar BI** na aba BI (Clean). Depois volte aqui.")
    else:
        price_rows = []
        for comp, pack in per_comp.items():
            if not pack.get("has_prev"):
                continue

            d = pack["comp_df"].copy()
            # considera mudança real
            d = d[(d["preco_min_old"].fillna(-1) != d["preco_min_new"].fillna(-1))].copy()
            if d.empty:
                continue

            d["concorrente"] = comp
            d["variacao_pct"] = d["preco_change_pct"]
            price_rows.append(
                d[
                    [
                        "key",
                        "titulo_new",
                        "marca_new",
                        "concorrente",
                        "preco_min_old",
                        "preco_min_new",
                        "variacao_pct",
                        "estoque_shared_new",
                        "anuncios_irmaos_new",
                    ]
                ]
            )

        if not price_rows:
            st.info("Não achei mudanças de preço entre o upload atual e o anterior.")
        else:
            price_all = pd.concat(price_rows, ignore_index=True)

            # tabela humanizada
            price_all = price_all.rename(
                columns={
                    "key": "SKU / ID",
                    "titulo_new": "Produto",
                    "marca_new": "Marca",
                    "concorrente": "Concorrente",
                    "preco_min_old": "Preço anterior",
                    "preco_min_new": "Preço atual",
                    "variacao_pct": "Variação preço",
                    "estoque_shared_new": "Estoque atual",
                    "anuncios_irmaos_new": "Anúncios irmãos",
                }
            )

            # formata
            price_all["_pct_num"] = pd.to_numeric(price_all["Variação preço"], errors="coerce")
            price_all["Variação preço"] = price_all["Variação preço"].apply(fmt_pct)
            price_all["Preço anterior"] = price_all["Preço anterior"].apply(fmt_price)
            price_all["Preço atual"] = price_all["Preço atual"].apply(fmt_price)

            up = price_all[price_all["_pct_num"].fillna(0) > 0].copy().sort_values("_pct_num", ascending=False)
            down = price_all[price_all["_pct_num"].fillna(0) < 0].copy().sort_values("_pct_num", ascending=True)

            up = up.drop(columns=["_pct_num"], errors="ignore")
            down = down.drop(columns=["_pct_num"], errors="ignore")

            c1, c2 = st.columns(2)

            def style_price_row(row):
                styles = [""] * len(row)
                cols = list(row.index)
                if "Variação preço" in cols:
                    v = row["Variação preço"]
                    if isinstance(v, str) and v != "—":
                        if v.strip().startswith("-"):
                            styles[cols.index("Variação preço")] = "color:#b91c1c;font-weight:700;"
                        elif v.strip().startswith("+"):
                            styles[cols.index("Variação preço")] = "color:#15803d;font-weight:700;"
                return styles

            with c1:
                st.markdown("### ✅ Preços que subiram")
                st.dataframe(up.style.apply(style_price_row, axis=1), use_container_width=True, height=520)
                st.download_button(
                    "Baixar (subiram) CSV",
                    up.to_csv(index=False).encode("utf-8"),
                    file_name=f"precos_subiram_{date.today().isoformat()}.csv",
                    mime="text/csv",
                )

            with c2:
                st.markdown("### 🔻 Preços que caíram")
                st.dataframe(down.style.apply(style_price_row, axis=1), use_container_width=True, height=520)
                st.download_button(
                    "Baixar (caíram) CSV",
                    down.to_csv(index=False).encode("utf-8"),
                    file_name=f"precos_cairam_{date.today().isoformat()}.csv",
                    mime="text/csv",
                )

with tab_ctrl:
    st.markdown("## 🧾 Controle (histórico)")
    st.caption("Aqui você vê tudo que já subiu e ficou guardado no histórico do app.")

    hist = list_snapshots(limit=200)
    st.dataframe(hist.drop(columns=["id"], errors="ignore"), use_container_width=True, height=520)

# =========================
# Persistência de estado entre abas
# =========================
# (Guarda per_comp pro tab de preços funcionar depois do BI)
if "per_comp" not in st.session_state:
    st.session_state["per_comp"] = None

# Quando processa no BI, guarda no session_state
try:
    if "per_comp" in locals() and per_comp:
        st.session_state["per_comp"] = per_comp
except Exception:
    pass

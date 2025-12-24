import os
import re
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date

# =========================================================
# CONFIG / DB
# =========================================================
DB_PATH = os.environ.get("DB_PATH", "/tmp/snapshots.db")

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Suba 3 exports do Nubmetrics (um por concorrente) → compara com o upload anterior + BI clean + Ranking de mais vendidos.")

# =========================================================
# DB HELPERS
# =========================================================
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

def save_snapshot(competitor: str, snapshot_date: str, filename: str, df: pd.DataFrame):
    payload = df.to_json(orient="records", force_ascii=False)
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
    df = pd.read_json(data_json, orient="records")
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}

def load_latest_snapshot(competitor: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT snapshot_date, uploaded_at, filename, data_json
            FROM snapshots
            WHERE competitor = ?
            ORDER BY snapshot_date DESC, id DESC
            LIMIT 1
            """,
            (competitor,)
        ).fetchone()
    if not row:
        return None
    d, uploaded_at, filename, data_json = row
    df = pd.read_json(data_json, orient="records")
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}

def load_snapshot_exact(competitor: str, snapshot_date: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT snapshot_date, uploaded_at, filename, data_json
            FROM snapshots
            WHERE competitor = ? AND snapshot_date = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (competitor, snapshot_date)
        ).fetchone()
    if not row:
        return None
    d, uploaded_at, filename, data_json = row
    df = pd.read_json(data_json, orient="records")
    return {"snapshot_date": d, "uploaded_at": uploaded_at, "filename": filename, "df": df}

# =========================================================
# PARSE DATE FROM FILENAME
# =========================================================
def parse_date_from_filename(name: str):
    """
    Tenta achar data no nome do arquivo:
    - 2025-12-21
    - 20251221
    - 21-12-2025
    - 21122025
    """
    s = name.strip()

    m = re.search(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    m = re.search(r"(20\d{2})(\d{2})(\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    m = re.search(r"(\d{2})[-_/](\d{2})[-_/](20\d{2})", s)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    m = re.search(r"(\d{2})(\d{2})(20\d{2})", s)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    return None

# =========================================================
# NORMALIZAÇÃO (Nubmetrics → colunas padrão)
# =========================================================
CANON_MAP = {
    "título": "titulo",
    "titulo": "titulo",
    "marca": "marca",
    "vendas em $": "vendas_valor",
    "vendas em unid.": "vendas_unid",
    "vendas em unid": "vendas_unid",
    "preço médio": "preco_medio",
    "preco médio": "preco_medio",
    "preco medio": "preco_medio",
    "tipo de publicação": "tipo_publicacao",
    "tipo de publicacao": "tipo_publicacao",
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
    "n peça": "n_peca",
    "estado": "estado",
    "mercadopago": "mercadopago",
    "republicada": "republicada",
    "condição": "condicao",
    "condicao": "condicao",
    "estoque": "estoque",
}

EXPECTED = [
    "titulo","marca","vendas_valor","vendas_unid","preco_medio","tipo_publicacao",
    "full","catalogo","frete_gratis","mercado_envios","desconto",
    "sku","oem","gtin","n_peca","estoque"
]

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: CANON_MAP.get(c, c) for c in df.columns})

    for c in EXPECTED:
        if c not in df.columns:
            df[c] = pd.NA

    # tipos numéricos
    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")
    df["vendas_valor"] = pd.to_numeric(df["vendas_valor"], errors="coerce")

    # chave robusta (prioridade: SKU > GTIN > N_Peca > OEM > fallback titulo+marca)
    def make_key(row):
        for k in ["sku","gtin","n_peca","oem"]:
            v = row.get(k)
            if pd.notna(v) and str(v).strip() != "":
                return str(v).strip()
        t = str(row.get("titulo") or "").strip().lower()
        m = str(row.get("marca") or "").strip().lower()
        if t:
            base = (m + " " + t).strip()
            base = re.sub(r"\s+", " ", base)
            return "TT_" + base[:120]
        return None

    df["key"] = df.apply(make_key, axis=1)
    df = df[df["key"].notna()].copy()
    df["key"] = df["key"].astype(str)

    # limpeza de "Sim/Não"
    for b in ["desconto","frete_gratis","full","mercado_envios","catalogo","mercadopago"]:
        if b in df.columns:
            df[b] = df[b].astype(str).str.strip()

    return df

# =========================================================
# AGREGAÇÃO "ANÚNCIOS IRMÃOS" (mesmo SKU/key)
# Regra: NÃO somar estoque (mesmo estoque compartilhado).
# - Estoque: usa MAX (seguro)
# - Vendas: soma (queremos ranking total do SKU)
# - Preço: usa MIN (melhor preço anunciado)
# - Título: pega do anúncio com mais vendas
# =========================================================
def aggregate_siblings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # garante colunas
    for c in ["titulo","marca","preco_medio","estoque","vendas_unid","vendas_valor"]:
        if c not in df.columns:
            df[c] = pd.NA

    # idx do "representante" por key (mais vendido)
    def pick_rep(group: pd.DataFrame):
        if group["vendas_unid"].notna().any():
            idx = group["vendas_unid"].fillna(-1).idxmax()
        else:
            idx = group.index[0]
        return idx

    reps = df.groupby("key").apply(pick_rep)
    rep_rows = df.loc[reps.values, ["key","titulo","marca"]].set_index("key")

    agg = df.groupby("key", as_index=False).agg(
        preco_min=("preco_medio","min"),
        preco_max=("preco_medio","max"),
        estoque_atual=("estoque","max"),
        vendas_unid=("vendas_unid","sum"),
        vendas_valor=("vendas_valor","sum"),
        anuncios_irmaos=("key","size"),
    )

    agg = agg.set_index("key").join(rep_rows, how="left").reset_index()
    # reordena
    agg = agg[[
        "key","titulo","marca",
        "vendas_unid","vendas_valor",
        "preco_min","preco_max",
        "estoque_atual",
        "anuncios_irmaos"
    ]]

    return agg

# =========================================================
# DIFF (produto-level)
# =========================================================
def safe_pct(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0:
        return pd.NA
    return (new - old) / old * 100.0

def compute_product_diff(prev_prod: pd.DataFrame, curr_prod: pd.DataFrame) -> pd.DataFrame:
    prev = prev_prod.copy()
    curr = curr_prod.copy()

    merged = curr.merge(prev, on="key", how="outer", suffixes=("_new","_old"), indicator=True)

    # Para ranking/BI: queremos só itens que existem nos dois (pra variação)
    both = merged[merged["_merge"] == "both"].copy()

    # variação preço: comparamos preco_min
    both["preco_old"] = both["preco_min_old"]
    both["preco_new"] = both["preco_min_new"]
    both["preco_pct"] = both.apply(lambda r: safe_pct(r["preco_new"], r["preco_old"]), axis=1)

    # variação estoque: comparamos estoque_atual
    both["estoque_old"] = both["estoque_atual_old"]
    both["estoque_new"] = both["estoque_atual_new"]
    both["estoque_pct"] = both.apply(lambda r: safe_pct(r["estoque_new"], r["estoque_old"]), axis=1)

    # "queda estoque" em % (positivo significa caiu)
    # ex: old=100 new=40 => queda=60%
    def drop_pct(new, old):
        if pd.isna(new) or pd.isna(old) or old == 0:
            return pd.NA
        return (old - new) / old * 100.0

    both["queda_estoque_pct"] = both.apply(lambda r: drop_pct(r["estoque_new"], r["estoque_old"]), axis=1)

    out = both[[
        "key",
        "titulo_new","marca_new",
        "vendas_unid_new","vendas_valor_new",
        "preco_old","preco_new","preco_pct",
        "estoque_old","estoque_new","queda_estoque_pct",
        "anuncios_irmaos_new"
    ]].copy()

    out.rename(columns={
        "titulo_new": "produto",
        "marca_new": "marca",
        "vendas_unid_new": "vendas_unid",
        "vendas_valor_new": "vendas_valor",
        "anuncios_irmaos_new": "anuncios_irmaos"
    }, inplace=True)

    return out

# =========================================================
# FORMATAÇÃO (tabelas clean)
# =========================================================
def fmt_currency(x):
    if pd.isna(x):
        return ""
    try:
        return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(x)

def fmt_pct(x):
    if pd.isna(x):
        return ""
    try:
        v = float(x)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.2f}%".replace(".", ",")
    except:
        return str(x)

def style_pct_series(s: pd.Series):
    # negativos em vermelho negrito, positivos em verde negrito
    styles = []
    for v in s:
        if v == "" or pd.isna(v):
            styles.append("")
            continue
        # v já pode estar numérico ou string
        try:
            num = float(v)
        except:
            # tenta extrair
            try:
                num = float(str(v).replace("%","").replace(".","").replace(",",".").replace("+",""))
            except:
                styles.append("")
                continue
        if num < 0:
            styles.append("color:#b00020; font-weight:700;")
        elif num > 0:
            styles.append("color:#0b6b0b; font-weight:700;")
        else:
            styles.append("")
    return styles

def style_drop_series(s: pd.Series):
    # Queda estoque (positivo = caiu) -> vermelho negrito quando alto
    styles = []
    for v in s:
        if v == "" or pd.isna(v):
            styles.append("")
            continue
        try:
            num = float(v)
        except:
            try:
                num = float(str(v).replace("%","").replace(".","").replace(",",".").replace("+",""))
            except:
                styles.append("")
                continue
        if num > 0:
            styles.append("color:#b00020; font-weight:700;")
        else:
            styles.append("")
    return styles

# =========================================================
# UI: TABS
# =========================================================
tab_bi, tab_ranking, tab_controle = st.tabs(["🎯 BI (Clean)", "🏆 Ranking Mais Vendidos", "🗂️ Controle (histórico)"])

# =========================================================
# TAB BI (CLEAN)
# =========================================================
with tab_bi:
    st.subheader("Regras (você controla aqui)")

    c1, c2, c3, c4, c5, c6 = st.columns([1.1,1.2,1.2,1.2,1.2,1.0])
    with c1:
        estoque_critico = st.number_input("Estoque crítico (≤)", min_value=0, value=30, step=1)
    with c2:
        exigir_estoque_critico = st.checkbox("Exigir estoque crítico no ATAQUE", value=False)
    with c3:
        queda_min_estoque_pct = st.number_input("Queda mínima estoque (%)", min_value=0.0, value=10.0, step=1.0)
    with c4:
        alta_min_preco_pct = st.number_input("Alta mínima preço (%)", min_value=0.0, value=5.0, step=1.0)
    with c5:
        sinal_min_concorrentes = st.number_input("Sinal em quantos concorrentes?", min_value=1, max_value=3, value=1, step=1)
    with c6:
        top_n = st.number_input("Mostrar TOP N", min_value=5, max_value=200, value=40, step=5)

    st.caption("ATAQUE = estoque caiu (forte) + preço subiu (forte), sempre comparando com o upload anterior do MESMO concorrente. Depois junta sinais entre concorrentes.")

    st.subheader("Upload (3 arquivos do Nubmetrics)")

    pick_date_from_name = st.checkbox("Tentar pegar data automaticamente do nome do arquivo", value=True)

    snap_date = st.date_input("Se não achar no nome, usa esta data do snapshot", value=date.today())
    snap_date_str = snap_date.isoformat()

    ucols = st.columns(3)
    uploaded = {}
    for i, comp in enumerate(COMPETITORS):
        with ucols[i]:
            uploaded[comp] = st.file_uploader(f"{comp} (Export Nubmetrics)", type=["xlsx","xls"], key=f"up_{comp}")

    process = st.button("Processar e gerar BI", type="primary")

    # =========================================================
    # PROCESS
    # =========================================================
    if process:
        missing = [c for c, f in uploaded.items() if f is None]
        if missing:
            st.error(f"Faltou upload: {', '.join(missing)}")
            st.stop()

        per_comp = {}
        debug_rows = []

        # decide snapshot_date: se pegar do nome, usa do primeiro arquivo que tiver data
        effective_date = snap_date_str
        if pick_date_from_name:
            for comp, f in uploaded.items():
                d = parse_date_from_filename(f.name)
                if d:
                    effective_date = d
                    break

        for comp, f in uploaded.items():
            raw = pd.read_excel(f)
            norm = normalize_df(raw)
            prod = aggregate_siblings(norm)

            prev = load_last_snapshot_before(comp, effective_date)

            # salva o snapshot produto-level
            save_snapshot(comp, effective_date, f.name, prod)

            if prev is None:
                per_comp[comp] = {"prev": None, "curr": prod, "diff": None}
                debug_rows.append({
                    "concorrente": comp,
                    "comparou_com": "(primeiro snapshot)",
                    "itens_em_comum": 0,
                    "itens_no_upload": len(prod)
                })
                continue

            prev_df = prev["df"].copy()
            # garante colunas (caso mudou estrutura no passado)
            for c in prod.columns:
                if c not in prev_df.columns:
                    prev_df[c] = pd.NA

            # diff produto-level
            diff = compute_product_diff(prev_df, prod)

            # debug: itens em comum
            common = set(prev_df["key"].astype(str)).intersection(set(prod["key"].astype(str)))
            debug_rows.append({
                "concorrente": comp,
                "comparou_com": f'{prev["snapshot_date"]} ({prev["filename"]})',
                "itens_em_comum": len(common),
                "itens_no_upload": len(prod)
            })

            per_comp[comp] = {"prev": prev, "curr": prod, "diff": diff}

        st.divider()
        st.markdown("### 🔎 Debug rápido (pra você ter certeza que comparou)")
        debug_df = pd.DataFrame(debug_rows)
        st.dataframe(debug_df, use_container_width=True, height=180)

        # =========================================================
        # BI CLEAN: "ATAQUE" (queda estoque + alta preço)
        # =========================================================
        st.divider()
        st.markdown("## 🎯 Produtos para ATACAR (BI CLEAN)")

        signals = []
        for comp, d in per_comp.items():
            if d["diff"] is None:
                continue

            df = d["diff"].copy()
            df["concorrente"] = comp

            # critérios base
            df["sinal_preco"] = df["preco_pct"].fillna(-9999) >= float(alta_min_preco_pct)
            df["sinal_estoque"] = df["queda_estoque_pct"].fillna(-9999) >= float(queda_min_estoque_pct)

            if exigir_estoque_critico:
                df["sinal_critico"] = df["estoque_new"].fillna(10**9) <= float(estoque_critico)
            else:
                df["sinal_critico"] = True

            df["sinal"] = df["sinal_preco"] & df["sinal_estoque"] & df["sinal_critico"]

            df_sig = df[df["sinal"]].copy()
            if len(df_sig):
                signals.append(df_sig)

        if not signals:
            st.info("Ainda não apareceu nada forte com essas regras (ou é o primeiro upload de algum concorrente). Ajuste os %/regras e rode de novo.")
        else:
            sig_all = pd.concat(signals, ignore_index=True)

            # agrega por produto (key) juntando concorrentes com sinal
            def mk_estoques_por_conc(g):
                parts = []
                for _, r in g.iterrows():
                    e = "" if pd.isna(r["estoque_new"]) else int(r["estoque_new"])
                    parts.append(f'{r["concorrente"]}:{e}')
                return " | ".join(parts)

            out = sig_all.groupby("key", as_index=False).agg(
                produto=("produto","first"),
                marca=("marca","first"),
                concorrentes_com_sinal=("concorrente","nunique"),
                estoque_atual_min=("estoque_new","min"),
                estoques_por_concorrente=("estoque_new", lambda s: ""),  # placeholder
                queda_estoque_pct_max=("queda_estoque_pct","max"),
                preco_anterior=("preco_old","min"),
                preco_atual=("preco_new","min"),
                variacao_preco_pct_max=("preco_pct","max"),
                anuncios_irmaos=("anuncios_irmaos","max"),
            )

            # preencher estoques_por_concorrente via apply (mais controlável)
            sto_map = sig_all.groupby("key", as_index=False).apply(mk_estoques_por_conc).reset_index(drop=True)
            out["estoques_por_concorrente"] = sto_map

            # filtra por qtd concorrentes
            out = out[out["concorrentes_com_sinal"] >= int(sinal_min_concorrentes)].copy()

            # ordena (mais concorrentes + maior queda + maior alta)
            out["_ord"] = out["concorrentes_com_sinal"]*1000 + out["queda_estoque_pct_max"].fillna(0)*10 + out["variacao_preco_pct_max"].fillna(0)
            out = out.sort_values("_ord", ascending=False).drop(columns=["_ord"]).head(int(top_n))

            # formatação
            out_show = out.rename(columns={
                "key":"SKU / ID",
                "produto":"Produto",
                "marca":"Marca",
                "concorrentes_com_sinal":"Concorrentes c/ sinal",
                "estoque_atual_min":"Estoque atual (mín)",
                "estoques_por_concorrente":"Estoques por concorrente",
                "queda_estoque_pct_max":"Queda estoque",
                "preco_anterior":"Preço anterior",
                "preco_atual":"Preço atual",
                "variacao_preco_pct_max":"Variação preço",
                "anuncios_irmaos":"Anúncios irmãos",
            }).copy()

            # aplica formatos
            out_show["Queda estoque"] = out_show["Queda estoque"].apply(fmt_pct)
            out_show["Variação preço"] = out_show["Variação preço"].apply(fmt_pct)
            out_show["Preço anterior"] = out_show["Preço anterior"].apply(fmt_currency)
            out_show["Preço atual"] = out_show["Preço atual"].apply(fmt_currency)
            out_show["Estoque atual (mín)"] = out_show["Estoque atual (mín)"].fillna("").apply(lambda x: "" if x=="" else int(x))

            sty = out_show.style
            if "Queda estoque" in out_show.columns:
                # vermelho negrito quando caiu
                sty = sty.apply(style_drop_series, subset=["Queda estoque"])
            if "Variação preço" in out_show.columns:
                sty = sty.apply(style_pct_series, subset=["Variação preço"])

            st.dataframe(sty, use_container_width=True, height=420)

            st.download_button(
                "Baixar BI (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"bi_clean_{effective_date}.csv",
                mime="text/csv"
            )

        st.success("Pronto. Pode subir amanhã de novo — ele compara sempre com o último upload de cada concorrente.")

# =========================================================
# TAB RANKING MAIS VENDIDOS
# =========================================================
with tab_ranking:
    st.subheader("🏆 Ranking Mais Vendidos (último upload de cada concorrente)")

    # controles
    r1, r2, r3 = st.columns([1.2, 1.2, 2.0])
    with r1:
        metric = st.radio("Ordenar por", ["Vendas (unid)", "Vendas (R$)"], horizontal=True)
    with r2:
        top_rank = st.number_input("TOP N", min_value=10, max_value=1000, value=200, step=10)
    with r3:
        q = st.text_input("Buscar (produto / marca / SKU)", value="")

    rows = []
    for comp in COMPETITORS:
        latest = load_latest_snapshot(comp)
        if latest is None:
            continue

        curr = latest["df"].copy()
        curr["concorrente"] = comp
        curr["snapshot_date"] = latest["snapshot_date"]

        prev = load_last_snapshot_before(comp, latest["snapshot_date"])
        if prev is not None:
            diff = compute_product_diff(prev["df"], curr)
            diff["concorrente"] = comp
            diff["snapshot_date"] = latest["snapshot_date"]
            rows.append(diff)
        else:
            # sem comparação, ainda assim mostra estoque/preço atual (variações em branco)
            base = curr.copy()
            base = base.rename(columns={
                "key":"key",
                "titulo":"produto",
                "marca":"marca",
                "vendas_unid":"vendas_unid",
                "vendas_valor":"vendas_valor",
                "preco_min":"preco_new",
                "estoque_atual":"estoque_new",
                "anuncios_irmaos":"anuncios_irmaos"
            })
            base["preco_old"] = pd.NA
            base["preco_pct"] = pd.NA
            base["estoque_old"] = pd.NA
            base["queda_estoque_pct"] = pd.NA
            base = base[[
                "key","produto","marca","vendas_unid","vendas_valor",
                "preco_old","preco_new","preco_pct",
                "estoque_old","estoque_new","queda_estoque_pct",
                "anuncios_irmaos"
            ]].copy()
            base["concorrente"] = comp
            base["snapshot_date"] = latest["snapshot_date"]
            rows.append(base)

    if not rows:
        st.info("Ainda não tem dados salvos. Primeiro rode pelo menos 1 upload na aba BI (Clean).")
    else:
        df_rank = pd.concat(rows, ignore_index=True)

        # filtro busca
        if q.strip():
            qq = q.strip().lower()
            df_rank = df_rank[
                df_rank["produto"].astype(str).str.lower().str.contains(qq, na=False) |
                df_rank["marca"].astype(str).str.lower().str.contains(qq, na=False) |
                df_rank["key"].astype(str).str.lower().str.contains(qq, na=False)
            ].copy()

        # ordenar
        sort_col = "vendas_unid" if metric == "Vendas (unid)" else "vendas_valor"
        df_rank[sort_col] = pd.to_numeric(df_rank[sort_col], errors="coerce").fillna(0)

        df_rank = df_rank.sort_values(sort_col, ascending=False).head(int(top_rank))

        # montar tabela final
        out = df_rank.rename(columns={
            "key":"SKU / ID",
            "produto":"Produto",
            "marca":"Marca",
            "concorrente":"Concorrente",
            "snapshot_date":"Data (último upload)",
            "vendas_unid":"Vendas (unid)",
            "vendas_valor":"Vendas (R$)",
            "preco_old":"Preço anterior",
            "preco_new":"Preço atual",
            "preco_pct":"Variação preço",
            "estoque_old":"Estoque anterior",
            "estoque_new":"Estoque atual",
            "queda_estoque_pct":"Queda estoque",
            "anuncios_irmaos":"Anúncios irmãos",
        }).copy()

        # formatos
        out["Vendas (unid)"] = out["Vendas (unid)"].fillna(0).astype(int)
        out["Vendas (R$)"] = out["Vendas (R$)"].apply(fmt_currency)
        out["Preço anterior"] = out["Preço anterior"].apply(fmt_currency)
        out["Preço atual"] = out["Preço atual"].apply(fmt_currency)
        out["Variação preço"] = out["Variação preço"].apply(fmt_pct)
        out["Queda estoque"] = out["Queda estoque"].apply(fmt_pct)

        out["Estoque atual"] = out["Estoque atual"].fillna("").apply(lambda x: "" if x=="" else int(x) if pd.notna(x) else "")
        out["Estoque anterior"] = out["Estoque anterior"].fillna("").apply(lambda x: "" if x=="" else int(x) if pd.notna(x) else "")

        sty = out.style
        sty = sty.apply(style_pct_series, subset=["Variação preço"])
        sty = sty.apply(style_drop_series, subset=["Queda estoque"])

        st.dataframe(sty, use_container_width=True, height=560)

        st.download_button(
            "Baixar Ranking (CSV)",
            df_rank.to_csv(index=False).encode("utf-8"),
            file_name=f"ranking_mais_vendidos.csv",
            mime="text/csv"
        )

# =========================================================
# TAB CONTROLE (HISTÓRICO)
# =========================================================
with tab_controle:
    st.subheader("🗂️ Controle (histórico de uploads)")

    with sqlite3.connect(DB_PATH) as conn:
        hist = conn.execute("""
            SELECT competitor as concorrente, snapshot_date as data, uploaded_at as enviado_em, filename as arquivo
            FROM snapshots
            ORDER BY snapshot_date DESC, id DESC
            LIMIT 300
        """).fetchall()

    if not hist:
        st.info("Sem histórico ainda. Faça um upload na aba BI (Clean).")
    else:
        hist_df = pd.DataFrame(hist, columns=["concorrente","data","enviado_em","arquivo"])
        st.dataframe(hist_df, use_container_width=True, height=420)

        st.caption("Dica: se um dia não comparar, normalmente é porque o export veio com chave diferente (SKU/GTIN/N° peça).")

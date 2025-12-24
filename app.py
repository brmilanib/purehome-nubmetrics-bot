
import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime, date

DB_PATH = "snapshots.db"

# --------- CONFIG (suas regras) ----------
STOCK_ALERT_MAX = 20              # estoque <= 20
PRICE_ALERT_PCT = 10.0            # variação >= 10% (pra cima ou pra baixo)
# ----------------------------------------

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Upload diário de 3 planilhas → relatório de mudanças + oportunidades por estoque baixo (sem alertas externos).")

COMPETITORS = ["AUMA", "BAGATELLE", "PERFUMES_BHZ"]

# ---------- DB ----------
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
            SELECT snapshot_date, uploaded_at, data_json
            FROM snapshots
            WHERE competitor = ? AND snapshot_date < ?
            ORDER BY snapshot_date DESC, id DESC
            LIMIT 1
            """,
            (competitor, snapshot_date)
        ).fetchone()
    if not row:
        return None, None, None
    d, uploaded_at, data_json = row
    df = pd.read_json(data_json, orient="records")
    return d, uploaded_at, df

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

# ---------- NORMALIZAÇÃO ----------
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

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: CANON_MAP.get(c, c) for c in df.columns})

    # garante colunas esperadas (se não existir, cria)
    for c in WATCH_COLS + ["sku", "gtin", "n_peca", "titulo", "marca"]:
        if c not in df.columns:
            df[c] = pd.NA

    # tipos
    df["estoque"] = pd.to_numeric(df["estoque"], errors="coerce")
    df["preco_medio"] = pd.to_numeric(df["preco_medio"], errors="coerce")
    df["vendas_unid"] = pd.to_numeric(df["vendas_unid"], errors="coerce")

    # chave robusta
    def make_key(row):
        for k in ["sku", "gtin", "n_peca"]:
            v = row.get(k)
            if pd.notna(v) and str(v).strip() != "":
                return str(v).strip()
        # fallback (pior caso)
        t = row.get("titulo")
        if pd.notna(t) and str(t).strip() != "":
            return "TIT_" + str(t).strip()[:80]
        return None

    df["key"] = df.apply(make_key, axis=1)
    df = df[df["key"].notna()].copy()
    df["key"] = df["key"].astype(str)

    # remove duplicados por chave (pega o mais “completo”)
    df = df.sort_values(by=["key", "vendas_unid"], ascending=[True, False], na_position="last")
    df = df.drop_duplicates(subset=["key"], keep="first")

    # booleanos amigáveis
    for b in ["desconto", "frete_gratis", "full", "mercado_envios", "catalogo", "mercadopago"]:
        if b in df.columns:
            df[b] = df[b].astype(str).str.strip()

    return df

# ---------- DIFF ----------
def compute_diff(prev: pd.DataFrame, curr: pd.DataFrame):
    prev = prev.copy()
    curr = curr.copy()

    merged = curr.merge(prev, on="key", how="outer", suffixes=("_new", "_old"), indicator=True)

    # itens novos / removidos
    added = merged[merged["_merge"] == "left_only"].copy()
    removed = merged[merged["_merge"] == "right_only"].copy()
    both = merged[merged["_merge"] == "both"].copy()

    # mudanças por coluna
    changes = []
    for col in WATCH_COLS:
        newc = f"{col}_new"
        oldc = f"{col}_old"
        if newc not in both.columns or oldc not in both.columns:
            continue

        # compara string p/ não errar por NaN
        diff_mask = both[newc].astype(str) != both[oldc].astype(str)
        if diff_mask.any():
            temp = both.loc[diff_mask, ["key", oldc, newc, "titulo_new", "marca_new"]].copy()
            temp["campo"] = col
            temp.rename(columns={
                oldc: "antes",
                newc: "depois",
                "titulo_new": "titulo",
                "marca_new": "marca"
            }, inplace=True)
            changes.append(temp)

    changes_df = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame(
        columns=["key", "antes", "depois", "titulo", "marca", "campo"]
    )

    # métricas de preço (pct)
    if "preco_medio_new" in both.columns and "preco_medio_old" in both.columns:
        both["pct_change_preco"] = (both["preco_medio_new"] - both["preco_medio_old"]) / both["preco_medio_old"] * 100.0
    else:
        both["pct_change_preco"] = pd.NA

    # alertas (por concorrente)
    both["alerta_preco"] = both["pct_change_preco"].abs() >= PRICE_ALERT_PCT
    both["dir_preco"] = both["pct_change_preco"].apply(lambda x: "UP" if pd.notna(x) and x >= PRICE_ALERT_PCT else ("DOWN" if pd.notna(x) and x <= -PRICE_ALERT_PCT else "—"))

    curr_alert = curr[["key", "titulo", "marca", "preco_medio", "estoque", "desconto", "frete_gratis", "full", "tipo_publicacao", "vendas_unid"]].copy()
    curr_alert["alerta_estoque"] = curr_alert["estoque"].fillna(10**9) <= STOCK_ALERT_MAX

    return added, removed, changes_df, both[["key", "titulo_new", "marca_new", "preco_medio_old", "preco_medio_new", "pct_change_preco", "alerta_preco", "dir_preco"]], curr_alert

# ---------- UI ----------
st.subheader("1) Suba as 3 planilhas do dia")
col1, col2 = st.columns([1, 2])
with col1:
    snap_date = st.date_input("Data do snapshot", value=date.today())
snap_date_str = snap_date.isoformat()

uploaded_files = {}
u_cols = st.columns(3)
for i, comp in enumerate(COMPETITORS):
    with u_cols[i]:
        uploaded_files[comp] = st.file_uploader(f"{comp} (Excel Nubmetrics)", type=["xlsx", "xls"], key=f"up_{comp}")

process = st.button("Processar e gerar relatório", type="primary")

st.divider()
st.subheader("2) Resultado")

def show_card(title, df):
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True, height=320)

if process:
    # valida uploads
    missing = [c for c, f in uploaded_files.items() if f is None]
    if missing:
        st.error(f"Faltou upload: {', '.join(missing)}")
        st.stop()

    # processa cada concorrente
    per_comp = {}
    for comp, f in uploaded_files.items():
        df = pd.read_excel(f)
        df = normalize_df(df)

        prev_date, prev_uploaded_at, prev_df = load_last_snapshot_before(comp, snap_date_str)
        # salva snapshot atual
        save_snapshot(comp, snap_date_str, f.name, df)

        if prev_df is None:
            per_comp[comp] = {
                "prev_exists": False,
                "prev_date": None,
                "added": pd.DataFrame(),
                "removed": pd.DataFrame(),
                "changes": pd.DataFrame(),
                "price_moves": pd.DataFrame(),
                "curr": df
            }
        else:
            prev_df = normalize_df(prev_df)
            added, removed, changes, price_moves, curr_alert = compute_diff(prev_df, df)
            per_comp[comp] = {
                "prev_exists": True,
                "prev_date": prev_date,
                "added": added,
                "removed": removed,
                "changes": changes,
                "price_moves": price_moves,
                "curr_alert": curr_alert,
                "curr": df
            }

    # ---- INTELIGÊNCIA CROSS-CONCORRENTES ----
    # Junta variação de preço (quando existe prev)
    price_frames = []
    stock_frames = []
    for comp, d in per_comp.items():
        curr = d["curr"][["key", "titulo", "marca", "preco_medio", "estoque"]].copy()
        curr["concorrente"] = comp
        curr["estoque_alerta"] = curr["estoque"].fillna(10**9) <= STOCK_ALERT_MAX
        stock_frames.append(curr)

        if d.get("prev_exists"):
            pm = d["price_moves"].copy()
            pm.rename(columns={"titulo_new": "titulo", "marca_new": "marca"}, inplace=True)
            pm["concorrente"] = comp
            price_frames.append(pm)

    stock_all = pd.concat(stock_frames, ignore_index=True)
    # tabela de oportunidades por estoque (no dia)
    stock_summary = (stock_all
        .groupby("key", as_index=False)
        .agg(
            titulo=("titulo", "first"),
            marca=("marca", "first"),
            conc_criticos=("estoque_alerta", "sum"),
            estoque_min=("estoque", "min"),
            estoque_med=("estoque", "median"),
        )
    )
    stock_summary["oportunidade_ataque"] = stock_summary["conc_criticos"] >= 2

    # tabela de movimentos coordenados de preço (precisa de prev)
    if price_frames:
        price_all = pd.concat(price_frames, ignore_index=True)
        coord = (price_all
            .groupby("key", as_index=False)
            .agg(
                titulo=("titulo", "first"),
                marca=("marca", "first"),
                up_count=("dir_preco", lambda s: (s == "UP").sum()),
                down_count=("dir_preco", lambda s: (s == "DOWN").sum()),
                any_alert=("alerta_preco", "sum"),
            )
        )
        coord["movimento_coordenado"] = (coord["up_count"] >= 2) | (coord["down_count"] >= 2)
        coord["direcao"] = coord.apply(lambda r: "UP" if r["up_count"] >= 2 else ("DOWN" if r["down_count"] >= 2 else "—"), axis=1)
    else:
        coord = pd.DataFrame(columns=["key","titulo","marca","up_count","down_count","any_alert","movimento_coordenado","direcao"])

    # ---- AÇÕES DO DIA ----
    # Critérios:
    # - ATACAR: 2+ concorrentes com estoque <= 20
    # - DEFENDER: movimento coordenado de preço (2+ concorrentes com ±10%)
    # - MONITORAR: 1 concorrente com estoque crítico ou 1 com variação ±10%
    # - IGNORAR: sem sinal
    actions = stock_summary[["key","titulo","marca","conc_criticos","estoque_min","estoque_med","oportunidade_ataque"]].copy()
    actions = actions.merge(coord[["key","movimento_coordenado","direcao"]], on="key", how="left")
    actions["movimento_coordenado"] = actions["movimento_coordenado"].fillna(False)
    actions["direcao"] = actions["direcao"].fillna("—")

    # indicadores adicionais
    actions["status"] = "IGNORAR"
    actions.loc[actions["oportunidade_ataque"], "status"] = "ATACAR"
    actions.loc[~actions["oportunidade_ataque"] & actions["movimento_coordenado"], "status"] = "DEFENDER"

    # MONITORAR: casos intermediários
    actions.loc[(actions["conc_criticos"] == 1) & (actions["status"] == "IGNORAR"), "status"] = "MONITORAR"
    actions.loc[(actions["movimento_coordenado"] == False) & (actions["status"] == "IGNORAR") & (actions["direcao"] != "—"), "status"] = "MONITORAR"

    def suggest(row):
        if row["status"] == "ATACAR":
            return "Mercado com estoque apertado (2+ concorrentes ≤20). Se você tiver estoque: segure preço ou suba 3–5% e destaque entrega/FULL."
        if row["status"] == "DEFENDER":
            if row["direcao"] == "DOWN":
                return "Queda coordenada (2+ concorrentes -10%). Avaliar ajuste de preço OU reforçar valor (FULL, combo, descrição) antes de baixar."
            if row["direcao"] == "UP":
                return "Alta coordenada (2+ concorrentes +10%). Você pode subir preço sem medo se tiver estoque."
            return "Movimento coordenado. Monitorar e decidir."
        if row["status"] == "MONITORAR":
            return "Sinal isolado (estoque ou preço). Não reage de imediato — observa mais 1 dia."
        return "Sem sinal forte hoje."

    actions["acao_sugerida"] = actions.apply(suggest, axis=1)

    # Ordena para decisão
    order_map = {"ATACAR": 0, "DEFENDER": 1, "MONITORAR": 2, "IGNORAR": 3}
    actions["ord"] = actions["status"].map(order_map).fillna(9)
    actions = actions.sort_values(["ord","conc_criticos","estoque_min"]).drop(columns=["ord"])

    # ---- Mostra painéis ----
    st.markdown("## Painel executivo (Ações do dia)")
    st.dataframe(actions, use_container_width=True, height=420)

    st.download_button(
        "Baixar Ações do Dia (CSV)",
        actions.to_csv(index=False).encode("utf-8"),
        file_name=f"acoes_do_dia_{snap_date_str}.csv",
        mime="text/csv"
    )

    st.divider()
    st.markdown("## Detalhe por concorrente (mudanças D-1 → D-0)")
    for comp in COMPETITORS:
        d = per_comp[comp]
        st.markdown(f"### {comp}")
        if not d.get("prev_exists"):
            st.info("Primeiro snapshot desse concorrente. Amanhã já sai comparação D-1 → D-0.")
            st.dataframe(d["curr"].head(30), use_container_width=True)
            continue

        c1, c2, c3 = st.columns(3)
        c1.metric("Novos itens", len(d["added"]))
        c2.metric("Itens removidos", len(d["removed"]))
        c3.metric("Mudanças detectadas", len(d["changes"]))

        tabs = st.tabs(["Mudanças", "Novos", "Removidos", "Preço (±10%)", "Estoque (≤20)"])

        with tabs[0]:
            show_card("Mudanças (campo / antes / depois)", d["changes"])
        with tabs[1]:
            show_card("Novos itens", d["added"])
        with tabs[2]:
            show_card("Itens removidos", d["removed"])
        with tabs[3]:
            pm = d["price_moves"].copy()
            pm.rename(columns={"titulo_new": "titulo", "marca_new": "marca"}, inplace=True)
            pm = pm[pm["alerta_preco"] == True].sort_values("pct_change_preco")
            show_card("Alertas de preço (>=10%)", pm)
        with tabs[4]:
            ca = d["curr_alert"].copy()
            ca = ca[ca["alerta_estoque"] == True].sort_values("estoque")
            show_card("Alertas de estoque (<=20)", ca)

    st.success("Pronto. Amanhã é só subir os 3 Excels de novo que você vai ter o comparativo + decisões do dia.")

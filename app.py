import os
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date

DB_PATH = os.environ.get("DB_PATH", "/tmp/snapshots.db")

# --------- CONFIG (suas regras) ----------
STOCK_ALERT_MAX = 20              # estoque <= 20
PRICE_ALERT_PCT = 10.0            # variação >= 10% (pra cima ou pra baixo)
# ----------------------------------------

st.set_page_config(page_title="PureHome • Monitor Nubmetrics", layout="wide")
st.title("PureHome • Monitor de Concorrentes (Nubmetrics)")
st.caption("Upload diário de 3 planilhas → BI clean de ATAQUE + detalhes (opcional).")

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

    # garante colunas esperadas
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

    # booleanos amigáveis (mantém como string "Sim"/"Não" etc)
    for b in ["desconto", "frete_gratis", "full", "mercado_envios", "catalogo", "mercadopago"]:
        if b in df.columns:
            df[b] = df[b].astype(str).str.strip()

    return df

# ---------- DIFF ----------
def compute_diff(prev: pd.DataFrame, curr: pd.DataFrame):
    prev = prev.copy()
    curr = curr.copy()

    merged = curr.merge(prev, on="key", how="outer", suffixes=("_new", "_old"), indicator=True)

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

        diff_mask = both[newc].astype(str) != both[oldc].astype(str)
        if diff_mask.any():
            temp = both.loc[diff_mask, ["key", oldc, newc, "titulo_new", "marca_new"]].copy()
            temp["campo"] = col
            temp.rename(columns={oldc: "antes", newc: "depois", "titulo_new": "titulo", "marca_new": "marca"}, inplace=True)
            changes.append(temp)

    changes_df = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame(
        columns=["key", "antes", "depois", "titulo", "marca", "campo"]
    )

    # métricas de preço (pct) com proteção contra divisão por zero
    both["pct_change_preco"] = pd.NA
    if "preco_medio_new" in both.columns and "preco_medio_old" in both.columns:
        denom = both["preco_medio_old"].copy()
        denom = denom.where(denom.notna() & (denom != 0), pd.NA)
        both["pct_change_preco"] = (both["preco_medio_new"] - both["preco_medio_old"]) / denom * 100.0

    both["alerta_preco"] = both["pct_change_preco"].abs() >= PRICE_ALERT_PCT
    both["dir_preco"] = both["pct_change_preco"].apply(
        lambda x: "UP" if pd.notna(x) and x >= PRICE_ALERT_PCT else ("DOWN" if pd.notna(x) and x <= -PRICE_ALERT_PCT else "—")
    )

    # estoque delta + flag_ataque (o que você pediu)
    if "estoque_new" in both.columns and "estoque_old" in both.columns:
        both["delta_estoque"] = both["estoque_new"] - both["estoque_old"]
    else:
        both["delta_estoque"] = pd.NA

    both["flag_ataque"] = (
        (both["delta_estoque"].fillna(0) < 0) &
        (both["estoque_new"].fillna(10**9) <= STOCK_ALERT_MAX) &
        (both["pct_change_preco"].fillna(-10**9) >= PRICE_ALERT_PCT)
    )

    # snapshot atual p/ alertas de estoque
    curr_alert = curr[["key", "titulo", "marca", "preco_medio", "estoque", "desconto", "frete_gratis", "full", "tipo_publicacao", "vendas_unid"]].copy()
    curr_alert["alerta_estoque"] = curr_alert["estoque"].fillna(10**9) <= STOCK_ALERT_MAX

    price_moves = both[[
        "key",
        "titulo_new", "marca_new",
        "preco_medio_old", "preco_medio_new",
        "pct_change_preco", "alerta_preco", "dir_preco",
        "estoque_old", "estoque_new",
        "delta_estoque", "flag_ataque"
    ]].copy()

    return added, removed, changes_df, price_moves, curr_alert

# ---------- UI ----------
st.subheader("1) Suba as 3 planilhas do dia")
with st.container():
    snap_date = st.date_input("Data do snapshot", value=date.today())
snap_date_str = snap_date.isoformat()

uploaded_files = {}
u_cols = st.columns(3)
for i, comp in enumerate(COMPETITORS):
    with u_cols[i]:
        uploaded_files[comp] = st.file_uploader(f"{comp} (Excel Nubmetrics)", type=["xlsx", "xls"], key=f"up_{comp}")

process = st.button("Processar e gerar BI", type="primary")

st.divider()
st.subheader("2) Resultado")

def show_card(title, df):
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True, height=360)

if process:
    missing = [c for c, f in uploaded_files.items() if f is None]
    if missing:
        st.error(f"Faltou upload: {', '.join(missing)}")
        st.stop()

    per_comp = {}
    for comp, f in uploaded_files.items():
        df = pd.read_excel(f)
        df = normalize_df(df)

        prev_date, prev_uploaded_at, prev_df = load_last_snapshot_before(comp, snap_date_str)
        save_snapshot(comp, snap_date_str, f.name, df)

        if prev_df is None:
            per_comp[comp] = {
                "prev_exists": False,
                "prev_date": None,
                "added": pd.DataFrame(),
                "removed": pd.DataFrame(),
                "changes": pd.DataFrame(),
                "price_moves": pd.DataFrame(),
                "curr_alert": pd.DataFrame(),
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

    # ---- CROSS ----
    price_frames = []
    for comp, d in per_comp.items():
        if d.get("prev_exists"):
            pm = d["price_moves"].copy()
            pm["concorrente"] = comp
            price_frames.append(pm)

    # ---------- BI CLEAN: ATAQUES ----------
    st.markdown("## 🎯 ATAQUES do dia (estoque caiu + estoque ≤20 + preço subiu ≥10%)")

    if not price_frames:
        st.warning("Ainda não existe comparação (primeiro dia). Suba o dia anterior e depois este dia.")
    else:
        price_all = pd.concat(price_frames, ignore_index=True)
        ataques = price_all[price_all["flag_ataque"] == True].copy()

        if ataques.empty:
            st.info("Nenhum ATAQUE detectado hoje com as regras atuais.")
        else:
            # resumo por SKU (e quantos concorrentes deram o mesmo sinal)
            resumo = (
                ataques
                .groupby(["key", "titulo_new", "marca_new"], as_index=False)
                .agg(
                    conc_ataque=("concorrente", "nunique"),
                    concorrentes=("concorrente", lambda x: ", ".join(sorted(set(x)))),
                    estoque_min=("estoque_new", "min"),
                    delta_estoque_min=("delta_estoque", "min"),
                    preco_antigo_min=("preco_medio_old", "min"),
                    preco_novo_max=("preco_medio_new", "max"),
                    pct_up_max=("pct_change_preco", "max"),
                )
                .sort_values(["conc_ataque", "estoque_min", "pct_up_max"], ascending=[False, True, False])
            )

            exigir_2 = st.checkbox("Mostrar só se 2+ concorrentes deram sinal (mais confiável)", value=True)
            if exigir_2:
                resumo = resumo[resumo["conc_ataque"] >= 2]

            resumo = resumo.rename(columns={"titulo_new": "titulo", "marca_new": "marca"})

            c1, c2, c3 = st.columns(3)
            c1.metric("SKUs em ATAQUE", len(resumo))
            c2.metric("Concorrentes monitorados", len(COMPETITORS))
            c3.metric("Regras", f"Estoque ≤{STOCK_ALERT_MAX} + Preço ≥{PRICE_ALERT_PCT}%")

            st.dataframe(
                resumo[[
                    "key","titulo","marca",
                    "conc_ataque","concorrentes",
                    "estoque_min","delta_estoque_min",
                    "preco_antigo_min","preco_novo_max","pct_up_max"
                ]],
                use_container_width=True,
                height=520
            )

            st.download_button(
                "Baixar ATAQUES (CSV)",
                resumo.to_csv(index=False).encode("utf-8"),
                file_name=f"ataques_{snap_date_str}.csv",
                mime="text/csv"
            )

    # ---------- Detalhes (opcional) ----------
    with st.expander("📌 Ver detalhes completos por concorrente (opcional)"):
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

            tabs = st.tabs(["Mudanças", "Novos", "Removidos", "Preço (±10%)", "Estoque (≤20)", "Ataques (esse concorrente)"])

            with tabs[0]:
                show_card("Mudanças (campo / antes / depois)", d["changes"])
            with tabs[1]:
                show_card("Novos itens", d["added"])
            with tabs[2]:
                show_card("Itens removidos", d["removed"])
            with tabs[3]:
                pm = d["price_moves"].copy()
                pm = pm[pm["alerta_preco"] == True].sort_values("pct_change_preco")
                pm = pm.rename(columns={"titulo_new": "titulo", "marca_new": "marca"})
                show_card("Alertas de preço (>=10%)", pm)
            with tabs[4]:
                ca = d["curr_alert"].copy()
                ca = ca[ca["alerta_estoque"] == True].sort_values("estoque")
                show_card("Alertas de estoque (<=20)", ca)
            with tabs[5]:
                atk = d["price_moves"].copy()
                atk = atk[atk["flag_ataque"] == True].copy()
                atk = atk.rename(columns={"titulo_new": "titulo", "marca_new": "marca"})
                show_card("ATAQUES (estoque caiu + estoque ≤20 + preço subiu ≥10%)", atk)

    st.success("Pronto. Agora o topo mostra só o que você deve ATACAR. O resto ficou escondido nos detalhes.")

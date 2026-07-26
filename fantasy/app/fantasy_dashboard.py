"""
Fantasy Premier League Dashboard — predicciones + optimizador.

Ejecutar:
    venv/bin/streamlit run fantasy/app/fantasy_dashboard.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="FPL Predictor", layout="wide", page_icon="⚽")

# ═══════════════ DATOS ═══════════════
@st.cache_data
def load_all():
    boot = json.load(open(ROOT / "data/raw/fpl_bootstrap.json"))
    df_hist = pd.read_parquet(ROOT / "data/processed/fpl_features.parquet")

    last_per_player = df_hist.sort_values(["name", "_season", "GW"]).groupby("name").tail(1).set_index("name")
    teams_by_id = {t["id"]: t["name"] for t in boot["teams"]}
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    rows = []
    for p in boot["elements"]:
        name = p["web_name"]
        hist = None
        if name in last_per_player.index:
            hist = last_per_player.loc[name]
            if isinstance(hist, pd.DataFrame): hist = hist.iloc[0]
        rows.append({
            "id": p["id"], "name": name,
            "team": teams_by_id.get(p["team"], f"team_{p['team']}"),
            "position": pos_map.get(p["element_type"], "UNK"),
            "value_M": p["now_cost"] / 10.0,
            "total_points": p["total_points"],
            "form": float(p.get("form", "0") or 0),
            "selected_pct": float(p.get("selected_by_percent", "0") or 0),
            "goals_scored": p.get("goals_scored", 0),
            "assists": p.get("assists", 0),
            "clean_sheets": p.get("clean_sheets", 0),
            "minutes": p.get("minutes", 0),
            "hist_row": hist,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def load_model():
    return (
        lgb.Booster(model_file=str(ROOT / "models/fpl_predictor_lgb.txt")),
        json.load(open(ROOT / "models/feature_cols.json")),
    )


@st.cache_data
def predict_all(_df):  # _df para evitar hash
    model, feature_cols = load_model()
    rows = []
    for _, r in _df.iterrows():
        hist = r["hist_row"]
        feat = {}
        for c in feature_cols:
            if hist is not None and c in hist.index and pd.notna(hist[c]):
                feat[c] = float(hist[c])
            elif c.startswith("pos_"):
                feat[c] = 1.0 if c == f"pos_{r['position']}" else 0.0
            elif c == "value_M":
                feat[c] = r["value_M"]
            elif c == "was_home":
                feat[c] = 0.5
            else:
                feat[c] = 0.0
        rows.append(feat)
    X = pd.DataFrame(rows)[feature_cols].astype(np.float32).values
    predictions = np.clip(model.predict(X), 0, None)
    df = _df.copy()
    df["pred_points"] = predictions
    df["ppm"] = df["pred_points"] / df["value_M"].where(df["value_M"] > 0, 1)
    return df


with st.spinner("Cargando datos FPL..."):
    df_players = load_all()
    df_players = predict_all(df_players)

# ═══════════════ SIDEBAR ═══════════════
st.sidebar.title("⚽ FPL Predictor")
st.sidebar.markdown(f"**{len(df_players)}** jugadores FPL 26/27")
st.sidebar.markdown(f"Modelo: LightGBM entrenado sobre 7 temporadas históricas")
st.sidebar.markdown("---")

page = st.sidebar.radio("Sección:", [
    "🏠 Home",
    "🎯 Top predicciones",
    "💰 Value picks",
    "🏆 Optimizador plantilla",
    "🔍 Buscar jugador",
    "📊 Comparar 2 jugadores",
])

# ═══════════════ PÁGINA HOME ═══════════════
if page == "🏠 Home":
    st.title("⚽ Fantasy Premier League — Predictor")
    st.markdown("Modelo predictivo entrenado sobre **185.964 registros** de 7 temporadas FPL histórico.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores actuales", len(df_players))
    c2.metric("Presupuesto FPL", "£100.0M")
    c3.metric("MAE modelo", "1.05 pts", "-9% vs xP oficial FPL")
    c4.metric("Temporadas train", "5 (19/20 - 23/24)")

    st.markdown("---")

    # Distribución precios por posición
    st.subheader("💰 Distribución precios por posición")
    fig = px.box(df_players, x="position", y="value_M", points="all",
                 color="position", hover_data=["name", "team"])
    fig.update_layout(yaxis_title="Precio (£M)")
    st.plotly_chart(fig, use_container_width=True)

    # Top 20 por puntos totales (25/26 acumulados)
    st.subheader("🏆 Top 20 puntos acumulados 25/26")
    top20 = df_players.nlargest(20, "total_points")[
        ["name", "team", "position", "value_M", "total_points", "form", "selected_pct", "pred_points"]
    ]
    st.dataframe(top20.reset_index(drop=True), use_container_width=True)

# ═══════════════ PÁGINA TOP PREDICCIONES ═══════════════
elif page == "🎯 Top predicciones":
    st.title("🎯 Top predicciones puntos próxima GW")

    pos_filter = st.selectbox("Filtrar posición:", ["Todas", "GK", "DEF", "MID", "FWD"])
    n_show = st.slider("Nº mostrados:", 5, 50, 20)

    df_f = df_players if pos_filter == "Todas" else df_players[df_players["position"] == pos_filter]
    top = df_f.nlargest(n_show, "pred_points")

    fig = px.bar(top, x="pred_points", y="name", orientation="h", color="position",
                 hover_data=["team", "value_M", "total_points"],
                 text="pred_points",
                 title=f"Top {n_show} puntos esperados próxima GW ({pos_filter})",
                 height=max(400, n_show*25))
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detalle")
    st.dataframe(top[["name", "team", "position", "value_M", "pred_points", "form", "total_points", "selected_pct"]].reset_index(drop=True),
                 use_container_width=True)

# ═══════════════ PÁGINA VALUE PICKS ═══════════════
elif page == "💰 Value picks":
    st.title("💰 Value picks — puntos esperados por £M")

    pos_filter = st.selectbox("Filtrar posición:", ["Todas", "GK", "DEF", "MID", "FWD"], key="vp_pos")
    max_price = st.slider("Precio máximo (£M):", 4.0, 15.0, 10.0, step=0.5)

    df_f = df_players[df_players["value_M"] <= max_price].copy()
    if pos_filter != "Todas":
        df_f = df_f[df_f["position"] == pos_filter]

    top = df_f.nlargest(20, "ppm")
    st.markdown(f"### Top 20 mejor ratio puntos/£M (≤ £{max_price:.1f}M, {pos_filter})")

    fig = px.scatter(df_f, x="value_M", y="pred_points", color="position",
                     hover_data=["name", "team", "form"], size="ppm",
                     labels={"value_M": "Precio (£M)", "pred_points": "Puntos esperados"},
                     title="Value picks (tamaño = ratio pts/£M)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top[["name", "team", "position", "value_M", "pred_points", "ppm", "selected_pct"]].reset_index(drop=True),
                 use_container_width=True)

# ═══════════════ PÁGINA OPTIMIZADOR ═══════════════
elif page == "🏆 Optimizador plantilla":
    st.title("🏆 Optimizador plantilla FPL")
    st.markdown("Elige presupuesto y el modelo te sugiere los **15 jugadores óptimos**.")

    budget = st.slider("Presupuesto (£M):", 80.0, 105.0, 100.0, step=0.5)

    if st.button("▶️ Optimizar plantilla", type="primary"):
        with st.spinner("Optimizando (Linear Programming)..."):
            df_opt = df_players[df_players["value_M"] > 0].dropna(subset=["pred_points"]).reset_index(drop=True)
            n = len(df_opt)
            prob = pulp.LpProblem("fpl", pulp.LpMaximize)
            x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
            prob += pulp.lpSum(x[i] * df_opt.loc[i, "pred_points"] for i in range(n))
            prob += pulp.lpSum(x[i] * df_opt.loc[i, "value_M"] for i in range(n)) <= budget
            prob += pulp.lpSum(x[i] for i in range(n)) == 15
            prob += pulp.lpSum(x[i] for i in range(n) if df_opt.loc[i, "position"] == "GK") == 2
            prob += pulp.lpSum(x[i] for i in range(n) if df_opt.loc[i, "position"] == "DEF") == 5
            prob += pulp.lpSum(x[i] for i in range(n) if df_opt.loc[i, "position"] == "MID") == 5
            prob += pulp.lpSum(x[i] for i in range(n) if df_opt.loc[i, "position"] == "FWD") == 3
            for team in df_opt["team"].unique():
                prob += pulp.lpSum(x[i] for i in range(n) if df_opt.loc[i, "team"] == team) <= 3
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            selected = [i for i in range(n) if x[i].value() == 1]
            team = df_opt.loc[selected].reset_index(drop=True)

        total_cost = team["value_M"].sum()
        total_pts = team["pred_points"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Coste total", f"£{total_cost:.1f}M", f"/{budget:.0f}M")
        c2.metric("⭐ Puntos esperados", f"{total_pts:.2f}")
        c3.metric("📊 PPM promedio", f"{total_pts/total_cost:.2f}")

        st.markdown("### 👥 Plantilla óptima (15 jugadores)")
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = team[team["position"] == pos].sort_values("pred_points", ascending=False)
            st.markdown(f"#### {pos} ({len(sub)})")
            st.dataframe(sub[["name", "team", "value_M", "pred_points", "form"]].reset_index(drop=True),
                         use_container_width=True)

        # Distribución por club
        st.markdown("### 🏟️ Distribución por club")
        club_count = team["team"].value_counts().reset_index()
        club_count.columns = ["Club", "Jugadores"]
        fig = px.bar(club_count, x="Club", y="Jugadores", color="Club",
                     title="Plantilla por club (máximo 3 por regla FPL)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════ PÁGINA BUSCAR JUGADOR ═══════════════
elif page == "🔍 Buscar jugador":
    st.title("🔍 Buscar jugador")

    search = st.text_input("Nombre jugador (parcial):", "Haaland")
    matches = df_players[df_players["name"].str.contains(search, case=False, na=False)]

    if len(matches) == 0:
        st.warning("Sin resultados")
    else:
        st.markdown(f"### {len(matches)} coincidencias:")
        for _, r in matches.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1, 1, 1.5])
            c1.markdown(f"**{r['name']}** — {r['team']}")
            c2.metric("Posición", r['position'])
            c3.metric("£M", f"{r['value_M']:.1f}")
            c4.metric("Total pts", f"{r['total_points']}")
            c5.metric("Pred próx GW", f"{r['pred_points']:.2f}")

# ═══════════════ PÁGINA COMPARAR 2 JUGADORES ═══════════════
elif page == "📊 Comparar 2 jugadores":
    st.title("📊 Comparar 2 jugadores")

    names = sorted(df_players["name"].tolist())
    c1, c2 = st.columns(2)
    p1_name = c1.selectbox("Jugador 1:", names, index=names.index("Haaland") if "Haaland" in names else 0)
    p2_name = c2.selectbox("Jugador 2:", names, index=names.index("Saka") if "Saka" in names else 1)

    p1 = df_players[df_players["name"] == p1_name].iloc[0]
    p2 = df_players[df_players["name"] == p2_name].iloc[0]

    # Tabla comparativa
    metrics = ["value_M", "pred_points", "ppm", "form", "total_points",
               "goals_scored", "assists", "clean_sheets", "minutes", "selected_pct"]
    labels = ["Precio (£M)", "Puntos esperados próx GW", "Ratio pts/£M",
              "Forma (últimos 4)", "Puntos totales 25/26", "Goles", "Asistencias",
              "Clean sheets", "Minutos", "% ownership"]
    df_comp = pd.DataFrame({
        "Métrica": labels,
        p1_name: [p1[m] for m in metrics],
        p2_name: [p2[m] for m in metrics],
    })
    st.dataframe(df_comp, use_container_width=True)

    # Radar chart
    st.markdown("### 📊 Radar comparativo")
    categories = ["Puntos esp.", "Form", "Total pts", "Goles", "Asist.", "Minutos"]
    v1 = [p1["pred_points"], p1["form"], p1["total_points"]/50,
          p1["goals_scored"], p1["assists"], p1["minutes"]/500]
    v2 = [p2["pred_points"], p2["form"], p2["total_points"]/50,
          p2["goals_scored"], p2["assists"], p2["minutes"]/500]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1, theta=categories, fill="toself", name=p1_name))
    fig.add_trace(go.Scatterpolar(r=v2, theta=categories, fill="toself", name=p2_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
    st.plotly_chart(fig, use_container_width=True)

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
    "👤 Mi Equipo (asistente)",
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

# ═══════════════ PÁGINA MI EQUIPO — ASISTENTE PRÁCTICO ═══════════════
elif page == "👤 Mi Equipo (asistente)":
    st.title("👤 Mi Equipo — asistente semanal FPL")
    st.markdown("Configura tu equipo actual y el modelo te dice qué hacer esta jornada.")

    # Método de carga
    load_method = st.radio("¿Cómo cargas tu equipo?", ["🔗 Automático (Team ID FPL)", "✍️ Manual"], horizontal=True)

    my_team = None
    my_bank = 0.0
    my_free_transfers = 1

    if load_method == "🔗 Automático (Team ID FPL)":
        st.info("Tu Team ID está en la URL cuando entras a 'Points' en fantasy.premierleague.com — algo como `/entry/123456/`")
        team_id = st.number_input("Team ID FPL:", min_value=1, value=1, step=1)
        if st.button("📥 Cargar mi equipo"):
            import requests
            try:
                # Endpoint público FPL
                r = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/", timeout=15)
                if r.status_code == 200:
                    entry = r.json()
                    current_gw = entry.get("current_event", 1)
                    my_bank = entry.get("last_deadline_bank", 0) / 10
                    my_free_transfers = entry.get("last_deadline_total_transfers", 0)
                    st.success(f"✅ {entry.get('name', 'Equipo')} — GW{current_gw} — Banco £{my_bank:.1f}M")

                    # Cargar picks
                    r2 = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{current_gw}/picks/", timeout=15)
                    if r2.status_code == 200:
                        picks = r2.json()["picks"]
                        player_ids = [p["element"] for p in picks]
                        my_team_df = df_players[df_players["id"].isin(player_ids)].copy()
                        st.session_state["my_team"] = my_team_df
                        st.session_state["my_bank"] = my_bank
                        st.session_state["my_free_transfers"] = 1
                else:
                    st.error(f"No se pudo cargar Team ID {team_id} (HTTP {r.status_code})")
            except Exception as e:
                st.error(f"Error: {e}")

    else:  # Manual
        st.markdown("### Selecciona tus 15 jugadores")
        c1, c2, c3, c4 = st.columns(4)

        # Filtrar por posición para selección
        gks = sorted(df_players[df_players["position"] == "GK"]["name"].tolist())
        defs = sorted(df_players[df_players["position"] == "DEF"]["name"].tolist())
        mids = sorted(df_players[df_players["position"] == "MID"]["name"].tolist())
        fwds = sorted(df_players[df_players["position"] == "FWD"]["name"].tolist())

        with c1:
            st.markdown("**Porteros (2)**")
            sel_gks = st.multiselect("GK:", gks, default=gks[:2] if len(gks) >= 2 else [])
        with c2:
            st.markdown("**Defensas (5)**")
            sel_defs = st.multiselect("DEF:", defs, default=defs[:5] if len(defs) >= 5 else [])
        with c3:
            st.markdown("**Mediocampistas (5)**")
            sel_mids = st.multiselect("MID:", mids, default=mids[:5] if len(mids) >= 5 else [])
        with c4:
            st.markdown("**Delanteros (3)**")
            sel_fwds = st.multiselect("FWD:", fwds, default=fwds[:3] if len(fwds) >= 3 else [])

        selected_names = sel_gks + sel_defs + sel_mids + sel_fwds
        if len(selected_names) == 15:
            my_team_df = df_players[df_players["name"].isin(selected_names)].copy()
            st.session_state["my_team"] = my_team_df

            colb1, colb2 = st.columns(2)
            with colb1:
                bank = st.number_input("💰 Banco disponible (£M):", 0.0, 20.0, 2.0, step=0.1)
                st.session_state["my_bank"] = bank
            with colb2:
                ft = st.number_input("🔄 Transferencias gratis restantes:", 0, 5, 1)
                st.session_state["my_free_transfers"] = ft

    # ═══════════════ Análisis del equipo ═══════════════
    if "my_team" in st.session_state:
        my_team_df = st.session_state["my_team"]
        my_bank = st.session_state.get("my_bank", 0)
        my_free_transfers = st.session_state.get("my_free_transfers", 1)

        st.markdown("---")
        st.markdown("## 📋 Tu equipo actual")

        # Métricas globales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coste plantilla", f"£{my_team_df['value_M'].sum():.1f}M")
        c2.metric("Banco disponible", f"£{my_bank:.1f}M")
        c3.metric("Presupuesto total", f"£{my_team_df['value_M'].sum() + my_bank:.1f}M")
        c4.metric("Transferencias gratis", my_free_transfers)

        # Tabla equipo con predicciones
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = my_team_df[my_team_df["position"] == pos].sort_values("pred_points", ascending=False)
            st.markdown(f"#### {pos} ({len(sub)})")
            st.dataframe(sub[["name", "team", "value_M", "pred_points", "form", "selected_pct"]].reset_index(drop=True),
                         use_container_width=True)

        st.markdown("---")

        # ═══════════════ RECOMENDACIONES ═══════════════
        st.markdown("## 🎯 Recomendaciones semanales")

        # 1. Capitán óptimo
        st.markdown("### ⭐ 1. Capitán de la semana (puntos ×2)")
        top3_capt = my_team_df.nlargest(3, "pred_points")[["name", "team", "position", "pred_points"]]
        c1, c2, c3 = st.columns(3)
        for col, (_, r) in zip([c1, c2, c3], top3_capt.iterrows()):
            with col:
                st.metric(f"{'👑 ' if _ == top3_capt.index[0] else ''}{r['name']}",
                          f"{r['pred_points']:.2f} pts",
                          f"×2 = {r['pred_points']*2:.2f} pts")

        # 2. XI titular óptimo (de tus 15)
        st.markdown("### 🏟️ 2. XI titular óptimo (de tus 15 jugadores)")
        # LP simple: elegir 11 mejores respetando formación válida
        d = my_team_df.reset_index(drop=True)
        n = len(d)
        prob = pulp.LpProblem("xi", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        prob += pulp.lpSum(x[i] * d.loc[i, "pred_points"] for i in range(n))
        prob += pulp.lpSum(x[i] for i in range(n)) == 11
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "GK") == 1
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "DEF") >= 3
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "DEF") <= 5
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "MID") >= 3
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "MID") <= 5
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "FWD") >= 1
        prob += pulp.lpSum(x[i] for i in range(n) if d.loc[i, "position"] == "FWD") <= 3
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        selected = [i for i in range(n) if x[i].value() == 1]
        xi = d.loc[selected]
        bench = d.drop(selected)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**Titulares (11)** — pts esperados: {xi['pred_points'].sum():.2f}")
            for pos in ["GK", "DEF", "MID", "FWD"]:
                sub = xi[xi["position"] == pos].sort_values("pred_points", ascending=False)
                if len(sub) > 0:
                    st.markdown(f"*{pos}*: " + ", ".join(sub["name"].tolist()))
        with c2:
            st.markdown(f"**Banquillo (4)** — {bench['pred_points'].sum():.2f} pts")
            for _, r in bench.sort_values("pred_points", ascending=False).iterrows():
                st.write(f"• {r['name']} ({r['position']})")

        # 3. Transferencias sugeridas
        st.markdown("### 🔄 3. Transferencias sugeridas")
        st.markdown(f"""
        Regla: cada transferencia adicional a la {my_free_transfers} gratis cuesta **−4 puntos** (hit).
        Solo se sugieren swaps con **delta esperado > 4 puntos** (superan el hit).
        """)

        # Para cada jugador tuyo, buscar mejor reemplazo en su posición dentro de presupuesto
        max_transfers = st.slider("Máx sugerencias a mostrar:", 1, 10, 5)
        my_names = set(my_team_df["name"])
        my_team_ids = set(my_team_df["id"])

        suggestions = []
        for _, mine in my_team_df.iterrows():
            # Presupuesto para reemplazo = venta jugador + banco
            budget_for_swap = mine["value_M"] + my_bank
            # Candidatos: mismos position, dentro de presupuesto, no en mi equipo, no exceder 3 por club actuales
            club_count = my_team_df["team"].value_counts()
            def can_add(team_name):
                current = club_count.get(team_name, 0)
                if mine["team"] == team_name:
                    return True  # sustituimos por otro del mismo club
                return current < 3

            candidates = df_players[
                (df_players["position"] == mine["position"]) &
                (df_players["value_M"] <= budget_for_swap) &
                (~df_players["name"].isin(my_names)) &
                (df_players["team"].apply(can_add))
            ]
            if len(candidates) == 0: continue
            best = candidates.nlargest(1, "pred_points").iloc[0]
            delta = best["pred_points"] - mine["pred_points"]
            if delta > 0:
                suggestions.append({
                    "Vender": mine["name"], "Team out": mine["team"], "Pos": mine["position"],
                    "Precio venta (£M)": mine["value_M"], "Pts esp. actual": mine["pred_points"],
                    "Comprar": best["name"], "Team in": best["team"],
                    "Precio compra (£M)": best["value_M"], "Pts esp. nuevo": best["pred_points"],
                    "Delta pts": delta, "Coste (£M)": best["value_M"] - mine["value_M"],
                })

        if suggestions:
            df_sug = pd.DataFrame(suggestions).sort_values("Delta pts", ascending=False).head(max_transfers)
            # Marcar si hit vale la pena
            def worth_hit(row_idx, delta):
                if row_idx < my_free_transfers:
                    return "✅ Gratis"
                elif delta > 4:
                    return "✅ Merece hit −4"
                else:
                    return "⚠️ NO merece hit"
            df_sug["Decisión"] = [worth_hit(i, d) for i, d in enumerate(df_sug["Delta pts"])]
            st.dataframe(df_sug.round(2).reset_index(drop=True), use_container_width=True)

            # Resumen
            gratis = df_sug.head(my_free_transfers)
            st.success(f"""
            **Recomendación**: aprovecha tus **{my_free_transfers} transferencia(s) gratis**:
            {' + '.join([f'{r["Vender"]}→{r["Comprar"]} (+{r["Delta pts"]:.2f} pts)' for _, r in gratis.iterrows()])}
            """)
        else:
            st.info("Tu equipo ya está optimizado — no hay mejoras claras dentro del presupuesto.")

        # 4. Chip advice
        st.markdown("### 🎴 4. Consejo de chips")
        avg_pts_team = my_team_df["pred_points"].mean()
        max_capt_pts = my_team_df["pred_points"].max()
        st.markdown(f"""
        - **Triple Captain**: usar cuando tu mejor jugador tiene fixture MUY favorable.
          Tu top actual = {top3_capt.iloc[0]['name']} con {max_capt_pts:.2f} pts.
          Triple = {max_capt_pts*3:.2f} pts vs Capitán normal {max_capt_pts*2:.2f} = **+{max_capt_pts:.2f} pts extras**.
          {'✅ Considera usar' if max_capt_pts > 6 else '⏸️ Espera mejor fixture'}
        - **Bench Boost**: usar cuando tu banquillo tiene puntos esperados altos.
          Tu bench actual = {bench['pred_points'].sum():.2f} pts. {'✅ Considera' if bench['pred_points'].sum() > 8 else '⏸️ Bench bajo, espera'}
        - **Wildcard**: usar cuando >6 transferencias serían necesarias (rehacer plantilla).
          Sugerencias actuales: {len(suggestions)}. {'✅ Considera wildcard' if len(suggestions) > 6 else '⏸️ No es necesario aún'}
        - **Free Hit**: usar para 1 GW específica con doble jornada o muchos jugadores sin fixture.
        """)


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

"""
LaLiga Fantasy Dashboard — baseline por equipos.

Sin datos individuales por jugador (scraping pendiente), este dashboard usa
la fuerza ofensiva/defensiva de cada equipo LaLiga para orientar decisiones
de fichaje según posición y club del jugador.

Ejecutar:
    venv/bin/streamlit run fantasy_laliga/app/laliga_dashboard.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="LaLiga Fantasy", layout="wide", page_icon="🇪🇸")

# ═══════════════ DATOS ═══════════════
@st.cache_data
def load_team_strengths():
    p = ROOT / "data/processed/team_strengths.csv"
    if not p.exists():
        st.error(f"Ejecuta primero: venv/bin/python fantasy_laliga/src/team_strength_model.py")
        st.stop()
    return pd.read_csv(p)

@st.cache_data
def load_elos():
    p = Path("/home/Projects/futbol_elo_poisson/data/processed/final_elo_ratings_uefa_calibrated.json")
    return json.load(open(p))

df_strengths = load_team_strengths()
elos = load_elos()

# Sistema de puntos LaLiga Fantasy
POINTS_GOAL = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
POINTS_ASSIST = 3
POINTS_CS = {"GK": 5, "DEF": 4, "MID": 1, "FWD": 0}
HOME_ADV = 65

def expected_score(rh, ra, ha=HOME_ADV):
    return 1 / (1 + 10**((ra - (rh + ha))/400))

def predict_matchup(home_team, away_team, df):
    h = df[df["team"] == home_team]
    a = df[df["team"] == away_team]
    if h.empty or a.empty: return None
    h, a = h.iloc[0], a.iloc[0]

    # Ajuste goles esperados por rival
    league_avg_goals = 1.3
    xg_h = h["goals_per_match"] * (a["goals_conceded_per_match"] / league_avg_goals)
    xg_a = a["goals_per_match"] * (h["goals_conceded_per_match"] / league_avg_goals)
    # Home advantage
    xg_h *= 1.15
    xg_a *= 0.85

    p_win_h = expected_score(h["elo"], a["elo"], HOME_ADV)
    p_cs_h = max(0, 1 - xg_a * 0.7)
    p_cs_a = max(0, 1 - xg_h * 0.7)

    return {
        "home": home_team, "away": away_team,
        "xg_h": xg_h, "xg_a": xg_a,
        "p_win_h": p_win_h, "p_cs_h": p_cs_h, "p_cs_a": p_cs_a,
        "h_row": h, "a_row": a,
    }


def fantasy_points_for_position(pred, position, is_home):
    if pred is None: return 0
    xg = pred["xg_h"] if is_home else pred["xg_a"]
    p_cs = pred["p_cs_h"] if is_home else pred["p_cs_a"]
    xg_conceded = pred["xg_a"] if is_home else pred["xg_h"]

    base = 2  # jugar 60+ min
    goals_pts = 0
    cs_pts = p_cs * POINTS_CS.get(position, 0)
    conceded_penalty = 0

    if position == "GK":
        goals_pts = 0
        conceded_penalty = -xg_conceded * 1
    elif position == "DEF":
        goals_pts = (xg / 3.5) * POINTS_GOAL["DEF"] * 0.15  # baja prob defensa marca
        conceded_penalty = -(xg_conceded // 2)
    elif position == "MID":
        goals_pts = (xg / 3.5) * POINTS_GOAL["MID"] * 0.5
    elif position == "FWD":
        goals_pts = (xg / 2.5) * POINTS_GOAL["FWD"] * 1.0  # más prob delantero marca

    return round(base + goals_pts + cs_pts + conceded_penalty, 2)


# ═══════════════ SIDEBAR ═══════════════
st.sidebar.title("🇪🇸 LaLiga Fantasy")
st.sidebar.markdown(f"**{len(df_strengths)}** equipos LaLiga")
st.sidebar.markdown("**Sistema puntos**: LaLiga Fantasy oficial")
st.sidebar.markdown("---")
st.sidebar.warning("""
⚠️ **Modelo baseline por equipos**

Predicciones actuales usan fuerza ofensiva/defensiva del CLUB.

Para modelo por jugador individual, esperando scraping FBref player_stats (en background).
""")

page = st.sidebar.radio("Sección:", [
    "🏠 Home",
    "⚽ Predecir enfrentamiento",
    "🎯 Ranking equipos por posición",
    "📊 Fuerza equipos LaLiga",
])

# ═══════════════ PÁGINAS ═══════════════
if page == "🏠 Home":
    st.title("🇪🇸 LaLiga Fantasy Assistant — Baseline")
    st.markdown("""
    Este dashboard orienta decisiones de fichaje para **LaLiga Fantasy oficial** basándose en
    la **fuerza actual de cada equipo LaLiga** (últimas 3 temporadas).

    ### Cómo usarlo hasta que arranque LaLiga 2026/27:
    - **📊 Fuerza equipos**: ranking ofensivo y defensivo
    - **🎯 Ranking equipos por posición**: qué clubes convienen para GK/DEF/MID/FWD
    - **⚽ Predecir enfrentamiento**: puntos fantasy esperados de un partido concreto
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Equipos LaLiga cubiertos", len(df_strengths))
    c2.metric("Partidos base", int(df_strengths["matches"].sum()))
    c3.metric("Temporadas", "3 (23/24, 24/25, 25/26)")

    st.markdown("### 🏆 Top equipos LaLiga por ELO")
    top = df_strengths.nlargest(10, "elo")[["team", "elo", "goals_per_match", "clean_sheet_ratio"]]
    fig = px.bar(top, x="elo", y="team", orientation="h", color="elo",
                 color_continuous_scale="Blues", text="elo", height=400)
    fig.update_traces(texttemplate='%{text:.0f}')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚽ Predecir enfrentamiento":
    st.title("⚽ Predicción de enfrentamiento — LaLiga Fantasy")

    teams = sorted(df_strengths["team"].tolist())
    c1, c2 = st.columns(2)
    home = c1.selectbox("🏠 Local:", teams, index=teams.index("Real Madrid") if "Real Madrid" in teams else 0)
    away = c2.selectbox("✈️ Visitante:", teams, index=teams.index("Barcelona") if "Barcelona" in teams else 1)

    if home == away:
        st.warning("Elige dos equipos distintos")
        st.stop()

    pred = predict_matchup(home, away, df_strengths)

    # Métricas match
    st.markdown("### 📊 Predicción del partido")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"xG {home}", f"{pred['xg_h']:.2f}")
    c2.metric(f"xG {away}", f"{pred['xg_a']:.2f}")
    c3.metric(f"P({home} gana)", f"{pred['p_win_h']*100:.1f}%")
    c4.metric("Diff ELO", f"{pred['h_row']['elo']-pred['a_row']['elo']:+.0f}")

    # Puntos fantasy por posición
    st.markdown("### 🎯 Puntos fantasy esperados por posición (jugador titular 90 min)")
    data = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        data.append({
            "Posición": pos,
            f"{home} (local)": fantasy_points_for_position(pred, pos, is_home=True),
            f"{away} (visit.)": fantasy_points_for_position(pred, pos, is_home=False),
        })
    df_pts = pd.DataFrame(data)
    st.dataframe(df_pts, use_container_width=True)

    # Gráfica comparativa
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"{home} (local)", x=df_pts["Posición"],
                         y=df_pts[f"{home} (local)"], marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name=f"{away} (visit.)", x=df_pts["Posición"],
                         y=df_pts[f"{away} (visit.)"], marker_color="#e74c3c"))
    fig.update_layout(barmode='group', title="Puntos fantasy esperados por posición",
                      yaxis_title="Puntos esperados 90 min", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **Interpretación**:
    - Si `xG local` alto → considera **DELANTEROS** del equipo local
    - Si `P(clean sheet)` alto → considera **DEFENSAS/PORTERO** de ese equipo
    - Un mediocampista de equipo ofensivo suele dar buenos puntos por asistencias
    """)

elif page == "🎯 Ranking equipos por posición":
    st.title("🎯 Ranking equipos LaLiga por posición fantasy")
    st.markdown("¿De qué equipos LaLiga fichar tus jugadores según su posición?")

    pos = st.selectbox("Posición:", ["GK", "DEF", "MID", "FWD"])

    # Calcular puntos fantasy esperados MEDIOS (agregando local + visitante)
    scores = []
    for _, row in df_strengths.iterrows():
        team = row["team"]
        # Simulación media: enfrentar a un rival de fuerza media (Getafe ~ELO 1500)
        avg_rival_elo = 1550
        # Pred jugando local
        pred_h = predict_matchup(team, df_strengths[df_strengths["elo"].between(avg_rival_elo-30, avg_rival_elo+30)].iloc[0]["team"], df_strengths) if len(df_strengths[df_strengths["elo"].between(avg_rival_elo-30, avg_rival_elo+30)]) > 0 else None
        pred_a = pred_h  # aproximación
        if pred_h:
            pts_h = fantasy_points_for_position(pred_h, pos, is_home=True)
            pts_a = fantasy_points_for_position(pred_h, pos, is_home=False)
            scores.append({
                "team": team, "elo": row["elo"],
                "pts_home": pts_h, "pts_away": pts_a,
                "pts_avg": (pts_h + pts_a) / 2,
                "goals_per_match": row["goals_per_match"],
                "clean_sheet_ratio": row["clean_sheet_ratio"],
            })

    df_pos = pd.DataFrame(scores).sort_values("pts_avg", ascending=False)

    st.markdown(f"### Ranking {pos} — mejor a peor equipo para fichar")
    fig = px.bar(df_pos.head(20), x="pts_avg", y="team", orientation="h",
                 color="elo", color_continuous_scale="Greens",
                 text="pts_avg", title=f"Puntos fantasy esperados por partido — {pos}",
                 height=500)
    fig.update_traces(texttemplate='%{text:.2f}')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_pos[["team", "elo", "pts_avg", "pts_home", "pts_away", "goals_per_match", "clean_sheet_ratio"]].round(2).reset_index(drop=True), use_container_width=True)

elif page == "📊 Fuerza equipos LaLiga":
    st.title("📊 Fuerza equipos LaLiga")
    st.markdown("Métricas base para elegir jugadores según su club:")

    # Scatter: fuerza ofensiva vs defensiva
    fig = px.scatter(df_strengths, x="goals_conceded_per_match", y="goals_per_match",
                     size="elo", color="elo", hover_data=["team"],
                     text="team",
                     title="Fuerza ofensiva vs defensiva por equipo",
                     labels={"goals_conceded_per_match": "Goles encajados/partido (menos = mejor defensa)",
                             "goals_per_match": "Goles marcados/partido (más = mejor ataque)"},
                     height=600, color_continuous_scale="RdYlGn")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Ranking ATAQUE (mejores DELANTEROS/MEDIAPUNTAS aquí)")
    st.dataframe(df_strengths.nlargest(10, "goals_per_match")[["team", "elo", "goals_per_match", "goals_conceded_per_match"]].round(2).reset_index(drop=True), use_container_width=True)

    st.markdown("### 🛡️ Ranking DEFENSA (mejores DEFENSAS/PORTEROS aquí)")
    st.dataframe(df_strengths.nsmallest(10, "goals_conceded_per_match")[["team", "elo", "clean_sheet_ratio", "goals_conceded_per_match"]].round(3).reset_index(drop=True), use_container_width=True)

"""
Dashboard v2 — Simulador interactivo de partidos + visualizaciones.

Ejecutar:
    venv/bin/streamlit run app/dashboard_v2.py

Páginas:
1. 🏠 Home — ranking global + top competición
2. ⚽ Simulador 1v1 — elige 2 equipos, obtén probabilidades + gráficas
3. 📊 Ranking por competición — top 20 con gráfica
4. 🇪🇺 Comparativa cross-liga — histograma ELOs por país
5. 📈 Evolución temporal — ELO de 1-3 equipos a lo largo del tiempo
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
from scipy.stats import poisson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Fútbol ELO Predictor", layout="wide", page_icon="⚽")

# ═══════════════ DATOS ═══════════════
@st.cache_data
def load_elos():
    p = ROOT / "data/processed/final_elo_ratings_uefa_calibrated.json"
    if not p.exists():
        p = ROOT / "mini/mini_final_elo.json"
    return json.load(open(p))

@st.cache_data
def load_matches():
    return pd.read_parquet(ROOT / "data/processed/matches_all_multi.parquet")

@st.cache_data
def load_uefa_coefs():
    p = ROOT / "data/processed/uefa_country_k.json"
    return json.load(open(p)) if p.exists() else {}

elos = load_elos()
df_all = load_matches()
uefa_k = load_uefa_coefs()

# Mapping equipo → país (para info visual)
COUNTRY_BY_COMP = {
    "domestic_premier": "England 🏴󠁧󠁢󠁥󠁮󠁧󠁿", "domestic_laliga": "Spain 🇪🇸",
    "domestic_seriea": "Italy 🇮🇹", "domestic_bundesliga": "Germany 🇩🇪",
    "domestic_ligue1": "France 🇫🇷", "domestic_portugal": "Portugal 🇵🇹",
}
@st.cache_data
def build_team_country():
    tc = {}
    for comp, country in COUNTRY_BY_COMP.items():
        sub = df_all[df_all["competition"] == comp]
        for t in set(sub["home_team"]) | set(sub["away_team"]):
            if t not in tc:
                tc[t] = country
    return tc
team_country = build_team_country()

# ═══════════════ FUNCIONES CORE ═══════════════
POISSON = {"α_H": 0.3883, "β_H": 0.001842, "α_A": 0.1763, "β_A": -0.001824}
HOME_ADV = 65

def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1

def score_matrix(lh, la, rho=-0.03, size=9):
    m = np.zeros((size, size))
    ph = poisson.pmf(np.arange(size), lh)
    pa = poisson.pmf(np.arange(size), la)
    for i in range(size):
        for j in range(size):
            m[i,j] = ph[i] * pa[j] * dc_tau(i,j,lh,la,rho)
    return m / m.sum()

def predict_match(elo_h, elo_a, neutral=False):
    ha = 0 if neutral else HOME_ADV
    diff = elo_h - elo_a + ha
    lh = np.exp(POISSON["α_H"] + POISSON["β_H"] * (elo_h - elo_a))
    la = np.exp(POISSON["α_A"] + POISSON["β_A"] * (elo_h - elo_a))
    if neutral:
        avg = np.sqrt(lh * la); lh, la = avg, avg
    mat = score_matrix(lh, la)
    n = mat.shape[0]
    ph = sum(mat[i,j] for i in range(n) for j in range(n) if i > j)
    pd_ = sum(mat[i,i] for i in range(n))
    pa = sum(mat[i,j] for i in range(n) for j in range(n) if i < j)
    over25 = sum(mat[i,j] for i in range(n) for j in range(n) if i+j > 2)
    btts = sum(mat[i,j] for i in range(1,n) for j in range(1,n))
    return {"lh": lh, "la": la, "mat": mat, "P_H": ph, "P_D": pd_, "P_A": pa,
            "P_over25": over25, "P_btts": btts}

# ═══════════════ SIDEBAR ═══════════════
st.sidebar.title("⚽ Fútbol ELO Predictor")
st.sidebar.markdown("**Datos**: 23.537 partidos, 9 competiciones, ELO UEFA-calibrado")
page = st.sidebar.radio("Sección:", [
    "🏠 Home",
    "⚽ Simulador 1v1",
    "📊 Ranking por competición",
    "🇪🇺 Comparativa cross-liga",
    "📈 Evolución temporal ELO",
    "🔍 Histórico H2H",
])
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 {len(elos)} equipos ratedos")
st.sidebar.caption(f"📅 Datos hasta {df_all['date'].max().date()}")

# ═══════════════ PÁGINA HOME ═══════════════
if page == "🏠 Home":
    st.title("⚽ Fútbol ELO Predictor")
    st.markdown("Simulador de partidos basado en **ELO cross-league** calibrado con **coeficiente UEFA**.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos totales", f"{len(df_all):,}")
    c2.metric("Competiciones", "9")
    c3.metric("Temporadas", "9 (2017-2026)")
    c4.metric("Equipos", f"{len(elos)}")

    st.markdown("---")
    st.subheader("🏆 Top 25 ELO Global")
    top = sorted(elos.items(), key=lambda x: -x[1])[:25]
    df_top = pd.DataFrame(top, columns=["Equipo", "ELO"])
    df_top["País"] = df_top["Equipo"].map(team_country).fillna("Otro 🌍")

    fig = px.bar(df_top, x="ELO", y="Equipo", color="País", orientation="h",
                 title="Top 25 equipos por ELO cross-league",
                 height=700, text="ELO")
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 UEFA Country Coefficient (calculado sobre nuestros datos)")
    if uefa_k:
        df_k = pd.DataFrame(sorted(uefa_k.items(), key=lambda x: -x[1]),
                            columns=["País", "K liga"])
        fig2 = px.bar(df_k, x="País", y="K liga",
                      title="K por liga doméstica (derivado del UEFA coefficient 5-year)",
                      color="K liga", color_continuous_scale="Blues", text="K liga")
        fig2.update_traces(texttemplate='%{text:.2f}')
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════ PÁGINA SIMULADOR 1v1 ═══════════════
elif page == "⚽ Simulador 1v1":
    st.title("⚽ Simulador de Partidos")
    st.markdown("Elige dos equipos y obtén predicciones basadas en su ELO actual.")

    teams_sorted = sorted(elos.keys())
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        home = st.selectbox("🏠 Equipo LOCAL", teams_sorted,
                            index=teams_sorted.index("Real Madrid") if "Real Madrid" in teams_sorted else 0)
    with c2:
        st.markdown("### VS")
        neutral = st.checkbox("Campo neutral", help="Sin ventaja de local")
    with c3:
        away = st.selectbox("✈️ Equipo VISITANTE", teams_sorted,
                            index=teams_sorted.index("Barcelona") if "Barcelona" in teams_sorted else 1)

    if home == away:
        st.warning("Elige dos equipos distintos")
        st.stop()

    elo_h = elos[home]
    elo_a = elos[away]
    pred = predict_match(elo_h, elo_a, neutral=neutral)

    # ELO comparativa
    st.markdown("### 📊 ELO ratings")
    ce1, ce2, ce3 = st.columns(3)
    ce1.metric(f"{home}", f"{elo_h:.0f}", team_country.get(home, ""))
    ce2.metric("Diferencia", f"{elo_h - elo_a:+.0f}")
    ce3.metric(f"{away}", f"{elo_a:.0f}", team_country.get(away, ""))

    # Goles esperados
    st.markdown("### ⚽ Goles esperados (Poisson GLM)")
    cg1, cg2 = st.columns(2)
    cg1.metric(f"λ_{home}", f"{pred['lh']:.2f}")
    cg2.metric(f"λ_{away}", f"{pred['la']:.2f}")

    # Probabilidades 1X2 — gráfica de barras horizontal
    st.markdown("### 🎯 Probabilidades 1X2")
    df_probs = pd.DataFrame([
        {"Resultado": f"{home} gana", "Prob": pred["P_H"]*100, "Color": "#2ecc71"},
        {"Resultado": "Empate",      "Prob": pred["P_D"]*100, "Color": "#95a5a6"},
        {"Resultado": f"{away} gana", "Prob": pred["P_A"]*100, "Color": "#e74c3c"},
    ])
    fig = go.Figure(go.Bar(
        x=df_probs["Prob"], y=df_probs["Resultado"],
        orientation='h', marker=dict(color=df_probs["Color"]),
        text=df_probs["Prob"].round(1).astype(str) + "%", textposition="outside",
    ))
    fig.update_layout(height=300, xaxis_title="Probabilidad (%)", xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Matriz heatmap de resultados
    st.markdown("### 🎲 Matriz de resultados exactos (Poisson + Dixon-Coles)")
    mat = pred["mat"][:6, :6]
    fig_h = go.Figure(data=go.Heatmap(
        z=mat*100,
        x=[f"{away} = {j}" for j in range(6)],
        y=[f"{home} = {i}" for i in range(6)],
        colorscale="Reds",
        text=[[f"{v*100:.1f}%" for v in row] for row in mat],
        texttemplate="%{text}",
        colorbar=dict(title="Prob %"),
    ))
    fig_h.update_layout(height=400)
    st.plotly_chart(fig_h, use_container_width=True)

    # Top 5 resultados
    st.markdown("### 🥇 Top 5 resultados más probables")
    flat = [(i, j, pred["mat"][i,j]) for i in range(9) for j in range(9)]
    flat.sort(key=lambda x: -x[2])
    df_top5 = pd.DataFrame([
        {"Resultado": f"{home} {i}-{j} {away}", "Prob %": p*100}
        for i, j, p in flat[:5]
    ])
    fig_top = px.bar(df_top5, x="Prob %", y="Resultado", orientation='h',
                     text="Prob %", height=350)
    fig_top.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True)

    # Mercados adicionales
    st.markdown("### 📈 Otros mercados")
    cm1, cm2 = st.columns(2)
    cm1.metric("P(Over 2.5 goles)", f"{pred['P_over25']*100:.1f}%")
    cm2.metric("P(Ambos marcan)", f"{pred['P_btts']*100:.1f}%")


# ═══════════════ PÁGINA RANKING POR COMPETICIÓN ═══════════════
elif page == "📊 Ranking por competición":
    st.title("📊 Ranking por competición")
    comp = st.selectbox("Competición:", list(COUNTRY_BY_COMP.keys()),
                       format_func=lambda x: COUNTRY_BY_COMP[x])
    teams_in = sorted(set(df_all[(df_all["competition"] == comp) &
                                  (df_all["season"] == "2025-26")]["home_team"]))
    df_rank = pd.DataFrame([{"Equipo": t, "ELO": elos.get(t, 1500)}
                            for t in teams_in]).sort_values("ELO", ascending=True)

    st.markdown(f"### {COUNTRY_BY_COMP[comp]} — {len(teams_in)} equipos (temporada 25/26)")

    fig = px.bar(df_rank, x="ELO", y="Equipo", orientation="h",
                 title=f"Ranking ELO {COUNTRY_BY_COMP[comp]}",
                 color="ELO", color_continuous_scale="RdYlGn",
                 text="ELO", height=max(400, len(teams_in)*30))
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_rank.sort_values("ELO", ascending=False).reset_index(drop=True),
                 use_container_width=True)


# ═══════════════ PÁGINA COMPARATIVA CROSS-LIGA ═══════════════
elif page == "🇪🇺 Comparativa cross-liga":
    st.title("🇪🇺 Comparativa cross-liga")
    st.markdown("Distribución de ELOs por liga en la temporada 25/26.")

    data_all = []
    for comp, country in COUNTRY_BY_COMP.items():
        teams = sorted(set(df_all[(df_all["competition"] == comp) &
                                   (df_all["season"] == "2025-26")]["home_team"]))
        for t in teams:
            data_all.append({"País": country, "Equipo": t, "ELO": elos.get(t, 1500)})
    df_cross = pd.DataFrame(data_all)

    fig = px.box(df_cross, x="País", y="ELO", points="all", color="País",
                 hover_data=["Equipo"], title="Distribución ELO por liga (boxplot)",
                 height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Promedios y máximos por liga")
    df_stats = df_cross.groupby("País")["ELO"].agg(["mean", "median", "max", "min", "count"]).round(1).reset_index()
    df_stats.columns = ["País", "Media", "Mediana", "Máx", "Mín", "Equipos"]
    df_stats = df_stats.sort_values("Media", ascending=False)
    st.dataframe(df_stats, use_container_width=True)

    fig2 = px.bar(df_stats, x="País", y="Media", color="Media",
                  title="ELO promedio por liga", text="Media",
                  color_continuous_scale="Viridis")
    fig2.update_traces(texttemplate='%{text:.0f}')
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════ PÁGINA EVOLUCIÓN TEMPORAL ═══════════════
elif page == "📈 Evolución temporal ELO":
    st.title("📈 Evolución temporal ELO")
    st.markdown("Elige hasta 5 equipos para ver cómo evolucionó su ELO temporada a temporada.")

    all_teams = sorted(elos.keys())
    selected = st.multiselect("Equipos:", all_teams,
                              default=["Barcelona", "Real Madrid", "Bayern Munich"])

    if not selected:
        st.info("Selecciona al menos un equipo")
        st.stop()

    # Recomputar ELO trackeando final de temporada por equipo
    from elo import expected_score, goal_diff_multiplier
    K_UCL = {"champions_league": 50, "europa_league": 40, "conference_league": 30}
    K_LIGA = {"domestic_premier": 20, "domestic_laliga": uefa_k.get("Spain", 20),
              "domestic_seriea": uefa_k.get("Italy", 20), "domestic_bundesliga": uefa_k.get("Germany", 20),
              "domestic_ligue1": uefa_k.get("France", 20), "domestic_portugal": uefa_k.get("Portugal", 20)}

    @st.cache_data
    def compute_elo_history_per_season(selected_teams):
        df_sorted = df_all.sort_values("date").reset_index(drop=True)
        ratings = {}
        history = {t: [] for t in selected_teams}
        current_season = None
        for _, row in df_sorted.iterrows():
            if row["season"] != current_season:
                for t in selected_teams:
                    if t in ratings:
                        history[t].append({"season": current_season, "elo": ratings[t]})
                current_season = row["season"]
            h, a = row["home_team"], row["away_team"]
            rh = ratings.get(h, 1500)
            ra = ratings.get(a, 1500)
            k = K_UCL.get(row["competition"], K_LIGA.get(row["competition"], 20))
            gh, ga = int(row["home_goals"]), int(row["away_goals"])
            actual = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
            exp = expected_score(rh, ra, home_adv=65)
            mult = goal_diff_multiplier(gh - ga)
            delta = k * mult * (actual - exp)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
        for t in selected_teams:
            if t in ratings:
                history[t].append({"season": current_season, "elo": ratings[t]})
        return history

    with st.spinner("Calculando historia ELO..."):
        hist = compute_elo_history_per_season(tuple(selected))

    data = []
    for team, points in hist.items():
        for p in points:
            if p["season"]:
                data.append({"season": p["season"], "team": team, "elo": p["elo"]})
    df_hist = pd.DataFrame(data)

    fig = px.line(df_hist, x="season", y="elo", color="team",
                  markers=True, title="Evolución ELO por temporada",
                  height=500, labels={"season": "Temporada", "elo": "ELO", "team": "Equipo"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_hist.pivot(index="season", columns="team", values="elo").round(1),
                 use_container_width=True)


# ═══════════════ PÁGINA H2H ═══════════════
elif page == "🔍 Histórico H2H":
    st.title("🔍 Histórico enfrentamientos")

    teams_sorted = sorted(set(df_all["home_team"]) | set(df_all["away_team"]))
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Equipo 1:", teams_sorted, key="t1",
                      index=teams_sorted.index("Real Madrid") if "Real Madrid" in teams_sorted else 0)
    t2 = c2.selectbox("Equipo 2:", teams_sorted, key="t2",
                      index=teams_sorted.index("Barcelona") if "Barcelona" in teams_sorted else 1)

    h2h = df_all[
        ((df_all["home_team"] == t1) & (df_all["away_team"] == t2)) |
        ((df_all["home_team"] == t2) & (df_all["away_team"] == t1))
    ].sort_values("date", ascending=False)

    if len(h2h) == 0:
        st.warning(f"No hay enfrentamientos {t1} vs {t2} en nuestro dataset")
        st.stop()

    # Estadísticas
    t1_wins = ((h2h["home_team"] == t1) & (h2h["home_goals"] > h2h["away_goals"])).sum() + \
              ((h2h["away_team"] == t1) & (h2h["away_goals"] > h2h["home_goals"])).sum()
    t2_wins = ((h2h["home_team"] == t2) & (h2h["home_goals"] > h2h["away_goals"])).sum() + \
              ((h2h["away_team"] == t2) & (h2h["away_goals"] > h2h["home_goals"])).sum()
    draws = (h2h["home_goals"] == h2h["away_goals"]).sum()

    st.markdown(f"### {t1} vs {t2} — {len(h2h)} enfrentamientos")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{t1} ganó", int(t1_wins))
    c2.metric("Empates", int(draws))
    c3.metric(f"{t2} ganó", int(t2_wins))

    # Gráfica pie
    fig = px.pie(values=[t1_wins, draws, t2_wins],
                 names=[t1, "Empate", t2],
                 title=f"Distribución H2H ({len(h2h)} partidos)",
                 color_discrete_sequence=["#2ecc71", "#95a5a6", "#e74c3c"])
    st.plotly_chart(fig, use_container_width=True)

    # Tabla completa
    st.markdown("### Detalle de enfrentamientos")
    df_display = h2h[["date", "competition", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
    df_display["Resultado"] = df_display.apply(
        lambda r: f"{r['home_team']} {r['home_goals']}-{r['away_goals']} {r['away_team']}", axis=1)
    st.dataframe(df_display[["date", "competition", "Resultado"]].reset_index(drop=True),
                 use_container_width=True)

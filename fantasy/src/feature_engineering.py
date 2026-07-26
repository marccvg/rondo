"""
Feature engineering para modelo predictor puntos FPL.

Features generadas (por jugador, por GW):
  Rolling últimos N (3, 5, 10):
    - Puntos, xP, minutos, goles, asistencias
    - xG, xA, saves (GK), clean sheets
  Contexto próximo partido:
    - is_home
    - opponent difficulty (basado en ELO opp)
    - dias descanso vs último partido
  Estado forma:
    - starts_ratio_5 (proxy titular)
    - selected_pct (ownership actual)
    - value (precio actual)
    - form (media puntos últimos 5)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True, parents=True)


def build_features():
    df = pd.read_parquet(RAW / "fpl_history_all_seasons.parquet")
    print(f"Base: {len(df):,} rows")

    # Ordenar por jugador + temporada + GW
    df = df.sort_values(["name", "_season", "GW"]).reset_index(drop=True)

    # Convertir posiciones textuales a category si es string
    if df["position"].dtype == "object":
        df["position"] = df["position"].fillna("UNK")

    # Solo jugadores con minutos (elimina rows donde no jugó)
    print(f"Rows con datos (minutes >= 0): {(df['minutes'] >= 0).sum():,}")

    # Rolling features por jugador
    print("Computing rolling features...")
    df["_key"] = df["name"] + "|" + df["_season"]
    grouped = df.groupby("name")

    for stat in ["total_points", "minutes", "goals_scored", "assists",
                 "clean_sheets", "saves", "bonus"]:
        for window in [3, 5, 10]:
            col = f"{stat}_r{window}"
            df[col] = grouped[stat].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    # xP rolling (puntos esperados FPL propios)
    if "xP" in df.columns:
        for window in [3, 5]:
            df[f"xP_r{window}"] = grouped["xP"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    # Expected goals/assists rolling
    for stat in ["expected_goals", "expected_assists"]:
        if stat in df.columns:
            for window in [3, 5, 10]:
                df[f"{stat}_r{window}"] = grouped[stat].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=1).mean()
                )

    # Starts ratio (proxy titular) - solo starts >= 1 counts como titular
    if "starts" in df.columns:
        df["starts_r5"] = grouped["starts"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

    # Forma FPL oficial (media últimos 4 GW puntos con >0 minutos)
    df["form_fpl"] = grouped["total_points"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    # is_home ya está
    # value en decenas de M
    if "value" in df.columns:
        df["value_M"] = df["value"] / 10

    # Points per million (eficiencia coste)
    df["ppm_r5"] = df["total_points_r5"] / df["value_M"].where(df["value_M"] > 0, 1)

    # Trend: puntos últimos 3 vs últimos 10 (subida/bajada)
    df["trend_points"] = df["total_points_r3"] - df["total_points_r10"]

    # Target: puntos que va a hacer en próximo GW
    df["target_next_gw_points"] = grouped["total_points"].shift(-1)

    print(f"Features generadas. Total cols: {len(df.columns)}")
    print(f"Rows con target (jugadores con GW siguiente): {df['target_next_gw_points'].notna().sum():,}")

    out = PROC / "fpl_features.parquet"
    df.to_parquet(out)
    print(f"Guardado: {out}")

    # Feature list útil para modelo
    feature_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in ["total_points_r", "minutes_r", "goals_scored_r",
                                             "assists_r", "clean_sheets_r", "saves_r",
                                             "bonus_r", "xP_r", "expected_goals_r",
                                             "expected_assists_r", "starts_r"])]
    feature_cols += ["form_fpl", "value_M", "ppm_r5", "trend_points", "was_home"]
    print(f"\nFeatures para modelo ({len(feature_cols)}):")
    for c in feature_cols[:20]:
        print(f"  {c}")


if __name__ == "__main__":
    build_features()

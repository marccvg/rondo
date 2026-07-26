"""
Ingesta completa FPL API oficial.
Endpoints (todos gratis, sin auth):
  /bootstrap-static/     — snapshot completo (jugadores, equipos, jornadas, precios)
  /fixtures/             — todos los partidos con difficulty
  /element-summary/{id}/ — historial GW por GW de cada jugador
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/home/Projects/futbol_elo_poisson/fantasy")
RAW = ROOT / "data" / "raw"
RAW.mkdir(exist_ok=True, parents=True)

BASE = "https://fantasy.premierleague.com/api"


def fetch_bootstrap():
    print("→ Bootstrap FPL...")
    r = requests.get(f"{BASE}/bootstrap-static/", timeout=30)
    data = r.json()
    (RAW / "fpl_bootstrap.json").write_text(json.dumps(data, indent=2))
    print(f"  {len(data['elements'])} jugadores, {len(data['teams'])} equipos, {len(data['events'])} jornadas")
    return data


def fetch_fixtures():
    print("→ Fixtures FPL...")
    r = requests.get(f"{BASE}/fixtures/", timeout=30)
    data = r.json()
    (RAW / "fpl_fixtures.json").write_text(json.dumps(data, indent=2))
    print(f"  {len(data)} partidos")
    return data


def fetch_all_player_history(bootstrap):
    """Historial detallado GW por GW de cada jugador."""
    print(f"→ Historial detallado {len(bootstrap['elements'])} jugadores...")
    all_history = []
    for i, p in enumerate(bootstrap["elements"]):
        pid = p["id"]
        if i % 50 == 0:
            print(f"  {i}/{len(bootstrap['elements'])}...")
        try:
            r = requests.get(f"{BASE}/element-summary/{pid}/", timeout=15)
            if r.status_code != 200:
                continue
            hist = r.json()
            for gw in hist.get("history", []):
                gw["player_id"] = pid
                gw["web_name"] = p["web_name"]
                gw["element_type"] = p["element_type"]  # 1=GK 2=DEF 3=MID 4=FWD
                gw["team_id"] = p["team"]
                all_history.append(gw)
            time.sleep(0.05)  # rate-friendly
        except Exception as e:
            print(f"  fail {pid}: {e}")
    df = pd.DataFrame(all_history)
    out = RAW / "fpl_players_history.parquet"
    df.to_parquet(out)
    print(f"  Total {len(df):,} rows GW-jugador → {out}")
    return df


def main():
    bootstrap = fetch_bootstrap()
    fetch_fixtures()
    fetch_all_player_history(bootstrap)
    print("\n✅ Ingesta FPL completa")


if __name__ == "__main__":
    main()

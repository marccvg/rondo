"""Feature engineering — 7 familias.

Cada módulo `family_X_*.py` expone `compute_family_X(matches: pd.DataFrame) -> pd.DataFrame`
que devuelve un DataFrame con la MISMA longitud/orden que `matches` y columnas de features
prefijadas por familia (ej. `strength_elo_home`, `form_pts_5_home_decay`, `matchup_h2h_...`).

Contrato crítico: features SIEMPRE con `shift(1)` (o rolling excluyendo partido actual)
para evitar data leakage.
"""

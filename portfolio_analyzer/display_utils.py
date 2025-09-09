"""Utilidades de display unificadas para todos los módulos.

Evita repetición de lógica de impresión fila por fila y centraliza el uso
condicional de `IPython.display.display` cuando esté disponible.
"""
from __future__ import annotations
import pandas as pd

try:  # pragma: no cover
    from IPython.display import display  # type: ignore
    _HAS_IPYTHON_DISPLAY = True
except Exception:  # pragma: no cover
    _HAS_IPYTHON_DISPLAY = False


def _display_df(df: pd.DataFrame, title: str | None = None, max_rows: int = 50):
    """Muestra un DataFrame de forma compacta y segura.

    Args:
        df: DataFrame a mostrar.
        title: Título opcional que se imprime antes.
        max_rows: Número máximo de filas a mostrar (añade fila '…' si se recorta).
    """
    if df is None or df.empty:
        if title:
            print(f"{title} (sin datos)")
        return
    if title:
        print(title)
    df_to_show = df
    if len(df) > max_rows:
        df_to_show = df.head(max_rows).copy()
        # Añadir indicador de truncado solo si no existe índice string ya usado
        ellipsis_row = {col: '…' for col in df.columns}
        df_to_show.loc['…'] = ellipsis_row
    if _HAS_IPYTHON_DISPLAY:
        display(df_to_show)
    else:
        # to_string para mantener formato legible en consola
        print(df_to_show.to_string())

__all__ = ["_display_df"]

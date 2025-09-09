"""Benchmarks Package
=====================

Construcción de benchmarks compuestos para comparar la cartera real.
Incluye:
- BenchmarkBuilder: Construye un índice compuesto con pesos definidos.
- BenchmarkConfig: Configuración de tickers y pesos.

Metodología:
1. Descarga precios ajustados (Yahoo Finance) de los componentes.
2. Convierte a ARS si se solicita (multiplica activos USD por tipo de cambio USDARS).
3. Calcula retornos logarítmicos diarios.
4. Agrega retornos según pesos para formar el benchmark compuesto.
5. Devuelve precios base 100 (opcional) y retornos del índice.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import yfinance as yf

@dataclass
class BenchmarkConfig:
    tickers: Dict[str, float]  # ticker -> peso
    base_currency: str = "USD"
    convert_to_ars: bool = True
    usdars_ticker: str = "ARS=X"  # yfinance symbol for USDARS

DEFAULT_CONFIG = BenchmarkConfig(
    tickers={
        "QQQ": 0.70,
        "IAGG": 0.25,
        "^MERV": 0.05  # MERVAL índice (ya viene en ARS)
    },
    base_currency="USD",
    convert_to_ars=True,
    usdars_ticker="ARS=X"
)

class BenchmarkBuilder:
    def __init__(self, config: BenchmarkConfig = DEFAULT_CONFIG):
        self.config = config
        self.prices_raw: pd.DataFrame = pd.DataFrame()
        self.prices_ars: pd.DataFrame = pd.DataFrame()
        self.returns_log: pd.DataFrame = pd.DataFrame()
        self.composite_returns: Optional[pd.Series] = None
        self.composite_prices: Optional[pd.Series] = None

    def _download(self, tickers, start, end):
        # Descargar datos ajustados de precio
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        
        # Si las columnas son MultiIndex, aplanarlas
        if isinstance(raw.columns, pd.MultiIndex):
            # Si 'Adj Close' está en el nivel superior, usarlo. Si no, usar el primer nivel.
            level_to_use = 'Adj Close' if 'Adj Close' in raw.columns.levels[0] else raw.columns.levels[0][0]
            data = raw[level_to_use]
        else:
            data = raw

        # Asegurar DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0] if len(tickers) == 1 else 'data')
            
        return data.dropna(how='all')

    def _ensure_weights(self):
        w = self.config.tickers
        total = sum(w.values())
        if not np.isclose(total, 1.0):
            self.config.tickers = {k: v/total for k, v in w.items()}

    def build_composite(self, start: str, end: str, force: bool = False, base_100: bool = True):
        """Construye el benchmark compuesto y devuelve dict con:
          - component_prices
          - component_returns_log
          - composite_returns
          - composite_prices
        """
        if (not self.prices_raw.empty) and (not force):
            return True, self._package()

        try:
            self._ensure_weights()
            tickers = list(self.config.tickers.keys())
            prices = self._download(tickers, start, end)
            prices.columns = tickers if prices.shape[1] == len(tickers) else prices.columns
            self.prices_raw = prices.sort_index()

            # Conversión a ARS (QQQ, IAGG en USD; MERVAL ya ARS)
            if self.config.convert_to_ars:
                fx = self._download([self.config.usdars_ticker], start, end)
                if fx.empty:
                    print("⚠️ No se pudo descargar ARS=X, se mantienen precios en moneda base")
                    self.prices_ars = self.prices_raw.copy()
                else:
                    if self.config.usdars_ticker in fx.columns:
                        fx_rate = fx[self.config.usdars_ticker].reindex(self.prices_raw.index).ffill()
                    else:
                        fx_rate = fx.iloc[:,0].reindex(self.prices_raw.index).ffill()
                    prices_ars = self.prices_raw.copy()
                    for col in prices_ars.columns:
                        if col.startswith('^MERV'):
                            continue
                        prices_ars[col] = prices_ars[col] * fx_rate
                    self.prices_ars = prices_ars
            else:
                self.prices_ars = self.prices_raw.copy()

            self.returns_log = np.log(self.prices_ars / self.prices_ars.shift(1)).dropna()
            weights_vec = pd.Series(self.config.tickers)
            self.composite_returns = (self.returns_log[weights_vec.index] * weights_vec).sum(axis=1)

            comp_prices = (1 + self.composite_returns).cumprod()
            if base_100:
                comp_prices = 100 * comp_prices / comp_prices.iloc[0]
            self.composite_prices = comp_prices
            return True, self._package()
        except Exception as e:
            print(f"❌ Error construyendo benchmark compuesto: {e}")
            return False, {}

    def _package(self):
        return {
            'component_prices': self.prices_ars,
            'component_returns_log': self.returns_log,
            'composite_returns': self.composite_returns,
            'composite_prices': self.composite_prices
        }

    def get_config(self):
        return self.config

__all__ = ["BenchmarkBuilder", "BenchmarkConfig", "DEFAULT_CONFIG"]

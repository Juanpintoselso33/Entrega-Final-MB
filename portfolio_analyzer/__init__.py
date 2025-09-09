"""
Portfolio Analyzer - Análisis de Composición de Cartera Real
==========================================================

Sistema enfocado EXCLUSIVAMENTE en analizar la composición de tu cartera real:
- Carga de datos diarios de la cartera (portfolio_consolidado_completo.csv)
- Análisis de evolución de pesos por activo en el tiempo
- Métricas de concentración y diversificación
- Visualizaciones de composición temporal
- Cálculo de performance basado en datos reales

ENFOQUE: Analizar cómo cambia la composición de TU cartera día a día.

Módulos:
--------
- data_loader: Carga datos reales de la cartera diaria
- risk_analysis: Métricas aplicadas a la cartera real
- visualization: Gráficos de composición temporal
- optimization: Análisis de la eficiencia actual de la cartera
- simulation: Proyecciones basadas en la cartera real

Autor: Portfolio Management System
Versión: 1.0 - Portfolio Composition Focus
"""

__version__ = "1.0.0"
__author__ = "Portfolio Management System"

# Importaciones principales enfocadas en análisis de cartera real
from .data_loader import DataLoader, MovementsAnalyzer, PortfolioCompositionLoader
from .risk_analysis import RiskCalculator
from .optimization import PortfolioOptimizer
from .visualization import PortfolioVisualizer
from .simulation import MonteCarloSimulator
from .benchmarks import BenchmarkBuilder, BenchmarkConfig, DEFAULT_CONFIG

__all__ = [
    'DataLoader',
    'PortfolioCompositionLoader',  # Nueva clase principal
    'MovementsAnalyzer', 
    'RiskCalculator',
    'PortfolioOptimizer',
    'PortfolioVisualizer',
    'MonteCarloSimulator',
    'BenchmarkBuilder',
    'BenchmarkConfig',
    'DEFAULT_CONFIG'
]

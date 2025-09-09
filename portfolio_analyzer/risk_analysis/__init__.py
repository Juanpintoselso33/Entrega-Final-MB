"""
Risk Analysis Module - Análisis de Riesgo de la Cartera Real
==========================================================

Módulo ENFOCADO en analizar el riesgo y performance de TU cartera real:
- Calcular métricas aplicadas a los datos reales de tu cartera
- Analizar el riesgo de tu composición actual
- Performance de cada activo en tu cartera real
- Evolución del riesgo conforme cambia la composición
- Métricas basadas en tu serie diaria real (portfolio_consolidado_completo.csv)

OBJETIVO: Entender el perfil de riesgo de TU cartera específica.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from ..display_utils import _display_df


class RiskCalculator:
    """Calculadora completa de métricas de riesgo y performance"""
    
    def __init__(self, risk_free_rate=0.04):
        """
        Inicializa la calculadora de riesgo.
        
        Args:
            risk_free_rate (float): Tasa libre de riesgo anualizada.
        """
        self.risk_free_rate = risk_free_rate / 252  # Tasa diaria
        self.annual_rf = risk_free_rate
        self.benchmark = None
        self.benchmark_data = None
        self.fx_data = None
        
        print(f"✅ RiskCalculator inicializado con risk_free_rate={self.risk_free_rate}")
    
    def set_benchmark(self, benchmark_series):
        """Configura datos del benchmark para comparaciones - NUEVO MÉTODO"""
        if benchmark_series is not None and not benchmark_series.empty:
            self.benchmark_data = benchmark_series
            print(f"📊 Benchmark configurado: {len(benchmark_series)} observaciones")
        else:
            print("⚠️ Benchmark vacío - usando análisis absoluto")
            
    def set_fx_rate(self, fx_series):
        """Configura serie de tipo de cambio para análisis FX - NUEVO MÉTODO"""
        if fx_series is not None and not fx_series.empty:
            self.fx_data = fx_series
            print(f"💱 Tipo de cambio configurado: {len(fx_series)} observaciones")
        else:
            print("⚠️ Datos FX no disponibles")
        
    def set_dynamic_risk_free_rate(self, df):
        """Configura tasa libre de riesgo dinámica desde datos BCRA"""
        if 'Tasa_PlazoFijo_Diaria' in df.columns:
            self.dynamic_rf = df['Tasa_PlazoFijo_Diaria'].mean()
            self.annual_rf = self.dynamic_rf * 365
            self.rf_rate = self.dynamic_rf
            print(f"✅ Tasa libre de riesgo: {self.annual_rf:.2%} anual")
    
    def calculate_basic_metrics(self, returns):
        """Calcula métricas básicas de performance"""
        
        # Limpiar datos
        returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 5:  # Solo requerir mínimo 5 observaciones
            return None
            
        # Métricas básicas
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        excess_return = annual_return - self.annual_rf
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        sortino = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # VaR y CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'rendimiento_anual': annual_return,
            'volatilidad': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'observaciones': len(returns)
        }
    
    def calculate_alpha_beta(self, asset_returns, benchmark_returns):
        """Calcula Alpha y Beta usando regresión simple"""
        
        # Alinear datos
        if len(asset_returns) != len(benchmark_returns):
            min_len = min(len(asset_returns), len(benchmark_returns))
            asset_returns = asset_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        # Filtrar NaN
        mask = ~(np.isnan(asset_returns) | np.isnan(benchmark_returns))
        asset_returns = asset_returns[mask]
        benchmark_returns = benchmark_returns[mask]
        
        if len(asset_returns) < 5:  # Reducir requerimiento mínimo
            return None, None, None
        
        try:
            # Beta = Cov(R_asset, R_market) / Var(R_market)
            covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
            market_variance = np.var(benchmark_returns)
            
            if market_variance == 0:
                return None, None, None
            
            beta = covariance / market_variance
            
            # Alpha = R_asset_mean - Beta * R_market_mean
            alpha_daily = np.mean(asset_returns) - beta * np.mean(benchmark_returns)
            alpha_annual = alpha_daily * 252
            
            # R-squared
            correlation = np.corrcoef(asset_returns, benchmark_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            return alpha_annual, beta, r_squared
            
        except Exception:
            return None, None, None
    
    def calculate_fx_metrics(self, returns, df):
        """Calcula métricas de exposición al riesgo cambiario - MEJORADO PARA DATASET INTEGRADO"""
        
        fx_metrics = {
            'fx_correlation': None,
            'fx_beta': None, 
            'crisis_volatility_ratio': None
        }
        
        # Buscar columna de tipo de cambio según formato
        fx_col = None
        fx_candidates = ['USD_ARS_Rate', 'Tipo_Cambio_ARSUSD', 'Retorno_FX']
        
        for candidate in fx_candidates:
            if candidate in df.columns:
                fx_col = candidate
                break
        
        if fx_col is None:
            return fx_metrics
        
        # Calcular retornos FX si es necesario
        if fx_col in ['USD_ARS_Rate', 'Tipo_Cambio_ARSUSD']:
            fx_prices = df[fx_col].dropna()
            if len(fx_prices) < 5:
                return fx_metrics
            fx_returns = fx_prices.pct_change().dropna()
        else:
            fx_returns = df[fx_col].dropna()
        
        # Encontrar fechas comunes
        if hasattr(returns.index, 'date') and hasattr(fx_returns.index, 'date'):
            # Ambos tienen índices datetime
            common_dates = returns.index.intersection(fx_returns.index)
        else:
            # Alinear por posición
            min_len = min(len(returns), len(fx_returns))
            common_dates = range(min_len)
            
        if len(common_dates) < 5:
            return fx_metrics
            
        try:
            if hasattr(common_dates, '__iter__') and hasattr(common_dates[0], 'date'):
                # Usar índices datetime
                aligned_returns = returns.loc[common_dates]
                aligned_fx = fx_returns.loc[common_dates]
            else:
                # Usar posiciones
                aligned_returns = returns.iloc[:len(common_dates)]
                aligned_fx = fx_returns.iloc[:len(common_dates)]
            
            # Correlación con FX
            correlation = np.corrcoef(aligned_returns, aligned_fx)[0, 1]
            fx_metrics['fx_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Beta FX (sensibilidad al tipo de cambio)
            covariance = np.cov(aligned_returns, aligned_fx)[0, 1]
            fx_variance = np.var(aligned_fx)
            fx_metrics['fx_beta'] = covariance / fx_variance if fx_variance > 0 else 0
            
            # Volatilidad durante crisis cambiarias
            fx_threshold = aligned_fx.mean() + 2 * aligned_fx.std()
            crisis_periods = aligned_fx > fx_threshold
            
            if crisis_periods.any():
                crisis_vol = aligned_returns[crisis_periods].std() * np.sqrt(252)
                normal_vol = aligned_returns[~crisis_periods].std() * np.sqrt(252)
                fx_metrics['crisis_volatility_ratio'] = crisis_vol / normal_vol if normal_vol > 0 else 1
            else:
                fx_metrics['crisis_volatility_ratio'] = 1
                
        except Exception as e:
            # Solo mostrar error para casos inesperados
            pass
            
        return fx_metrics
    
    def calculate_advanced_metrics(self, returns, benchmark_returns=None):
        """
        Calcula métricas avanzadas. Si se provee benchmark_returns, calcula alpha y beta.
        """
        metrics = {}

        # Alpha y Beta (si se provee un benchmark)
        if benchmark_returns is not None:
            # Asegurarse que los datos estén alineados
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 1:
                aligned_returns = returns[common_index]
                aligned_benchmark = benchmark_returns[common_index]

                # Regresión lineal para beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                    metrics['beta'] = beta
                    
                    # Cálculo de Alpha
                    annual_return = aligned_returns.mean() * 252
                    annual_benchmark_return = aligned_benchmark.mean() * 252
                    alpha = annual_return - (self.risk_free_rate + beta * (annual_benchmark_return - self.risk_free_rate))
                    metrics['alpha'] = alpha
                else:
                    metrics['beta'] = None
                    metrics['alpha'] = None
            else:
                metrics['beta'] = None
                metrics['alpha'] = None
        
        # Kelly Criterion
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            win_rate = len(positive_returns) / len(returns)
            
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_f = (b * win_rate - (1 - win_rate)) / b
                metrics['kelly_criterion'] = max(0, min(kelly_f, 1))
            else:
                metrics['kelly_criterion'] = None
        else:
            metrics['kelly_criterion'] = None
        
        # Calmar Ratio
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        if max_dd < 0:
            annual_return = returns.mean() * 252
            metrics['calmar_ratio'] = annual_return / abs(max_dd)
        else:
            metrics['calmar_ratio'] = None
        
        # Ulcer Index
        drawdown_squared = (drawdown * 100) ** 2
        metrics['ulcer_index'] = np.sqrt(drawdown_squared.mean())
        
        return metrics
    
    def get_benchmark_returns(self, df, benchmark_asset=None):
        """Obtiene retornos del benchmark - MEJORADO PARA DATASET INTEGRADO"""
        
        # Usar benchmark específico si se proporciona
        if benchmark_asset and benchmark_asset in df.columns:
            benchmark_prices = df[benchmark_asset].dropna()
            if len(benchmark_prices) > 5:
                return benchmark_prices.pct_change().dropna().values
        
        # Usar benchmark configurado en la clase
        if hasattr(self, 'benchmark') and self.benchmark in df.columns:
            benchmark_prices = df[self.benchmark].dropna()
            if len(benchmark_prices) > 5:
                return benchmark_prices.pct_change().dropna().values
        
        # Benchmarks prioritarios para el nuevo dataset
        priority_benchmarks = ['^MERV', '^GSPC', 'SPY', '^IXIC', 'VTI']
        
        for benchmark in priority_benchmarks:
            if benchmark in df.columns:
                benchmark_prices = df[benchmark].dropna()
                if len(benchmark_prices) > 5:
                    self.benchmark = benchmark  # Guardar benchmark usado
                    return benchmark_prices.pct_change().dropna().values
        
        # Fallback: crear benchmark sintético con activos disponibles
        # Detectar si es DataFrame de precios (índice datetime) o formato anterior
        is_price_dataframe = hasattr(df.index, 'date') or pd.api.types.is_datetime64_any_dtype(df.index)
        
        if is_price_dataframe:
            # Nuevo formato: excluir columnas no-activos
            excluded_cols = ['USD_ARS_Rate', 'Tipo_Cambio_ARSUSD', 'TNA_PlazoFijo', 'Tasa_PlazoFijo_Diaria']
            available_assets = [col for col in df.columns if col not in excluded_cols]
        else:
            # Formato anterior
            available_assets = [col for col in df.columns 
                              if not col.startswith(('Peso_', 'Retorno_', 'Tipo_', 'Tasa_', 'Fecha', 'Valor_Total'))]
        
        if len(available_assets) >= 3:
            # Usar los primeros 5 activos para crear benchmark sintético
            market_assets = available_assets[:5]
            market_prices = df[market_assets].mean(axis=1, skipna=True)
            if len(market_prices.dropna()) > 5:
                return market_prices.pct_change().dropna().values
                
        return None
    
    def calculate_benchmark_metrics(self, returns, df, benchmark_asset=None):
        """Calcula métricas vs benchmark - MEJORADO PARA DATASET INTEGRADO"""
        
        metrics = {
            'alpha': None,
            'beta': None,
            'r_squared': None,
            'treynor_ratio': None,
            'information_ratio': None,
            'tracking_error': None
        }
        
        benchmark_returns = self.get_benchmark_returns(df, benchmark_asset)
        if benchmark_returns is None or len(benchmark_returns) < 5:
            return metrics
            
        try:
            alpha, beta, r_squared = self.calculate_alpha_beta(returns.values, benchmark_returns)
            
            if alpha is not None and beta is not None and r_squared is not None:
                metrics['alpha'] = alpha
                metrics['beta'] = beta
                metrics['r_squared'] = r_squared
                
                # Treynor Ratio
                if abs(beta) > 0.001:
                    excess_return = returns.mean() * 252 - self.annual_rf
                    metrics['treynor_ratio'] = excess_return / beta
                
                # Information Ratio y Tracking Error
                min_len = min(len(returns), len(benchmark_returns))
                if min_len > 5:
                    asset_trim = returns.values[:min_len]
                    bench_trim = benchmark_returns[:min_len]
                    
                    # Filtrar NaNs
                    mask = ~(np.isnan(asset_trim) | np.isnan(bench_trim))
                    if mask.sum() > 5:
                        excess_returns = asset_trim[mask] - bench_trim[mask]
                        tracking_error = np.std(excess_returns) * np.sqrt(252)
                        metrics['tracking_error'] = tracking_error
                        
                        if tracking_error > 0:
                            metrics['information_ratio'] = (np.mean(excess_returns) * 252) / tracking_error
                            
        except Exception as e:
            # Solo mostrar error si es diferente a los esperados
            if "benchmark" not in str(e).lower():
                print(f"⚠️ Error calculando métricas benchmark: {str(e)[:50]}")
        
        return metrics
    
    def calculate_comprehensive_metrics(self, returns, df=None, asset_name=None):
        """Calcula métricas completas para un activo"""
        
        if len(returns) < 5:  # Reducir requerimiento mínimo
            return None
        
        # Limpiar datos
        returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 5:  # Reducir requerimiento mínimo
            return None
        
        # Métricas básicas
        basic_metrics = self.calculate_basic_metrics(returns)
        if not basic_metrics:
            return None
            
        # Métricas avanzadas
        advanced_metrics = self.calculate_advanced_metrics(returns)
        basic_metrics.update(advanced_metrics)
        
        # Métricas vs benchmark
        if df is not None:
            benchmark_metrics = self.calculate_benchmark_metrics(returns, df)
            basic_metrics.update(benchmark_metrics)
            
            # Métricas FX
            fx_metrics = self.calculate_fx_metrics(returns, df)
            basic_metrics.update(fx_metrics)
        
        # Agregar identificación del activo
        if asset_name:
            basic_metrics['activo'] = asset_name
        
        return basic_metrics
    
    def analyze_portfolio_assets(self, df, asset_list, benchmark_asset=None):
        """Analiza todos los activos de la cartera - MEJORADO PARA DATASET INTEGRADO"""
        print(f"🔬 Analizando {len(asset_list)} activos...")

        # Detectar si el df es un DataFrame de precios (índice datetime) o el formato anterior
        is_price_dataframe = hasattr(df.index, 'date') or pd.api.types.is_datetime64_any_dtype(df.index)

        results = []

        # Configurar benchmark si se proporciona
        if benchmark_asset and benchmark_asset in df.columns:
            self.benchmark = benchmark_asset
            print(f"   📊 Benchmark configurado: {benchmark_asset}")

        progress_rows = []
        for asset in asset_list:
            if asset not in df.columns:
                progress_rows.append({'activo': asset, 'estado': 'NO_ENCONTRADO', 'obs': 0})
                continue

            prices = df[asset].dropna()
            obs = len(prices)
            if obs < 5:
                progress_rows.append({'activo': asset, 'estado': 'POCOS_DATOS', 'obs': obs})
                continue

            returns = prices.pct_change().dropna()
            metrics = self.calculate_comprehensive_metrics(returns, df, asset)
            if metrics:
                results.append(metrics)
                progress_rows.append({
                    'activo': asset,
                    'estado': 'OK',
                    'obs': obs,
                    'ret_anual_%': round(metrics.get('rendimiento_anual', 0)*100, 2),
                    'vol_%': round(metrics.get('volatilidad', 0)*100, 2),
                    'sharpe': round(metrics.get('sharpe_ratio', 0), 3),
                    'beta': round(metrics.get('beta', 0), 2) if metrics.get('beta') is not None else None
                })
            else:
                progress_rows.append({'activo': asset, 'estado': 'FALLO_METRICAS', 'obs': obs})
        
        if not results:
            print("   ❌ No se pudieron calcular métricas para ningún activo")
            return pd.DataFrame()

        df_metrics = pd.DataFrame(results)
        print(f"   ✅ Métricas calculadas para {len(df_metrics)} activos")
        progress_df = pd.DataFrame(progress_rows)
        _display_df(progress_df, title="📋 Progreso cálculo métricas")
        
        # Reordenar columnas por importancia
        priority_cols = [
            'activo', 'observaciones', 'rendimiento_anual', 'volatilidad',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'var_95', 'cvar_95'
        ]
        
        advanced_cols = [
            'alpha', 'beta', 'r_squared', 'treynor_ratio', 'information_ratio',
            'tracking_error', 'kelly_criterion', 'calmar_ratio', 'ulcer_index'
        ]
        
        fx_cols = ['fx_correlation', 'fx_beta', 'crisis_volatility_ratio']
        stat_cols = ['skewness', 'kurtosis']
        
        # Construir lista de columnas disponibles
        ordered_cols = []
        for col_group in [priority_cols, advanced_cols, fx_cols, stat_cols]:
            ordered_cols.extend([col for col in col_group if col in df_metrics.columns])
        
        # Agregar columnas restantes
        remaining_cols = [col for col in df_metrics.columns if col not in ordered_cols]
        all_cols = ordered_cols + remaining_cols
        
        return df_metrics[all_cols]
    
    def calculate_portfolio_weighted_metrics(self, metrics_df, weights_dict):
        """Calcula métricas ponderadas para la cartera completa"""
        
        if metrics_df.empty or not weights_dict:
            return {}
            
        # Filtrar solo activos con peso significativo
        significant_assets = [asset for asset, weight in weights_dict.items() if weight > 0.001]
        portfolio_metrics = metrics_df[metrics_df['activo'].isin(significant_assets)].copy()
        
        if portfolio_metrics.empty:
            return {}
            
        # Mapear pesos
        portfolio_metrics['peso'] = portfolio_metrics['activo'].map(weights_dict)
        portfolio_metrics = portfolio_metrics.dropna(subset=['peso'])
        
        # Calcular métricas ponderadas
        weighted_metrics = {}
        
        numeric_cols = [
            'sharpe_ratio', 'sortino_ratio', 'var_95', 'cvar_95', 
            'max_drawdown', 'kelly_criterion', 'alpha', 'beta',
            'fx_correlation', 'fx_beta'
        ]
        
        for col in numeric_cols:
            if col in portfolio_metrics.columns:
                values = portfolio_metrics[col].fillna(0 if col != 'beta' else 1)
                weighted_value = (values * portfolio_metrics['peso']).sum()
                weighted_metrics[f'{col}_global'] = weighted_value
        
        return weighted_metrics
    
    def format_metrics_for_display(self, df_metrics):
        """Formatea métricas para visualización"""
        
        if df_metrics.empty:
            return df_metrics
            
        display_df = df_metrics.copy()
        
        # Formatear porcentajes
        percentage_cols = [
            'rendimiento_anual', 'volatilidad', 'max_drawdown', 
            'var_95', 'cvar_95', 'alpha', 'tracking_error'
        ]
        
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(2)
        
        # Formatear ratios
        ratio_cols = [
            'sharpe_ratio', 'sortino_ratio', 'treynor_ratio', 
            'information_ratio', 'calmar_ratio', 'kelly_criterion',
            'beta', 'r_squared', 'ulcer_index', 'fx_correlation', 'fx_beta'
        ]
        
        for col in ratio_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        return display_df

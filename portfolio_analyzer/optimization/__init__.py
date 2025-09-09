"""
Portfolio Optimization Module - Análisis de Eficiencia de Cartera Real
======================================================================

Módulo ENFOCADO en analizar la eficiencia de TU cartera real:
- Evaluar si tu composición actual es eficiente
- Comparar tu cartera real vs carteras teóricamente óptimas
- Analizar el trade-off riesgo-retorno de tu composición actual
- Sugerir ajustes basados en tu composición histórica real
- Backtesting de tu estrategia real vs alternativas

OBJETIVO: Evaluar qué tan eficiente es TU cartera específica.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Utilidad de display unificada
from ..display_utils import _display_df


class PortfolioOptimizer:

    def generate_efficient_frontier(self, n_portfolios=50):
        """
        Genera la frontera eficiente simulando carteras con diferentes combinaciones de pesos.
        Devuelve un DataFrame con rendimiento, volatilidad y Sharpe ratio de cada cartera.
        """
        results = []
        num_assets = self.n_assets
        returns = self.returns
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        rf = self.rf_rate * 252
        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(num_assets), 1)[0]
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0
            results.append({
                'rendimiento': port_return,
                'volatilidad': port_vol,
                'sharpe_ratio': sharpe,
                'pesos': weights
            })
        df = pd.DataFrame(results)
        return df
    """Optimizador de carteras usando teoría moderna de portafolios"""
    
    def __init__(self, returns_data, risk_free_rate=0.02):
        """
        Inicializa el optimizador - MEJORADO PARA DATASET INTEGRADO
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            DataFrame con retornos de activos (índice: fechas, columnas: activos)
            Puede ser formato directo de precios o retornos precalculados
        risk_free_rate : float
            Tasa libre de riesgo anual
        """
        
        # Verificar si son precios o retornos
        if self._is_price_data(returns_data):
            print("💹 Detectados datos de precios - calculando retornos...")
            self.returns = returns_data.pct_change().dropna()
        else:
            print("📊 Detectados retornos precalculados...")
            self.returns = returns_data.dropna()
            
        # Limpiar datos extremos
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filtrar activos con datos suficientes
        min_observations = max(50, int(len(self.returns) * 0.1))  # Al menos 10% de observaciones o 50 días
        valid_assets = []
        
        excluded_rows = []
        for col in self.returns.columns:
            non_null_count = self.returns[col].count()
            if non_null_count >= min_observations:
                valid_assets.append(col)
            else:
                excluded_rows.append({
                    'Activo': col,
                    'Observaciones': non_null_count,
                    'Mínimo_Requerido': min_observations,
                    'Motivo': 'Datos insuficientes'
                })
        
        if len(valid_assets) < 2:
            raise ValueError("Se necesitan al menos 2 activos con datos suficientes para optimización")
            
        self.returns = self.returns[valid_assets]
        self.assets = valid_assets
        self.n_assets = len(self.assets)
        self.rf_rate = risk_free_rate / 252  # Convertir a tasa diaria
        
        print(f"✅ Optimizador inicializado: {self.n_assets} activos válidos, {len(self.returns)} observaciones")
        if excluded_rows:
            _display_df(pd.DataFrame(excluded_rows), title="⚠️ ACTIVOS EXCLUIDOS")
        
    def _is_price_data(self, data):
        """Detecta si los datos son precios o retornos"""
        # Heurística: si los valores promedio son > 1 y pocos negativos, probablemente son precios
        avg_values = data.mean().mean()
        negative_ratio = (data < 0).sum().sum() / (data.count().sum())
        
        # Si promedio alto y pocos negativos, son precios
        if avg_values > 1 and negative_ratio < 0.3:
            return True
        # Si muchos valores pequeños y muchos negativos, son retornos
        elif avg_values < 1 and negative_ratio > 0.3:
            return False
        else:
            # Caso ambiguo - usar primera diferencia como test
            first_diff = data.iloc[:5].mean().mean()
            return first_diff > 1
        
    def portfolio_performance(self, weights):
        """Calcula performance de la cartera dados los pesos"""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        return portfolio_return, portfolio_std
        
    def negative_sharpe(self, weights):
        """Función objetivo para maximizar Sharpe (minimizar -Sharpe)"""
        portfolio_return, portfolio_std = self.portfolio_performance(weights)
        sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
        return -sharpe
    
    def portfolio_volatility(self, weights):
        """Función objetivo para minimizar volatilidad"""
        _, portfolio_std = self.portfolio_performance(weights)
        return portfolio_std
    
    def optimize_sharpe(self, max_weight=1.0, min_weight=0.0):
        """
        Optimiza para máximo ratio de Sharpe
        
        Parameters:
        -----------
        max_weight : float
            Peso máximo por activo
        min_weight : float  
            Peso mínimo por activo
            
        Returns:
        --------
        dict : Resultado de la optimización
        """
        
        # Restricciones
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Peso inicial igual para todos los activos
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            result = minimize(
                self.negative_sharpe, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = dict(zip(self.assets, result.x))
                portfolio_return, portfolio_std = self.portfolio_performance(result.x)
                sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
                
                return {
                    'tipo': 'Máximo Sharpe',
                    'success': True,
                    'weights': weights,
                    'rendimiento': portfolio_return,
                    'volatilidad': portfolio_std,
                    'sharpe_ratio': sharpe,
                    'optimization_result': result
                }
            else:
                _display_df(pd.DataFrame([{ 'Detalle': result.message }]), title="⚠️ Optimización Sharpe falló")
                
        except Exception as e:
            _display_df(pd.DataFrame([{ 'Error': str(e)}]), title="❌ Error en optimización Sharpe")
            
        return None
    
    def optimize_min_volatility(self, max_weight=1.0, min_weight=0.0):
        """
        Optimiza para mínima volatilidad
        
        Parameters:
        -----------
        max_weight : float
            Peso máximo por activo
        min_weight : float
            Peso mínimo por activo
            
        Returns:
        --------
        dict : Resultado de la optimización
        """
        
        # Restricciones
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Peso inicial igual para todos los activos
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            result = minimize(
                self.portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = dict(zip(self.assets, result.x))
                portfolio_return, portfolio_std = self.portfolio_performance(result.x)
                sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
                
                return {
                    'tipo': 'Mínima Volatilidad',
                    'success': True,
                    'weights': weights,
                    'rendimiento': portfolio_return,
                    'volatilidad': portfolio_std,
                    'sharpe_ratio': sharpe,
                    'optimization_result': result
                }
            else:
                _display_df(pd.DataFrame([{ 'Detalle': result.message }]), title="⚠️ Optimización MinVol falló")
                
        except Exception as e:
            _display_df(pd.DataFrame([{ 'Error': str(e)}]), title="❌ Error en optimización MinVol")
            
        return None
    
    def optimize_risk_parity(self):
        """Optimiza para paridad de riesgo (Risk Parity)"""
        
        def risk_parity_objective(weights):
            """Función objetivo para Risk Parity"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
            
            # Contribución marginal de riesgo
            marginal_contrib = np.dot(self.returns.cov() * 252, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimizar la diferencia entre contribuciones
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Restricciones
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 0.5) for _ in range(self.n_assets))  # Límites más estrictos
        
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = dict(zip(self.assets, result.x))
                portfolio_return, portfolio_std = self.portfolio_performance(result.x)
                sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
                
                return {
                    'tipo': 'Risk Parity',
                    'success': True,
                    'weights': weights,
                    'rendimiento': portfolio_return,
                    'volatilidad': portfolio_std,
                    'sharpe_ratio': sharpe,
                    'optimization_result': result
                }
                
        except Exception as e:
            _display_df(pd.DataFrame([{ 'Error': str(e)}]), title="❌ Error en Risk Parity")
            
        return None
    
    def create_equal_weight_portfolio(self, assets=None):
        """Crea cartera con pesos iguales"""
        
        if assets is None:
            assets = self.assets
            
        equal_weights = {asset: 1/len(assets) for asset in assets}
        
        # Calcular métricas
        weights_array = np.array([equal_weights[asset] for asset in self.assets])
        portfolio_return, portfolio_std = self.portfolio_performance(weights_array)
        sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
        
        return {
            'tipo': 'Equal Weight',
            'weights': equal_weights,
            'rendimiento': portfolio_return,
            'volatilidad': portfolio_std,
            'sharpe_ratio': sharpe
        }
    
    def create_cap_weighted_portfolio(self, market_caps=None):
        """Crea cartera ponderada por capitalización simulada"""
        
        if market_caps is None:
            # Crear pesos simulados por capitalización
            market_caps = {}
            for asset in self.assets:
                if asset.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                    market_caps[asset] = 4.0  # Mega caps
                elif asset.upper() in ['SPY', 'QQQ', 'VTI']:
                    market_caps[asset] = 3.5  # ETFs grandes
                elif asset.upper() in ['GGAL', 'PAMP', 'YPF']:
                    market_caps[asset] = 2.5  # Large caps argentinas
                elif asset.upper() in ['BBAR', 'TXAR']:
                    market_caps[asset] = 2.0  # Mid caps argentinas
                else:
                    market_caps[asset] = 1.0  # Resto
        
        # Normalizar pesos
        total_cap = sum(market_caps.values())
        cap_weights = {asset: cap / total_cap for asset, cap in market_caps.items()}
        
        # Calcular métricas
        weights_array = np.array([cap_weights[asset] for asset in self.assets])
        portfolio_return, portfolio_std = self.portfolio_performance(weights_array)
        sharpe = (portfolio_return - self.rf_rate * 252) / portfolio_std
        
        return {
            'tipo': 'Cap Weighted',
            'weights': cap_weights,
            'rendimiento': portfolio_return,
            'volatilidad': portfolio_std,
            'sharpe_ratio': sharpe
        }


class BacktestEngine:
    """Motor de backtesting para validación de estrategias - MEJORADO PARA DATASET INTEGRADO"""
    
    def __init__(self, data_df, risk_free_rate=0.02):
        """
        Inicializa el motor de backtesting
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            DataFrame con precios históricos (índice: fechas, columnas: activos)
            O DataFrame con retornos precalculados
        risk_free_rate : float
            Tasa libre de riesgo anual
        """
        
        # Detectar tipo de datos
        init_rows = []
        if self._is_price_data(data_df):
            print("💹 Backtesting con datos de precios - calculando retornos...")
            self.prices = data_df.copy()
            self.returns = data_df.pct_change().dropna()
        else:
            print("📊 Backtesting con retornos precalculados...")
            self.returns = data_df.copy()
            self.prices = None
            
        self.rf_rate = risk_free_rate / 252
        init_rows.append({'Activos': len(self.returns.columns), 'Periodos': len(self.returns)})
        _display_df(pd.DataFrame(init_rows), title="⚡ BacktestEngine inicializado")
        
    def _is_price_data(self, data):
        """Detecta si los datos son precios o retornos"""
        avg_values = data.mean().mean()
        negative_ratio = (data < 0).sum().sum() / (data.count().sum())
        return avg_values > 1 and negative_ratio < 0.3
        
    def calculate_portfolio_returns(self, weights_dict, start_idx=0):
        """Calcula retornos históricos de una cartera - MEJORADO PARA DATASET INTEGRADO"""
        
        # Usar datos de retornos
        returns_data = self.returns.iloc[start_idx:].copy()
        portfolio_returns = []
        dates = []
        
        # Filtrar activos válidos
        valid_assets = [asset for asset in weights_dict.keys() if asset in returns_data.columns]
        if not valid_assets:
            print("⚠️ No se encontraron activos válidos en los datos")
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        # Normalizar pesos solo para activos válidos
        total_weight = sum(weights_dict[asset] for asset in valid_assets)
        if total_weight == 0:
            print("⚠️ Suma de pesos es cero")
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        normalized_weights = {asset: weights_dict[asset] / total_weight for asset in valid_assets}
        
        for i, (date, row) in enumerate(returns_data.iterrows()):
            # Calcular retorno ponderado del día
            day_return = 0
            
            for asset in valid_assets:
                if not pd.isna(row[asset]):
                    day_return += normalized_weights[asset] * row[asset]
            
            portfolio_returns.append(day_return)
            dates.append(date)
            
        return pd.Series(portfolio_returns, index=dates), pd.Series(dates)
    
    def backtest_strategy(self, weights_dict, strategy_name):
        """Ejecuta backtesting completo de una estrategia - MEJORADO"""
        
        # Calcular retornos
        returns, dates = self.calculate_portfolio_returns(weights_dict)
        
        if len(returns) == 0:
            return None
            
        # Convertir a numpy array si es necesario
        if isinstance(returns, pd.Series):
            returns_array = returns.values
            dates_array = returns.index.tolist()
        else:
            returns_array = returns
            dates_array = dates
            
        # Calcular valor acumulado (base 100)
        cumulative_value = np.cumprod(1 + returns_array) * 100
        
        # Métricas de performance
        annual_return = np.mean(returns_array) * 252
        annual_volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = (annual_return - self.rf_rate * 252) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(cumulative_value)
        drawdown = (cumulative_value - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # VaR 95%
        var_95 = np.percentile(returns_array, 5)
        
        return {
            'nombre': strategy_name,
            'fechas': dates_array,
            'valores': cumulative_value.tolist(),
            'retornos': returns_array,
            'rendimiento_anual': annual_return,
            'volatilidad_anual': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'valor_final': cumulative_value[-1],
            'total_return': (cumulative_value[-1] / 100) - 1
        }
    
    def compare_strategies(self, strategies_dict):
        """Compara múltiples estrategias"""
        
        results = []
        
        progress_rows = []
        for name, weights in strategies_dict.items():
            result = self.backtest_strategy(weights, name)
            status = 'OK' if result else 'Error'
            progress_rows.append({'Estrategia': name, 'Estado': status, 'Activos': len(weights)})
            if result:
                results.append(result)
        if progress_rows:
            _display_df(pd.DataFrame(progress_rows), title="📊 PROGRESO BACKTESTING")
                
        return results
    
    def create_benchmark_strategies(self, portfolio_assets):
        """Crea estrategias benchmark basadas en los activos de la cartera"""
        
        benchmarks = {}
        
        # Equal Weight
        equal_weights = {asset: 1/len(portfolio_assets) for asset in portfolio_assets}
        benchmarks['Equal Weight (Portfolio Assets)'] = equal_weights
        
        # Clasificar activos por región
        argentinos = []
        internacionales = []
        
        for asset in portfolio_assets:
            asset_upper = asset.upper()
            if asset_upper in ['GGAL', 'PAMP', 'TXAR', 'BBAR', 'YPF', 'ALUA', 'MIRG', 'TECO2', 'CEPU', 'METRO', 'BHIL']:
                argentinos.append(asset)
            else:
                internacionales.append(asset)
        
        # Benchmark Argentina/Internacional
        if len(argentinos) > 0 and len(internacionales) > 0:
            balanced_60_40 = {}
            peso_arg = 0.6 / len(argentinos)
            peso_int = 0.4 / len(internacionales)
            
            for asset in argentinos:
                balanced_60_40[asset] = peso_arg
            for asset in internacionales:
                balanced_60_40[asset] = peso_int
                
            benchmarks['Argentina 60/40 Balance'] = balanced_60_40
        
        # Cap-weighted simulado
        cap_weights = {}
        market_caps = {}
        
        for asset in portfolio_assets:
            asset_upper = asset.upper()
            if asset_upper in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                market_caps[asset] = 4.0
            elif asset_upper in ['SPY', 'QQQ', 'VTI']:
                market_caps[asset] = 3.5
            elif asset_upper in ['GGAL', 'PAMP', 'YPF']:
                market_caps[asset] = 2.5
            else:
                market_caps[asset] = 1.0
        
        total_cap = sum(market_caps.values())
        for asset, cap in market_caps.items():
            cap_weights[asset] = cap / total_cap
            
        benchmarks['Cap Weighted (Simulated)'] = cap_weights
        
        return benchmarks
    
    def generate_performance_summary(self, backtest_results):
        """Genera resumen de performance"""
        
        if not backtest_results:
            return pd.DataFrame()
            
        summary_data = []
        
        for result in backtest_results:
            summary_data.append({
                'Estrategia': result['nombre'],
                'Rendimiento_Anual_%': f"{result['rendimiento_anual']*100:.2f}",
                'Volatilidad_%': f"{result['volatilidad_anual']*100:.2f}",
                'Sharpe_Ratio': f"{result['sharpe_ratio']:.3f}",
                'Max_Drawdown_%': f"{result['max_drawdown']*100:.2f}",
                'VaR_95_%': f"{result['var_95']*100:.2f}",
                'Valor_Final': f"{result['valor_final']:.1f}",
                'Retorno_Total_%': f"{result['total_return']*100:.1f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Ordenar por Sharpe Ratio
        df_summary['Sharpe_Numeric'] = [float(x) for x in df_summary['Sharpe_Ratio']]
        df_summary = df_summary.sort_values('Sharpe_Numeric', ascending=False).drop('Sharpe_Numeric', axis=1)
        
        return df_summary

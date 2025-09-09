"""
Simulation Module - Simulaciones Basadas en Cartera Real
=======================================================

Módulo ENFOCADO en simular el futuro de TU cartera real:
- Simulaciones Monte Carlo basadas en tu composición actual
- Proyecciones de valor futuro de tu cartera específica
- Stress testing de tu cartera real ante escenarios adversos
- Análisis probabilístico del futuro de TU cartera
- Escenarios específicos aplicados a tu composición real

OBJETIVO: Proyectar el comportamiento futuro de TU cartera específica.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Utilidad display unificada
from ..display_utils import _display_df

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ Librería ARCH no disponible - simulaciones sin GARCH")


class MonteCarloSimulator:
    """Simulador Monte Carlo avanzado para carteras"""
    
    def __init__(self, risk_free_rate=0.04):
        self.rf_rate = risk_free_rate
        np.random.seed(42)
        
        # Escenarios específicos para Argentina
        self.argentina_scenarios = {
            'Crisis Cambiaria': {'shock': -0.50, 'duration': 90, 'prob': 0.15},
            'Devaluación Fuerte': {'shock': -0.35, 'duration': 60, 'prob': 0.20},
            'Crisis Política': {'shock': -0.40, 'duration': 120, 'prob': 0.10},
            'Recesión Local': {'shock': -0.25, 'duration': 180, 'prob': 0.25},
            'Default Soberano': {'shock': -0.60, 'duration': 365, 'prob': 0.08}
        }
        
        # Clasificación de activos para Argentina
        self.asset_classifications = {
            'argentina_stocks': ['GGAL', 'PAMP', 'TXAR', 'BBAR', 'YPF', 'ALUA', 'CEPU', 'METRO', 'BHIL'],
            'us_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'IBM'],
            'international_etf': ['SPY', 'QQQ', 'VTI', 'EWZ'],
            'commodities': ['GOLD', 'GLD', 'SLV']
        }
        
    def classify_asset(self, asset_name):
        """Clasifica activo según el contexto argentino"""
        asset_upper = asset_name.upper()
        
        for category, assets in self.asset_classifications.items():
            if asset_upper in assets:
                return category
                
        return 'us_stocks'  # Default
    
    def calculate_portfolio_returns(self, weights_dict, data_df):
        """Calcula retornos históricos de la cartera"""
        
        portfolio_returns = []
        
        for i in range(1, len(data_df)):
            day_return = 0
            valid_weights_sum = 0
            
            for asset, weight in weights_dict.items():
                if asset in data_df.columns and weight > 0.001:
                    current_price = data_df.iloc[i][asset]
                    previous_price = data_df.iloc[i-1][asset]
                    
                    if pd.notna(current_price) and pd.notna(previous_price) and previous_price != 0:
                        asset_return = (current_price / previous_price) - 1
                        day_return += weight * asset_return
                        valid_weights_sum += weight
            
            if valid_weights_sum > 0:
                day_return = day_return / valid_weights_sum
                portfolio_returns.append(day_return)
        
        return np.array(portfolio_returns)
    
    def fit_garch_model(self, returns_series):
        """Ajusta modelo GARCH a serie de retornos"""
        
        if not ARCH_AVAILABLE:
            return None
            
        try:
            # Limpiar datos
            returns_clean = returns_series.dropna()
            returns_clean = returns_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_clean) < 5:  # Reducir requerimiento
                return None
            
            # Convertir a porcentajes para mejor convergencia
            returns_pct = returns_clean * 100
            
            # Ajustar modelo GARCH(1,1)
            model = arch_model(returns_pct, vol='GARCH', p=1, q=1, rescale=False)
            fitted_model = model.fit(disp='off', show_warning=False)
            
            return fitted_model
            
        except Exception:
            return None
    
    def simulate_portfolio_paths(self, portfolio_returns, time_horizons, num_sims=10000, use_garch=True):
        """Simula trayectorias futuras de la cartera"""
        print(f"🔮 Simulando {num_sims:,} trayectorias...")
        
        # Estadísticas básicas
        daily_mean = np.mean(portfolio_returns)
        daily_std = np.std(portfolio_returns)
        
        # Aplicar límites conservadores
        if daily_mean > 0.0015:  # > 38% anual
            daily_mean = 0.0012  # ~30% anual máximo
            
        if daily_std > 0.025:  # > 40% anual
            daily_std = 0.022  # ~35% anual máximo
        
        _display_df(pd.DataFrame([{
            'Retorno_Diario_%': round(daily_mean*100,4),
            'Volatilidad_Diaria_%': round(daily_std*100,3)
        }]), title="📊 ESTADÍSTICAS DIARIAS")
        
        # Intentar ajustar GARCH
        garch_model = None
        if use_garch and ARCH_AVAILABLE and len(portfolio_returns) > 10:  # Reducir requerimiento
            garch_model = self.fit_garch_model(pd.Series(portfolio_returns))
            model_status = 'Modelo GARCH' if garch_model else 'Normal'
            _display_df(pd.DataFrame([{ 'Modelo_Volatilidad': model_status }]), title="⚙️ MODELO SELECCIONADO")
        
        results = {}
        
        horizon_rows = []
        for days in time_horizons:
            horizon_rows.append({'Horizonte_Dias': days, 'Meses': days//21})
            
            simulation_matrix = np.zeros((num_sims, days))
            
            for sim in range(num_sims):
                if garch_model and days >= 21:
                    # Usar GARCH para predicciones
                    try:
                        forecast = garch_model.forecast(horizon=min(days, 100), reindex=False)
                        garch_vol = np.sqrt(forecast.variance.values[-1, :min(days, len(forecast.variance.values[-1, :]))])
                        
                        # Extender si es necesario
                        if len(garch_vol) < days:
                            last_vol = garch_vol[-1]
                            garch_vol = np.concatenate([garch_vol, np.full(days - len(garch_vol), last_vol)])
                        
                        # Generar retornos con volatilidad GARCH
                        for day in range(days):
                            vol_day = garch_vol[day] / 100  # Convertir de % a decimal
                            ret_day = np.random.normal(daily_mean, vol_day)
                            simulation_matrix[sim, day] = ret_day
                            
                    except:
                        # Fallback a normal
                        simulation_matrix[sim, :] = np.random.normal(daily_mean, daily_std, days)
                        
                elif len(portfolio_returns) >= days * 2:
                    # Bootstrap histórico para períodos cortos
                    start_idx = np.random.randint(0, len(portfolio_returns) - days)
                    bootstrap_segment = portfolio_returns[start_idx:start_idx + days]
                    noise = np.random.normal(0, daily_std * 0.05, days)
                    simulation_matrix[sim, :] = bootstrap_segment + noise
                    
                else:
                    # Distribución normal estándar
                    simulation_matrix[sim, :] = np.random.normal(daily_mean, daily_std, days)
                
                # Aplicar límites conservadores
                simulation_matrix[sim, :] = np.clip(simulation_matrix[sim, :], -0.06, 0.06)
            
            # Calcular valores finales
            cumulative_returns = np.cumprod(1 + simulation_matrix, axis=1)
            final_values = cumulative_returns[:, -1]
            period_returns = final_values - 1
            
            # Estadísticas
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            period_percentiles = np.percentile(period_returns, percentiles)
            
            # Probabilidades de pérdida
            prob_loss = np.sum(period_returns < 0) / len(period_returns)
            prob_loss_10 = np.sum(period_returns < -0.10) / len(period_returns)
            prob_loss_20 = np.sum(period_returns < -0.20) / len(period_returns)
            
            results[days] = {
                'period_returns': period_returns,
                'percentiles': dict(zip(percentiles, period_percentiles)),
                'prob_loss': prob_loss,
                'prob_loss_10': prob_loss_10,
                'prob_loss_20': prob_loss_20,
                'expected_return': np.mean(period_returns),
                'volatility': np.std(period_returns),
                'paths': cumulative_returns
            }
        
        if horizon_rows:
            _display_df(pd.DataFrame(horizon_rows), title="🎯 HORIZONTES SIMULADOS")
        return results
    
    def stress_test_scenarios(self, portfolio_returns, weights_dict=None):
        """Ejecuta stress testing con escenarios específicos para Argentina"""
        print("🎯 Ejecutando stress testing con escenarios argentinos...")

        stress_results = {}
        progress_rows = []
        for scenario_name, params in self.argentina_scenarios.items():
            shock = params['shock']
            duration = min(params['duration'], len(portfolio_returns))
            
            # Aplicar shock
            stressed_returns = portfolio_returns.copy()
            
            # Aplicar shock gradual durante la duración
            shock_daily = shock / duration
            for i in range(min(duration, len(stressed_returns))):
                stressed_returns[-(i+1)] += shock_daily
            
            # Calcular métricas del escenario estresado
            cumulative_stressed = np.cumprod(1 + stressed_returns)
            final_value = cumulative_stressed[-1]
            total_return = final_value - 1
            
            # Máximo drawdown durante el stress
            running_max = np.maximum.accumulate(cumulative_stressed)
            drawdown = (cumulative_stressed - running_max) / running_max
            max_dd_stress = np.min(drawdown)
            
            # VaR durante el período de stress
            stress_period_returns = stressed_returns[-duration:] if duration <= len(stressed_returns) else stressed_returns
            var_stress = np.percentile(stress_period_returns, 5)
            
            stress_results[scenario_name] = {
                'shock_applied': shock,
                'duration_days': duration,
                'probability': params['prob'],
                'final_return': total_return,
                'max_drawdown': max_dd_stress,
                'var_95_stress': var_stress,
                'stressed_returns': stressed_returns
            }
            
            progress_rows.append({
                'Escenario': scenario_name,
                'Retorno_Final_%': round(total_return*100,1),
                'Max_DD_%': round(max_dd_stress*100,1),
                'Duracion_Dias': duration
            })
        if progress_rows:
            _display_df(pd.DataFrame(progress_rows), title="🧪 RESULTADOS STRESS TEST")
        return stress_results
    
    def generate_probability_analysis(self, simulation_results):
        """Genera análisis probabilístico detallado"""
        print("📊 Generando análisis probabilístico...")
        
        analysis = {}
        
        for horizon_days, results in simulation_results.items():
            period_months = horizon_days // 21
            returns = results['period_returns']
            
            # Análisis detallado de probabilidades
            prob_analysis = {
                'horizon_days': horizon_days,
                'horizon_months': period_months,
                'expected_return': results['expected_return'],
                'volatility': results['volatility'],
                'prob_positive': np.sum(returns > 0) / len(returns),
                'prob_loss': results['prob_loss'],
                'prob_loss_5': np.sum(returns < -0.05) / len(returns),
                'prob_loss_10': results['prob_loss_10'],
                'prob_loss_20': results['prob_loss_20'],
                'prob_gain_10': np.sum(returns > 0.10) / len(returns),
                'prob_gain_20': np.sum(returns > 0.20) / len(returns),
                'prob_gain_50': np.sum(returns > 0.50) / len(returns),
                'best_case_5': np.percentile(returns, 95),
                'worst_case_5': np.percentile(returns, 5),
                'median_return': np.percentile(returns, 50)
            }
            
            analysis[f"{period_months}M"] = prob_analysis
        
        return analysis
    
    def create_simulation_summary(self, simulation_results, stress_results=None):
        """Crea resumen completo de simulaciones"""
        print("📋 Creando resumen de simulaciones...")
        
        summary_data = []
        
        for horizon_days, results in simulation_results.items():
            period_months = horizon_days // 21
            
            summary_data.append({
                'Horizonte': f"{period_months}M",
                'Días': horizon_days,
                'Retorno_Esperado_%': f"{results['expected_return']*100:.1f}",
                'Volatilidad_%': f"{results['volatility']*100:.1f}",
                'Prob_Pérdida': f"{results['prob_loss']:.1%}",
                'Prob_Pérdida_>10%': f"{results['prob_loss_10']:.1%}",
                'Prob_Pérdida_>20%': f"{results['prob_loss_20']:.1%}",
                'VaR_95_%': f"{results['percentiles'][5]*100:.1f}",
                'VaR_99_%': f"{results['percentiles'][1]*100:.1f}",
                'Mejor_Caso_5%': f"{results['percentiles'][95]*100:.1f}",
                'Peor_Caso_5%': f"{results['percentiles'][5]*100:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def run_comprehensive_simulation(self, weights_dict, data_df, time_horizons=[21, 63, 126, 252], num_sims=10000):
        """Ejecuta simulación comprehensiva completa"""
        
        meta_df = pd.DataFrame([{
            'Activos': len(weights_dict),
            'Lista_Activos': ', '.join(weights_dict.keys()),
            'Horizontes_Meses': ', '.join([f"{h//21}M" for h in time_horizons]),
            'Simulaciones': num_sims
        }])
        _display_df(meta_df, title="🚀 PARÁMETROS SIMULACIÓN")
        
        # 1. Calcular retornos históricos de la cartera
        portfolio_returns = self.calculate_portfolio_returns(weights_dict, data_df)
        
        if len(portfolio_returns) < 5:  # Reducir requerimiento
            _display_df(pd.DataFrame([{ 'Error': 'Muy pocos datos históricos para simulación'}]), title="❌ ERROR")
            return None
            
        _display_df(pd.DataFrame([{ 'Retornos_Históricos': len(portfolio_returns)}]), title="📈 DATOS HISTÓRICOS")
        
        # 2. Simulación Monte Carlo
        simulation_results = self.simulate_portfolio_paths(
            portfolio_returns, time_horizons, num_sims, use_garch=True
        )
        
        # 3. Stress Testing
        stress_results = self.stress_test_scenarios(portfolio_returns, weights_dict)
        
        # 4. Análisis probabilístico
        probability_analysis = self.generate_probability_analysis(simulation_results)
        
        # 5. Resumen
        summary_df = self.create_simulation_summary(simulation_results, stress_results)
        print("✅ Simulación completada exitosamente")
        return {
                'simulation_results': simulation_results,
                'stress_results': stress_results,
                'probability_analysis': probability_analysis,
                'summary_df': summary_df,
                'portfolio_returns': portfolio_returns,
                'garch_available': ARCH_AVAILABLE
            }


class StressTestEngine:
    """Motor especializado en stress testing"""
    
    def __init__(self):
        # Escenarios macro específicos
        self.macro_scenarios = {
            'Recesión Global': {
                'equity_shock': -0.30, 'bond_shock': 0.05, 'fx_shock': 0.20, 'duration': 180
            },
            'Crisis Argentina': {
                'equity_shock': -0.50, 'bond_shock': -0.15, 'fx_shock': 0.60, 'duration': 120
            },
            'Hiperinflación': {
                'equity_shock': -0.25, 'bond_shock': -0.30, 'fx_shock': 1.00, 'duration': 365
            },
            'Crisis Energética': {
                'equity_shock': -0.20, 'bond_shock': 0.02, 'fx_shock': 0.15, 'duration': 90
            }
        }
    
    def apply_correlation_stress(self, returns_matrix, correlation_shock=0.8):
        """Aplica stress de correlación (todas las correlaciones tienden a 1)"""
        
        # Calcular matriz de correlación original
        original_corr = returns_matrix.corr()
        
        # Crear matriz de correlación estresada
        n_assets = len(original_corr)
        stressed_corr = original_corr.copy()
        
        # Aumentar todas las correlaciones hacia el shock level
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    original_val = original_corr.iloc[i, j]
                    stressed_corr.iloc[i, j] = original_val + (correlation_shock - original_val) * 0.7
        
        # Asegurar que la matriz sea semidefinida positiva
        eigenvals, eigenvecs = np.linalg.eigh(stressed_corr)
        eigenvals = np.maximum(eigenvals, 0.01)
        stressed_corr_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return pd.DataFrame(stressed_corr_fixed, index=original_corr.index, columns=original_corr.columns)
    
    def generate_stressed_returns(self, original_returns, stress_corr_matrix, shock_factor=1.5):
        """Genera retornos estresados usando nueva matriz de correlación"""
        
        # Descomponer retornos originales
        mean_returns = original_returns.mean()
        std_returns = original_returns.std()
        
        # Generar nuevos retornos correlacionados
        n_periods = len(original_returns)
        n_assets = len(original_returns.columns)
        
        # Cholesky decomposition de la matriz de correlación estresada
        try:
            L = np.linalg.cholesky(stress_corr_matrix.values)
        except np.linalg.LinAlgError:
            # Si falla, usar la matriz original
            L = np.linalg.cholesky(original_returns.corr().values)
        
        # Generar retornos independientes
        independent_returns = np.random.normal(0, 1, (n_periods, n_assets))
        
        # Aplicar correlación
        correlated_returns = independent_returns @ L.T
        
        # Escalar y centrar
        stressed_returns = pd.DataFrame(
            correlated_returns,
            columns=original_returns.columns,
            index=original_returns.index
        )
        
        for col in stressed_returns.columns:
            stressed_returns[col] = (
                stressed_returns[col] * std_returns[col] * shock_factor + 
                mean_returns[col] * 0.5  # Reducir retorno esperado
            )
        
        return stressed_returns

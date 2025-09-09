"""
Data Loader Module - Carga y Preparación de Datos
================================================

Módulo CONSOLIDADO responsable de:
- PRIMARIO: Cargar y analizar la composición real de la cartera diaria
- SECUNDARIO: Cargar datos auxiliares para benchmarks (si se necesitan)
- Analizar reportes de movimientos reales
- Integrar y limpiar datos
- Convertir CEDEARs de USD a ARS

ENFOQUE PRINCIPAL: portfolio_consolidado_completo.csv - Serie diaria de tu cartera real
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Utilidad de display unificada
from ..display_utils import _display_df



class PortfolioCompositionLoader:
    """CARGADOR PRINCIPAL - Se enfoca en la composición real de tu cartera"""
    
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.output_path = self.base_path / "outputs"
        
        # DATOS PRINCIPALES - Tu cartera real
        self.df_portfolio_daily = pd.DataFrame()  # Serie diaria de tu cartera
        self.df_movements = pd.DataFrame()        # Movimientos reales
        self.real_positions = {}                  # Posiciones reales calculadas
        
        # DATOS AUXILIARES - Solo si se necesitan
        self.df_benchmarks = pd.DataFrame()       # Precios auxiliares para benchmarks
        
        # METADATOS
        self.portfolio_composition = {}           # Composición diaria
        self.portfolio_weights_history = pd.DataFrame()  # Historia de pesos

        # Mapeo puntual para normalizar nombres extensos a ticker (solo activos solicitados)
        self._instrument_normalization_map = {
            'BANCO HIPOTECARIO ESCRIT.  D  (CATEGORIAS 1 Y 2) 3 VOTOS (BHIP)': 'BHIP',
            'METROGAS S.A. ESCRIT.  B  1 VOTO (METR)': 'METR'
        }

    def _normalize_instrument_names(self):
        """Normaliza nombres largos de instrumentos a su ticker para consistencia.
        Solo aplica a los activos especificados en _instrument_normalization_map.
        """
        if self.df_portfolio_daily.empty or 'instrumento' not in self.df_portfolio_daily.columns:
            return
        # Reemplazar nombres
        self.df_portfolio_daily['instrumento'] = self.df_portfolio_daily['instrumento'].replace(self._instrument_normalization_map)
        
    def load_portfolio_daily_data(self, filename="portfolio_consolidado_completo.csv"):
        """FUNCIÓN PRINCIPAL: Carga la serie diaria de tu cartera real"""
        
        file_path = self.output_path / filename
        
        if not file_path.exists():
            print(f"❌ No se encontró {filename} en {self.output_path}")
            print("💡 Este es el archivo PRINCIPAL que contiene tu cartera diaria")
            return False
            
        print(f"💼 CARGANDO CARTERA REAL DIARIA...")
        print("=" * 60)
        
        try:
            # Cargar datos de la cartera
            self.df_portfolio_daily = pd.read_csv(file_path, sep=';')
            self.df_portfolio_daily['fecha'] = pd.to_datetime(self.df_portfolio_daily['fecha'])

            # Normalizar nombres de instrumentos solicitados
            self._normalize_instrument_names()
            
            print(f"✅ CARTERA DIARIA CARGADA:")
            print(f"   📊 Total registros: {len(self.df_portfolio_daily)}")
            print(f"   📅 Período: {self.df_portfolio_daily['fecha'].min().date()} → {self.df_portfolio_daily['fecha'].max().date()}")
            print(f"   🎯 Instrumentos únicos: {self.df_portfolio_daily['instrumento'].nunique()}")
            
            # Mostrar instrumentos
            instrumentos = (
                self.df_portfolio_daily
                .groupby('instrumento')
                .agg(
                    dias_presencia=('fecha', 'count'),
                    primera_fecha=('fecha', 'min'),
                    ultima_fecha=('fecha', 'max'),
                    valor_total_acum=('total', 'sum')
                )
                .sort_values('dias_presencia', ascending=False)
            )
            instrumentos['primera_fecha'] = instrumentos['primera_fecha'].dt.date
            instrumentos['ultima_fecha'] = instrumentos['ultima_fecha'].dt.date
            instrumentos['valor_total_acum'] = instrumentos['valor_total_acum'].round(0).astype(int)
            _display_df(
                instrumentos.reset_index(),
                title="🎯 INSTRUMENTOS EN TU CARTERA (resumen)",
            )
            
            # Calcular composición diaria
            self._calculate_daily_composition()
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando cartera diaria: {e}")
            return False
    
    def _calculate_daily_composition(self):
        """Calcula la composición (pesos) diaria de la cartera"""
        print("📊 CALCULANDO COMPOSICIÓN DIARIA...")

        # Agrupar por fecha + instrumento por si la normalización generó duplicados
        df_grouped = (
            self.df_portfolio_daily
            .groupby(['fecha', 'instrumento'], as_index=False)
            .agg({
                'cantidad': 'sum',
                'precio': 'last',  # último precio del día
                'total': 'sum'
            })
        )

        # Calcular valor total por día
        daily_totals = df_grouped.groupby('fecha')['total'].sum().reset_index()
        daily_totals.columns = ['fecha', 'valor_total_cartera']

        # Unir con datos originales (agrupados)
        df_with_totals = df_grouped.merge(daily_totals, on='fecha')
        
        # Calcular peso de cada activo por día
        df_with_totals['peso'] = df_with_totals['total'] / df_with_totals['valor_total_cartera']
        df_with_totals['peso_pct'] = df_with_totals['peso'] * 100

        # Crear matriz de pesos por fecha y activo (pivot table)
        weights_pivot = df_with_totals.pivot(index='fecha', columns='instrumento', values='peso').fillna(0)

        self.portfolio_weights_history = weights_pivot
        self.portfolio_composition = df_with_totals

        print(f"✅ Composición calculada para {len(weights_pivot)} días")

        # Mostrar resumen de pesos promedio
        pesos_promedio = (
            weights_pivot
            .mean()
            .to_frame('peso_promedio')
            .sort_values('peso_promedio', ascending=False)
        )
        pesos_promedio['peso_promedio_pct'] = (pesos_promedio['peso_promedio'] * 100).round(2)
        _display_df(
            pesos_promedio[pesos_promedio['peso_promedio'] > 0.001]
            .head(15)
            .reset_index()
            .rename(columns={'instrumento': 'activo'}),
            title="💰 TOP PESOS PROMEDIO (>%0.1%)",
        )
    
    def get_portfolio_evolution(self):
        """Retorna la evolución diaria de la cartera"""
        if self.portfolio_composition.empty:
            print("❌ Primero carga los datos de la cartera")
            return None
            
        return self.portfolio_composition
    
    def get_portfolio_weights_matrix(self):
        """Retorna la matriz de pesos por fecha y activo"""
        if self.portfolio_weights_history.empty:
            print("❌ Primero carga los datos de la cartera")
            return None
            
        return self.portfolio_weights_history
    
    def get_current_composition(self, date=None):
        """Retorna la composición actual (o de una fecha específica)"""
        if self.portfolio_weights_history.empty:
            return {}
        
        if date is None:
            # Última fecha disponible
            last_date = self.portfolio_weights_history.index.max()
            weights = self.portfolio_weights_history.loc[last_date]
        else:
            date = pd.to_datetime(date)
            if date in self.portfolio_weights_history.index:
                weights = self.portfolio_weights_history.loc[date]
            else:
                print(f"❌ Fecha {date.date()} no disponible")
                return {}
        
        # Filtrar solo pesos > 0
        current_weights = weights[weights > 0.001].to_dict()
        
        comp_df = (
            pd.Series(current_weights, name='peso')
            .sort_values(ascending=False)
            .to_frame()
        )
        comp_df['peso_pct'] = (comp_df['peso'] * 100).round(2)
        _display_df(
            comp_df.reset_index().rename(columns={'index': 'activo'}),
            title=f"💼 COMPOSICIÓN {'ACTUAL' if date is None else f'AL {date.date()}'} (pesos)",
        )
        
        return current_weights
    
    def get_total_value_series(self):
        """Retorna la serie temporal del valor total de la cartera"""
        if self.portfolio_composition.empty:
            return pd.Series()
        
        daily_totals = self.portfolio_composition.groupby('fecha')['total'].sum()
        return daily_totals
    
    def load_movements_data(self, filename="movements_report_2025-01-01_2025-09-08.csv"):
        """Carga datos de movimientos reales (secundario)"""
        
        file_paths = [
            self.output_path / filename,
            self.base_path / filename  # También buscar en raíz
        ]
        
        file_path = None
        for path in file_paths:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            print(f"⚠️ No se encontró {filename} (opcional)")
            return False
        
        try:
            self.df_movements = pd.read_csv(file_path, sep=';')
            if 'fechaEjecucion' in self.df_movements.columns:
                self.df_movements['fechaEjecucion'] = pd.to_datetime(
                    self.df_movements['fechaEjecucion'], format='%d-%m-%Y', errors='coerce'
                )
            
            print(f"✅ Movimientos cargados: {len(self.df_movements)} registros")
            
            # Calcular posiciones reales
            self.real_positions = self._calculate_real_positions()
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando movimientos: {e}")
            return False
    
    def _calculate_real_positions(self):
        """Calcula posiciones reales basadas en movimientos"""
        if self.df_movements.empty:
            return {}

        instrument_mapping = {
            'CENTRAL PUERTO S.A. ORD. 1 VOTO ESCRIT (CEPU)': 'CEPU',
            'METROGAS S.A. ESCRIT.  B  1 VOTO (METR)': 'METRO',
            'BANCO HIPOTECARIO ESCRIT.  D  (CATEGORIAS 1 Y 2) 3 VOTOS (BHIP)': 'BHIL',
            'CEDEAR APPLE INC. (AAPL)': 'AAPL',
            'CEDEAR INTERNATIONAL BUSINESS MACHINES CORPORATION (IBM)': 'IBM',
            'CEDEAR SPDR S&P 500 (SPY)': 'SPY',
            'CEDEAR ISHARES MSCI BRAZIL CAP (EWZ)': 'EWZ',
            'FCI ALAMERICA RENTA MIXTA CL.A ESC (COCORMA)': 'COCORMA'
        }

        movimientos = self.df_movements.copy()
        if 'tipoOperacion' in movimientos.columns:
            movimientos['tipoOperacion'] = movimientos['tipoOperacion'].replace({
                'Suscripción': 'Compra', 'Rescate': 'Venta',
                'SUSCRIPCION': 'Compra', 'RESCATE': 'Venta'
            })

        asset_ops = movimientos[
            movimientos['tipoOperacion'].isin(['Compra', 'Venta']) & movimientos['instrumento'].notna()
        ].copy()
        asset_ops['simbolo'] = asset_ops['instrumento'].map(instrument_mapping)

        positions = {}
        print("💼 CALCULANDO POSICIONES REALES DESDE MOVIMIENTOS:")
        resumen_rows = []
        for symbol in asset_ops['simbolo'].dropna().unique():
            ops = asset_ops[asset_ops['simbolo'] == symbol].copy().sort_values('fechaEjecucion')
            total_compras = ops.loc[ops['tipoOperacion'] == 'Compra', 'total'].abs().sum()
            total_ventas = ops.loc[ops['tipoOperacion'] == 'Venta', 'total'].abs().sum()
            net_position_ars = total_compras - total_ventas
            if net_position_ars > 1000 or symbol == 'COCORMA':
                positions[symbol] = {
                    'valor_ars_actual': net_position_ars,
                    'total_invertido': total_compras,
                    'is_cedear': symbol in ['AAPL', 'IBM', 'SPY', 'EWZ'],
                    'operations': len(ops),
                    'last_date': ops['fechaEjecucion'].max()
                }
                resumen_rows.append({
                    'symbol': symbol,
                    'valor_ars_actual': round(net_position_ars, 2),
                    'total_invertido': round(total_compras, 2),
                    'operaciones': len(ops),
                    'ultima_fecha': ops['fechaEjecucion'].max().date(),
                    'is_cedear': symbol in ['AAPL', 'IBM', 'SPY', 'EWZ']
                })

        if resumen_rows:
            resumen_df = (
                pd.DataFrame(resumen_rows)
                .sort_values('valor_ars_actual', ascending=False)
                .reset_index(drop=True)
            )
            _display_df(resumen_df, title="   ✅ POSICIONES REALES (ARS)")
        return positions
    
    def load_benchmark_data(self, filename="precios_completos_ars.csv"):
        """AUXILIAR: Carga datos de benchmarks (solo si se necesitan)"""
        
        file_path = self.output_path / filename
        
        if not file_path.exists():
            print(f"⚠️ No se encontró {filename} (datos auxiliares)")
            print("💡 Este archivo es OPCIONAL para benchmarks adicionales")
            return False
        
        try:
            print("📊 CARGANDO DATOS AUXILIARES PARA BENCHMARKS...")
            
            self.df_benchmarks = pd.read_csv(file_path, sep=';', index_col=0)
            self.df_benchmarks.index = pd.to_datetime(self.df_benchmarks.index)
            
            # Remover USD_ARS_Rate
            if 'USD_ARS_Rate' in self.df_benchmarks.columns:
                self.usd_ars_rate = self.df_benchmarks['USD_ARS_Rate'].copy()
                self.df_benchmarks = self.df_benchmarks.drop('USD_ARS_Rate', axis=1)
            
            print(f"✅ Benchmarks auxiliares cargados:")
            print(f"   📊 Dimensiones: {self.df_benchmarks.shape}")
            print(f"   📅 Período: {self.df_benchmarks.index.min().date()} → {self.df_benchmarks.index.max().date()}")
            print(f"   🎯 Activos disponibles: {len(self.df_benchmarks.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando benchmarks auxiliares: {e}")
            return False
    
    def get_analysis_ready_data(self, include_benchmarks=False, max_benchmarks=10):
        """Retorna datos listos para análisis"""
        
        if self.df_portfolio_daily.empty:
            print("❌ Primero carga los datos principales de la cartera")
            return None, None
        
        # Crear series de precios desde tu cartera
        portfolio_prices = self.df_portfolio_daily.pivot(
            index='fecha', columns='instrumento', values='precio'
        )
        
        print("📊 DATOS PREPARADOS PARA ANÁLISIS:")
        print(f"   💼 Precios de cartera: {portfolio_prices.shape}")

        # Normalizar timezones (convertir todo a naive)
        if hasattr(portfolio_prices.index, 'tz') and portfolio_prices.index.tz is not None:
            portfolio_prices.index = portfolio_prices.index.tz_localize(None)
        if not self.df_benchmarks.empty and hasattr(self.df_benchmarks.index, 'tz') and self.df_benchmarks.index.tz is not None:
            self.df_benchmarks.index = self.df_benchmarks.index.tz_localize(None)
        
        # Si se solicitan benchmarks y están disponibles
        if include_benchmarks and not self.df_benchmarks.empty:
            
            # Encontrar activos comunes para alineación temporal
            # Asegurar índices sin tz
            p_start = portfolio_prices.index.min()
            b_start = self.df_benchmarks.index.min()
            p_end = portfolio_prices.index.max()
            b_end = self.df_benchmarks.index.max()

            common_period_start = max(p_start, b_start)
            common_period_end = min(p_end, b_end)
            
            # Filtrar por período común
            portfolio_aligned = portfolio_prices.loc[common_period_start:common_period_end]
            benchmarks_aligned = self.df_benchmarks.loc[common_period_start:common_period_end]
            
            # Seleccionar algunos benchmarks
            benchmark_cols = benchmarks_aligned.columns[:max_benchmarks]
            combined_data = pd.concat([
                portfolio_aligned,
                benchmarks_aligned[benchmark_cols]
            ], axis=1)
            
            print(f"   🎯 Con benchmarks: {combined_data.shape}")
            print(f"   📅 Período común: {common_period_start.date()} → {common_period_end.date()}")
            
            # Calcular retornos
            returns_data = combined_data.pct_change().dropna()
            
            return combined_data, returns_data
        
        else:
            # Solo datos de la cartera
            returns_data = portfolio_prices.pct_change().dropna()
            return portfolio_prices, returns_data
    
    def get_summary_report(self):
        """Genera reporte resumen de los datos cargados"""
        
        report = []
        report.append("=" * 60)
        report.append("RESUMEN DE DATOS CARGADOS - PORTFOLIO ANALYZER")
        report.append("=" * 60)
        
        # Datos principales
        if not self.df_portfolio_daily.empty:
            report.append("💼 CARTERA REAL DIARIA: ✅")
            report.append(f"   • Registros: {len(self.df_portfolio_daily):,}")
            report.append(f"   • Período: {self.df_portfolio_daily['fecha'].min().date()} → {self.df_portfolio_daily['fecha'].max().date()}")
            report.append(f"   • Instrumentos: {self.df_portfolio_daily['instrumento'].nunique()}")
            
            # Valor total actual
            if not self.portfolio_composition.empty:
                valor_actual = self.get_total_value_series().iloc[-1]
                report.append(f"   • Valor actual: ${valor_actual:,.0f} ARS")
        else:
            report.append("💼 CARTERA REAL DIARIA: ❌ No cargada")
        
        # Movimientos
        if not self.df_movements.empty:
            report.append("📋 MOVIMIENTOS REALES: ✅")
            report.append(f"   • Registros: {len(self.df_movements):,}")
            report.append(f"   • Posiciones calculadas: {len(self.real_positions)}")
        else:
            report.append("📋 MOVIMIENTOS REALES: ❌ No cargados")
        
        # Benchmarks auxiliares
        if not self.df_benchmarks.empty:
            report.append("📊 BENCHMARKS AUXILIARES: ✅")
            report.append(f"   • Activos: {len(self.df_benchmarks.columns):,}")
            report.append(f"   • Período: {self.df_benchmarks.index.min().date()} → {self.df_benchmarks.index.max().date()}")
        else:
            report.append("📊 BENCHMARKS AUXILIARES: ❌ No cargados (opcional)")
        
        report.append("" + "=" * 60)
        
        # Imprimir y retornar
        for line in report:
            print(line)
        
        return "".join(report)


class MovementsAnalyzer:
    """CLASE SIMPLIFICADA para compatibilidad - USA PortfolioCompositionLoader"""
    
    def __init__(self, movements_file=None):
        self.movements_file = movements_file
        self.loader = PortfolioCompositionLoader()
        self.positions = {}
    
    def calculate_real_positions(self, movements_df):
        """Calcula posiciones reales - delegado a PortfolioCompositionLoader"""
        self.loader.df_movements = movements_df
        self.positions = self.loader._calculate_real_positions()
        return self.positions
    
    def get_portfolio_weights(self):
        """Calcula pesos de cartera basados en posiciones reales"""
        if not self.positions:
            return {}
        
        total_value = sum([pos['valor_ars_actual'] for pos in self.positions.values()])
        weights = {}
        
        print(f"\💰 CALCULANDO PESOS DE CARTERA (Total: ${total_value:,.0f} ARS):")
        
        for symbol, data in self.positions.items():
            weight = data['valor_ars_actual'] / total_value
            weights[symbol] = weight
            print(f"   • {symbol}: ${data['valor_ars_actual']:,.0f} ARS ({weight:.1%})")
        
        return weights



    """Analiza reportes de movimientos reales de la cartera"""
    
    def __init__(self, movements_file=None):
        self.movements_file = movements_file
        self.positions = {}
        
    def load_movements_report(self, file_path):
        """Carga reporte de movimientos desde CSV"""
        try:
            df = pd.read_csv(file_path, sep=';')
            df['fechaEjecucion'] = pd.to_datetime(df['fechaEjecucion'], format='%d-%m-%Y')
            
            print(f"✅ Movimientos cargados: {len(df)} registros")
            print(f"📅 Período: {df['fechaEjecucion'].min().date()} → {df['fechaEjecucion'].max().date()}")
            
            return df
        except Exception as e:
            print(f"❌ Error cargando movimientos: {str(e)}")
            return pd.DataFrame()
    
    def calculate_real_positions(self, movements_df):
        """Calcula posiciones reales basadas en transacciones"""
        
        # Mapeo de instrumentos a símbolos
        instrument_mapping = {
            'CENTRAL PUERTO S.A. ORD. 1 VOTO ESCRIT (CEPU)': 'CEPU',
            'METROGAS S.A. ESCRIT.  B  1 VOTO (METR)': 'METRO', 
            'BANCO HIPOTECARIO ESCRIT.  D  (CATEGORIAS 1 Y 2) 3 VOTOS (BHIP)': 'BHIL',
            'CEDEAR APPLE INC. (AAPL)': 'AAPL',
            'CEDEAR INTERNATIONAL BUSINESS MACHINES CORPORATION (IBM)': 'IBM',
            'CEDEAR SPDR S&P 500 (SPY)': 'SPY',
            'CEDEAR ISHARES MSCI BRAZIL CAP (EWZ)': 'EWZ',
            'FCI ALAMERICA RENTA MIXTA CL.A ESC (COCORMA)': 'COCORMA'  # ¡AGREGADO!
        }
        
        # Filtrar operaciones de activos
        asset_ops = movements_df[
            movements_df['tipoOperacion'].isin(['Compra', 'Venta']) & 
            movements_df['instrumento'].notna()
        ].copy()
        
        asset_ops['simbolo'] = asset_ops['instrumento'].map(instrument_mapping)
        
        positions = {}
        
        print("💼 CALCULANDO POSICIONES REALES:")
        
        for symbol in asset_ops['simbolo'].dropna().unique():
            ops = asset_ops[asset_ops['simbolo'] == symbol].copy()
            ops = ops.sort_values('fechaEjecucion')
            
            # Calcular posición neta en ARS
            total_compras = 0
            total_ventas = 0
            
            for _, op in ops.iterrows():
                monto = abs(float(op['total']))
                if op['tipoOperacion'] == 'Compra':
                    total_compras += monto
                    print(f"   📈 {symbol}: Compra ${monto:,.0f} ARS ({op['fechaEjecucion']})")
                else:
                    total_ventas += monto
                    print(f"   📉 {symbol}: Venta ${monto:,.0f} ARS ({op['fechaEjecucion']})")
            
            # Posición neta final
            net_position_ars = total_compras - total_ventas
            
            # Clasificar activo
            is_cedear = symbol in ['AAPL', 'IBM', 'SPY', 'EWZ']
            
            if net_position_ars > 1000:  # Solo posiciones significativas
                positions[symbol] = {
                    'valor_ars_actual': net_position_ars,  # Valor REAL de la posición
                    'total_invertido': total_compras,      # Lo que invertiste originalmente
                    'is_cedear': is_cedear,
                    'currency': 'USD' if is_cedear else 'ARS',
                    'operations': len(ops),
                    'last_date': ops['fechaEjecucion'].max()
                }
                
                print(f"   ✅ {symbol}: Posición neta ${net_position_ars:,.0f} ARS (Invertido: ${total_compras:,.0f})")
        
        self.positions = positions
        return positions
    
    def get_portfolio_weights(self):
        """Calcula pesos de cartera basados en posiciones reales"""
        if not self.positions:
            return {}
            
        # Usar el valor real actual de las posiciones (no lo invertido originalmente)
        total_value = sum([pos['valor_ars_actual'] for pos in self.positions.values()])
        weights = {}
        
        print(f"💰 CALCULANDO PESOS DE CARTERA (Total: ${total_value:,.0f} ARS):")
        
        for symbol, data in self.positions.items():
            weight = data['valor_ars_actual'] / total_value
            weights[symbol] = weight
            print(f"   • {symbol}: ${data['valor_ars_actual']:,.0f} ARS ({weight:.1%})")
        
        return weights


class DataLoader:
    """CLASE PRINCIPAL CONSOLIDADA - Maneja carga de cartera real y datos auxiliares"""
    
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Usar la nueva clase principal
        self.portfolio_loader = PortfolioCompositionLoader(base_path)
        
        # Mantener compatibilidad con API anterior
        self.df = pd.DataFrame()
    
    def load_main_portfolio_data(self):
        """MÉTODO PRINCIPAL: Carga la serie diaria de tu cartera"""
        print("🚀 INICIANDO CARGA DE DATOS PRINCIPALES...")
        
        # 1. PRINCIPAL: Cargar serie diaria de la cartera
        success = self.portfolio_loader.load_portfolio_daily_data()
        if not success:
            return False
        
        # 2. SECUNDARIO: Cargar movimientos si están disponibles
        self.portfolio_loader.load_movements_data()
        
        # 3. OPCIONAL: Cargar benchmarks auxiliares si se necesitan
        # self.portfolio_loader.load_benchmark_data()
        
        # 4. Generar reporte resumen
        self.portfolio_loader.get_summary_report()
        
        return True
    
    def get_portfolio_composition(self, date=None):
        """Obtiene la composición de la cartera (pesos por activo)"""
        return self.portfolio_loader.get_current_composition(date)
    
    def get_portfolio_evolution(self):
        """Obtiene la evolución completa de la cartera"""
        return self.portfolio_loader.get_portfolio_evolution()
    
    def get_portfolio_weights_history(self):
        """Obtiene la historia de pesos por fecha"""
        return self.portfolio_loader.get_portfolio_weights_matrix()
    
    def get_total_value_series(self):
        """Obtiene la serie temporal del valor total"""
        return self.portfolio_loader.get_total_value_series()
    
    def get_analysis_data(self, include_benchmarks=False):
        """Obtiene datos listos para análisis"""
        return self.portfolio_loader.get_analysis_ready_data(include_benchmarks)
    
    # MÉTODOS DE COMPATIBILIDAD CON API ANTERIOR
        
    def load_excel_data(self, filename="Datos_historicos_de_la_cartera.xlsx", 
                       end_date="2025-09-05"):
        """Carga datos históricos desde Excel"""
        
        # Buscar archivo
        file_path = self._find_file(filename)
        if not file_path:
            raise FileNotFoundError(f"No se encontró {filename}")
            
        print(f"📊 Cargando datos desde: {file_path}")
        
        # Cargar datos
        df = pd.read_excel(file_path)
        
        # Filtrar hasta fecha límite
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            if end_date:
                df = df[df['Fecha'] <= pd.to_datetime(end_date)]
                print(f"📅 Datos filtrados hasta: {end_date}")
                
        # Limpiar datos
        df = df.ffill().bfill()
        
        self.df = df
        print(f"✅ Datos cargados: {len(df)} registros")
        
        return df
    
    def download_missing_assets(self, missing_assets, cedear_list):
        """Descarga activos faltantes desde yfinance"""
        if not missing_assets:
            print("✅ Todos los activos necesarios disponibles")
            return
            
        print(f"📡 Descargando {len(missing_assets)} activos faltantes...")
        
        start_date = self.df['Fecha'].min()
        end_date = self.df['Fecha'].max()
        
        for asset in missing_assets:
            try:
                print(f"   📊 Descargando {asset}...")
                ticker = yf.Ticker(asset)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Normalizar fechas
                    asset_df = pd.DataFrame({
                        'Fecha': pd.to_datetime(hist.index.date),
                        f'{asset}_USD_Original': hist['Close'].values
                    })
                    asset_df['Fecha'] = asset_df['Fecha'].dt.normalize()
                    self.df['Fecha'] = self.df['Fecha'].dt.normalize()
                    
                    # Merge
                    self.df = pd.merge(self.df, asset_df, on='Fecha', how='left')
                    self.df[f'{asset}_USD_Original'] = self.df[f'{asset}_USD_Original'].ffill()
                    
                    print(f"   ✅ {asset}: {len(hist)} registros")
                else:
                    print(f"   ❌ {asset}: Sin datos")
                    
            except Exception as e:
                print(f"   ❌ Error descargando {asset}: {str(e)}")
                continue
    
    def add_macro_variables(self):
        """Agrega variables macroeconómicas (CCL, tasas BCRA)"""
        
        # CCL histórico
        if 'Tipo_Cambio_ARSUSD' not in self.df.columns:
            self._add_ccl_data()
            
        # Tasas BCRA
        if 'Tasa_PlazoFijo_Diaria' not in self.df.columns:
            self._add_bcra_rates()
    
    def _add_ccl_data(self):
        """Obtiene datos del CCL"""
        try:
            print("💱 Obteniendo CCL histórico...")
            
            url = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                ccl_data = response.json()
                ccl_df = pd.DataFrame(ccl_data)
                ccl_df['fecha'] = pd.to_datetime(ccl_df['fecha'])
                ccl_df = ccl_df[['fecha', 'venta']].rename(
                    columns={'fecha': 'Fecha', 'venta': 'Tipo_Cambio_ARSUSD'}
                )
                
                # Filtrar por rango de fechas
                fecha_min = self.df['Fecha'].min()
                fecha_max = self.df['Fecha'].max()
                ccl_df = ccl_df[
                    (ccl_df['Fecha'] >= fecha_min) & 
                    (ccl_df['Fecha'] <= fecha_max)
                ]
                
                # Merge
                self.df = pd.merge(self.df, ccl_df, on='Fecha', how='left')
                self.df['Tipo_Cambio_ARSUSD'] = self.df['Tipo_Cambio_ARSUSD'].ffill().bfill()
                self.df['Retorno_FX'] = self.df['Tipo_Cambio_ARSUSD'].pct_change()
                
                print(f"✅ CCL agregado: {len(ccl_df)} registros")
                
        except Exception as e:
            print(f"❌ Error obteniendo CCL: {str(e)}")
    
    def _add_bcra_rates(self):
        """Obtiene tasas del BCRA"""
        try:
            print("🏛️ Obteniendo tasas BCRA...")
            
            fecha_min = self.df['Fecha'].min().date().isoformat()
            fecha_max = self.df['Fecha'].max().date().isoformat()
            
            base_url = "https://api.bcra.gob.ar/estadisticas/v3.0/Monetarias"
            
            # Obtener variables disponibles
            response = requests.get(base_url, timeout=30, verify=False)
            if response.status_code != 200:
                raise Exception(f"Error API BCRA: {response.status_code}")
                
            variables = response.json().get('results', [])
            
            # Buscar plazo fijo 30 días
            rate_vars = [v for v in variables if v.get('descripcion') and
                        ('plazo' in v['descripcion'].lower()) and
                        ('30 d' in v['descripcion'].lower())]
            
            if not rate_vars:
                rate_vars = [v for v in variables if 'badlar' in v.get('descripcion', '').lower()]
                
            if rate_vars:
                var_id = rate_vars[0]['idVariable']
                
                # Obtener datos de la variable
                params = {
                    'desde': f"{fecha_min}T00:00:00",
                    'hasta': f"{fecha_max}T23:59:59",
                    'limit': 3000
                }
                
                data_response = requests.get(f"{base_url}/{var_id}", 
                                           params=params, timeout=30, verify=False)
                
                if data_response.status_code == 200:
                    rate_data = data_response.json().get('results', [])
                    
                    rates_df = pd.DataFrame([
                        {
                            'Fecha': pd.to_datetime(item['fecha']).normalize(),
                            'TNA_PlazoFijo': float(item['valor'])
                        }
                        for item in rate_data if item.get('valor') is not None
                    ])
                    
                    # Merge y calcular tasa diaria
                    self.df = self.df.merge(rates_df, on='Fecha', how='left')
                    self.df['TNA_PlazoFijo'] = self.df['TNA_PlazoFijo'].ffill().bfill()
                    self.df['Tasa_PlazoFijo_Diaria'] = self.df['TNA_PlazoFijo'] / 36500.0
                    
                    print(f"✅ Tasas agregadas: {len(rates_df)} registros")
                    
        except Exception as e:
            print(f"❌ Error obteniendo tasas BCRA: {str(e)}")
    
    def convert_cedears_to_ars(self, cedear_assets, real_positions=None):
        """Convierte CEDEARs de USD a ARS usando el CCL"""
        
        if 'Tipo_Cambio_ARSUSD' not in self.df.columns:
            print("❌ No hay datos de CCL para conversión")
            return
            
        print(f"🔄 Convirtiendo {len(cedear_assets)} CEDEARs a ARS...")
        
        for asset in cedear_assets:
            usd_col = f'{asset}_USD_Original'
            if usd_col in self.df.columns:
                
                # Conversión especial para IBM si hay datos reales
                if asset == 'IBM' and real_positions and asset in real_positions:
                    real_position_ars = real_positions[asset]['valor_ars']
                    unit_price_usd = self.df[usd_col].dropna().iloc[0]
                    unit_price_ars = unit_price_usd * self.df['Tipo_Cambio_ARSUSD'].dropna().iloc[0]
                    
                    scale_factor = real_position_ars / unit_price_ars
                    self.df[asset] = (self.df[usd_col] * self.df['Tipo_Cambio_ARSUSD']) * scale_factor
                    
                    print(f"   🎯 {asset}: Escalado a posición real ${real_position_ars:,.0f} ARS")
                else:
                    # Conversión estándar
                    self.df[asset] = self.df[usd_col] * self.df['Tipo_Cambio_ARSUSD']
                
                # Calcular retornos
                self.df[f'Retorno_{asset}'] = self.df[asset].pct_change()
                
                # Stats
                avg_rate = self.df['Tipo_Cambio_ARSUSD'].mean()
                final_usd = self.df[usd_col].dropna().iloc[-1]
                final_ars = self.df[asset].dropna().iloc[-1]
                
                print(f"   ✅ {asset}: ${final_usd:.2f} USD → ${final_ars:,.0f} ARS (TC: {avg_rate:.0f})")
    
    def calculate_asset_returns(self, asset_list):
        """Calcula retornos para activos locales"""
        for asset in asset_list:
            ret_col = f'Retorno_{asset}'
            if asset in self.df.columns and ret_col not in self.df.columns:
                self.df[ret_col] = self.df[asset].pct_change()
    
    def calculate_portfolio_total(self, asset_list, portfolio_weights=None, real_positions=None):
        """Calcula el valor total de la cartera para cada fecha usando posiciones reales"""
        
        if real_positions:
            # NUEVO: Usar valores reales de posición en lugar de weights
            print(f"💼 Calculando total con POSICIONES REALES: {len(real_positions)} activos")
            
            self.df['TOTAL_CARTERA'] = 0
            
            print("📊 ESCALANDO POR POSICIONES REALES:")
            
            for asset, position_data in real_positions.items():
                if asset in self.df.columns:
                    # Obtener precio unitario del primer día
                    asset_prices = self.df[asset].dropna()
                    if len(asset_prices) > 0:
                        precio_unitario_inicial = asset_prices.iloc[0]
                        valor_posicion_real = position_data['valor_ars_actual']
                        
                        # Calcular cuántas "unidades" tienes
                        cantidad_unidades = valor_posicion_real / precio_unitario_inicial
                        
                        # Multiplicar toda la serie de precios por la cantidad
                        valor_posicion_serie = self.df[asset].fillna(method='ffill') * cantidad_unidades
                        
                        self.df['TOTAL_CARTERA'] += valor_posicion_serie
                        
                        print(f"   📈 {asset}: Precio inicial ${precio_unitario_inicial:,.0f} × {cantidad_unidades:.2f} unidades = ${valor_posicion_real:,.0f} ARS")
            
        elif portfolio_weights:
            # Fallback al método anterior con weights
            print(f"💼 Calculando total con pesos: {len(portfolio_weights)} activos")
            
            self.df['TOTAL_CARTERA'] = 0
            
            for asset, weight in portfolio_weights.items():
                if asset in self.df.columns:
                    asset_data = self.df[asset].dropna()
                    if len(asset_data) > 0:
                        base_value = asset_data.iloc[0] * weight
                        self.df['TOTAL_CARTERA'] += (self.df[asset].fillna(method='ffill') / asset_data.iloc[0]) * base_value
        
        else:
            # Usar equal weight si no hay pesos específicos
            print(f"📊 Calculando total con equal weight: {len(asset_list)} activos")
            
            portfolio_value = pd.Series(0, index=self.df.index)
            weight_per_asset = 1.0 / len(asset_list)
            
            for asset in asset_list:
                if asset in self.df.columns:
                    asset_data = self.df[asset].dropna()
                    if len(asset_data) > 0:
                        normalized = self.df[asset].fillna(method='ffill') / asset_data.iloc[0]
                        portfolio_value += normalized * weight_per_asset * 100000  # Base 100k por activo
            
            self.df['TOTAL_CARTERA'] = portfolio_value
        
        # Calcular retorno del total
        self.df['Retorno_TOTAL_CARTERA'] = self.df['TOTAL_CARTERA'].pct_change()
        
        # Stats
        total_inicial = self.df['TOTAL_CARTERA'].dropna().iloc[0]
        total_final = self.df['TOTAL_CARTERA'].dropna().iloc[-1]
        retorno_total = (total_final / total_inicial - 1) * 100
        
        print(f"✅ Total cartera calculado:")
        print(f"   💰 Valor inicial: ${total_inicial:,.0f} ARS")
        print(f"   💰 Valor final: ${total_final:,.0f} ARS")
        print(f"   📈 Retorno total: {retorno_total:.1f}%")
    
    def save_updated_data(self, output_filename="Datos_historicos_de_la_cartera_ACTUALIZADO.xlsx"):
        """Guarda el dataset actualizado con carpetas organizadas"""
        
        # Crear carpetas
        base_path = Path.cwd()
        datos_path = base_path / "outputs" / "datos"
        datos_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Guardar Excel
            excel_path = datos_path / output_filename
            self.df.to_excel(excel_path, index=False)
            print(f"✅ Excel guardado: {excel_path}")
            
            # Guardar CSV
            csv_filename = output_filename.replace('.xlsx', '.csv')
            csv_path = datos_path / csv_filename
            self.df.to_csv(csv_path, index=False)
            print(f"✅ CSV guardado: {csv_path}")
            
            # Guardar resumen
            summary_data = {
                'total_registros': len(self.df),
                'periodo_desde': str(self.df['Fecha'].min()),
                'periodo_hasta': str(self.df['Fecha'].max()),
                'activos': len([col for col in self.df.columns if col not in ['Fecha'] and not col.startswith('Retorno_') and not col.startswith('Tipo_Cambio') and not col.startswith('Tasa_')]),
                'tiene_total_cartera': 'TOTAL_CARTERA' in self.df.columns
            }
            
            summary_path = datos_path / "resumen_dataset.csv"
            pd.DataFrame([summary_data]).to_csv(summary_path, index=False)
            print(f"✅ Resumen guardado: {summary_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
            return False
    
    @staticmethod
    def create_output_folders():
        """Crea estructura de carpetas para outputs"""
        base_path = Path.cwd()
        
        folders = [
            "outputs/datos",
            "outputs/graficos", 
            "outputs/reportes",
            "outputs/simulaciones"
        ]
        
        for folder in folders:
            folder_path = base_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
        
        print("📁 Carpetas de output creadas:")
        for folder in folders:
            print(f"   ✅ {folder}")
        
        return {
            'datos': base_path / "outputs/datos",
            'graficos': base_path / "outputs/graficos", 
            'reportes': base_path / "outputs/reportes",
            'simulaciones': base_path / "outputs/simulaciones"
        }
    
    def _find_file(self, filename):
        """Busca archivo en ubicaciones posibles"""
        possible_paths = [
            self.base_path / filename,
            self.base_path / "data" / filename,
            self.base_path / "datos" / filename
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Buscar archivos Excel si no encuentra el específico
        excel_files = list(self.base_path.glob("*.xlsx"))
        if excel_files:
            return excel_files[0]
            
        return None
    
    def get_final_dataset(self):
        """Retorna el dataset final preparado"""
        return self.df
    
    def get_asset_list(self):
        """Retorna lista de activos válidos en el dataset"""
        assets = []
        for col in self.df.columns:
            if (col not in ['Fecha', 'Tipo_Cambio_ARSUSD', 'TNA_PlazoFijo', 
                           'Tasa_PlazoFijo_Diaria', 'Retorno_FX', 'TOTAL_CARTERA'] and
                not col.startswith('Retorno_') and 
                not col.startswith('Peso_') and
                not col.endswith('_USD_Original')):
                assets.append(col)
        return assets
    
    def get_macro_variables(self):
        """Retorna lista de variables macroeconómicas"""
        macro_vars = []
        for col in ['Tipo_Cambio_ARSUSD', 'Retorno_FX', 'TNA_PlazoFijo', 'Tasa_PlazoFijo_Diaria']:
            if col in self.df.columns:
                macro_vars.append(col)
        return macro_vars
    
    def get_return_columns(self):
        """Retorna lista de columnas de retornos"""
        return [col for col in self.df.columns if col.startswith('Retorno_') and col != 'Retorno_FX']

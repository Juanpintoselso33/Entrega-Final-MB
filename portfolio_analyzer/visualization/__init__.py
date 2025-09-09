"""
Visualization Module - Gráficos de Composición de Cartera Real
==============================================================

Módulo ENFOCADO en visualizar TU cartera real:
- Gráficos de evolución de composición (pesos) en el tiempo
- Visualización de cómo cambian los pesos de cada activo
- Gráficos de área apilada de la composición
- Evolución del valor total de la cartera
- Métricas de concentración temporal

OBJETIVO: Ver gráficamente cómo evoluciona la composición de TU cartera.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PortfolioVisualizer:
    def _save_plot(self, fig, save_path):
        """Guarda la figura plotly en la ruta especificada (como PNG y HTML)."""
        path = self._get_save_path(save_path)
        if path is not None:
            # Guardar como PNG
            try:
                fig.write_image(str(path), width=800, height=500, scale=2)
                print(f"✅ Gráfico guardado como imagen en: {path}")
            except Exception as e:
                print(f"⚠️ No se pudo guardar como imagen PNG: {e}")
            # Guardar como HTML interactivo
            html_path = str(path.with_suffix('.html'))
            try:
                fig.write_html(html_path)
                print(f"✅ Gráfico guardado como HTML interactivo en: {html_path}")
            except Exception as e:
                print(f"⚠️ No se pudo guardar como HTML: {e}")
    """Generador de visualizaciones para análisis de carteras"""
    
    def __init__(self, theme='plotly_white'):
        self.theme = theme
        self.color_palette = [
            '#e74c3c', '#2ecc71', '#3498db', '#f39c12', 
            '#9b59b6', '#1abc9c', '#34495e', '#e67e22'
        ]
    
    def _get_save_path(self, save_path):
        """Prepara la ruta de guardado usando carpeta outputs/graficos"""
        if save_path:
            # SIEMPRE guardar en outputs/graficos
            graficos_path = Path.cwd() / "outputs" / "graficos"
            graficos_path.mkdir(parents=True, exist_ok=True)
            
            # Extraer solo el nombre del archivo si viene con path
            file_name = Path(save_path).name
            # Añadir extensión .png si no tiene
            if not Path(file_name).suffix:
                file_name = file_name + ".png"
            return graficos_path / file_name
        return None
        
    def plot_normalized_performance(self, df, assets, title="Evolución Normalizada de Activos", save_path=None):
        """
        Grafica la evolución normalizada (base 100) de una lista de activos.
        
        Args:
            df (pd.DataFrame): DataFrame con precios o valores, indexado por fecha.
            assets (list): Lista de columnas (activos) a graficar.
            title (str): Título del gráfico.
            save_path (str, optional): Ruta para guardar el gráfico.
        """
        if not all(asset in df.columns for asset in assets):
            print(f"⚠️ Algunos activos no se encontraron en el DataFrame. Activos pedidos: {assets}, disponibles: {list(df.columns)}")
            return None

        # Normalizar los datos a base 100
        normalized_df = (df[assets] / df[assets].iloc[0]) * 100
        
        fig = px.line(
            normalized_df, 
            x=normalized_df.index, 
            y=assets,
            title=title,
            labels={'value': 'Performance (Base 100)', 'variable': 'Activo', 'index': 'Fecha'},
            template='plotly_white'
        )
        
        fig.update_layout(
            legend_title_text='Activos',
            hovermode='x unified'
        )
        
        if save_path:
            self._save_plot(fig, save_path)
            
        fig.show()
        return fig
    
    def plot_correlation_matrix(self, returns_data, save_path=None):
        """Crea matriz de correlación interactiva"""
        
        print("📊 Creando matriz de correlación...")
        
        # Calcular correlaciones
        corr_matrix = returns_data.corr()
        
        # Crear heatmap
        fig = go.Figure(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Matriz de Correlación de Retornos',
                'x': 0.5,
                'font': {'size': 16}
            },
            template=self.theme,
            height=500,
            width=600
        )
        
        # Mostrar
        fig.show()
        
        # Guardar
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=800, height=500, scale=2)
                print(f"✅ Matriz guardada en: {save_file_path}")
            except Exception as e:
                print(f"⚠️ No se pudo guardar: {str(e)}")
        
        # Estadísticas de correlación
        mean_corr = corr_matrix.mean().mean()
        print(f"📊 Correlación promedio: {mean_corr:.3f}")
        
        return fig
    
    def plot_backtesting_comparison(self, backtest_results, save_path=None):
        """Crea gráfico comparativo de backtesting"""
        
        print("📊 Creando gráfico comparativo de backtesting...")
        
        if not backtest_results:
            print("❌ No hay resultados de backtesting")
            return None
            
        fig = go.Figure()
        
        # Mapeo de colores por tipo de estrategia
        color_mapping = {
            'Tu Cartera Real': '#e74c3c',
            'Cartera Óptima Sharpe': '#2ecc71', 
            'Cartera Óptima MinVol': '#3498db',
            'Equal Weight': '#f39c12',
            'Argentina 60/40': '#9b59b6',
            'Cap Weighted': '#1abc9c'
        }
        
        for i, result in enumerate(backtest_results):
            # Determinar color y estilo de línea
            color = color_mapping.get(result['nombre'], self.color_palette[i % len(self.color_palette)])
            
            if 'Tu Cartera' in result['nombre']:
                width = 4
                dash = 'solid'
            elif 'Benchmark' in result['nombre'] or 'Equal Weight' in result['nombre']:
                width = 2
                dash = 'dash'
            else:
                width = 3
                dash = 'solid'
            
            fig.add_trace(go.Scatter(
                x=result['fechas'],
                y=result['valores'],
                mode='lines',
                name=result['nombre'],
                line=dict(width=width, color=color, dash=dash),
                hovertemplate=(
                    f"<b>{result['nombre']}</b><br>"
                    f"Fecha: %{{x}}<br>"
                    f"Valor: %{{y:.1f}}<br>"
                    f"Sharpe: {result['sharpe_ratio']:.3f}<br>"
                    f"<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title={
                'text': '🏆 Comparación de Estrategias - Backtesting',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Período',
            yaxis_title='Valor de Cartera (Base 100)',
            template=self.theme,
            height=700,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )
        
        # Línea de referencia
        fig.add_hline(y=100, line_dash="dot", line_color="gray",
                      annotation_text="Valor Inicial", annotation_position="bottom right")
        
        # Mostrar
        fig.show()
        
        # Guardar
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=1400, height=700, scale=2)
                print(f"✅ Gráfico comparativo guardado en: {save_file_path}")
            except Exception as e:
                print(f"⚠️ No se pudo guardar: {str(e)}")
                
        return fig
    
    def plot_risk_return_scatter(self, strategies_data, save_path=None):
        """Crea gráfico de dispersión riesgo-retorno"""
        
        print("📊 Creando gráfico riesgo-retorno...")
        
        if not strategies_data:
            return None
            
        # Extraer datos
        names = []
        returns = []
        risks = []
        sharpes = []
        
        for strategy in strategies_data:
            names.append(strategy['nombre'])
            # Compatibilidad: aceptar 'rendimiento' o 'rendimiento_anual'
            if 'rendimiento_anual' in strategy:
                ret = strategy['rendimiento_anual']
            else:
                ret = strategy.get('rendimiento', 0)
            if 'volatilidad_anual' in strategy:
                risk = strategy['volatilidad_anual']
            else:
                risk = strategy.get('riesgo', 0)
            returns.append(ret * 100)
            risks.append(risk * 100)
            sharpes.append(strategy['sharpe_ratio'])
        
        # Crear scatter plot
        fig = go.Figure()
        
        for i, (name, ret, risk, sharpe) in enumerate(zip(names, returns, risks, sharpes)):
            # Determinar color y tamaño
            if 'Tu Cartera' in name:
                color = '#e74c3c'
                size = 15
                symbol = 'star'
            elif 'Óptima' in name:
                color = '#2ecc71'
                size = 12
                symbol = 'diamond'
            else:
                color = '#3498db'
                size = 10
                symbol = 'circle'
            
            fig.add_trace(go.Scatter(
                x=[risk],
                y=[ret],
                mode='markers+text',
                name=name,
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                text=[name],
                textposition="top center",
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Retorno: {ret:.2f}%<br>"
                    f"Riesgo: {risk:.2f}%<br>"
                    f"Sharpe: {sharpe:.3f}<br>"
                    f"<extra></extra>"
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title={
                'text': 'Frontera Eficiente - Riesgo vs Retorno',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Riesgo (Volatilidad Anual %)',
            yaxis_title='Retorno Esperado Anual (%)',
            template=self.theme,
            height=600,
            width=800
        )
        
        # Mostrar
        fig.show()
        
        # Guardar
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=800, height=600, scale=2)
                print(f"✅ Gráfico riesgo-retorno guardado en: {save_file_path}")
            except:
                print("⚠️ No se pudo guardar el gráfico")
                
        return fig
    
    def plot_drawdown_analysis(self, backtest_results, save_path=None):
        """Analiza y visualiza drawdowns"""
        
        print("📊 Creando análisis de drawdown...")
        
        if not backtest_results:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Evolución del Valor de Cartera', 'Drawdown %'),
            vertical_spacing=0.1
        )
        
        for i, result in enumerate(backtest_results):
            color = self.color_palette[i % len(self.color_palette)]
            
            # Gráfico de valor
            fig.add_trace(
                go.Scatter(
                    x=result['fechas'],
                    y=result['valores'],
                    mode='lines',
                    name=result['nombre'],
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Calcular y graficar drawdown
            values = np.array(result['valores'])
            running_max = np.maximum.accumulate(values)
            drawdown_pct = ((values - running_max) / running_max) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=result['fechas'],
                    y=drawdown_pct,
                    mode='lines',
                    name=f"{result['nombre']} DD",
                    line=dict(color=color),
                    fill='tonexty' if i == 0 else None,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Análisis de Drawdown por Estrategia',
            template=self.theme,
            height=700
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Valor Base 100", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        
        # Mostrar
        fig.show()
        
        # Guardar
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=1200, height=700, scale=2)
                print(f"✅ Análisis de drawdown guardado en: {save_file_path}")
            except:
                print("⚠️ No se pudo guardar el análisis")
                
        return fig
    
    def create_portfolio_composition_chart(self, weights_dict, title="Composición de Cartera"):
        """Crea gráfico de composición de cartera (pie chart)"""
        
        # Filtrar pesos significativos (>1%)
        significant_weights = {k: v for k, v in weights_dict.items() if v > 0.01}
        
        if not significant_weights:
            print("❌ No hay pesos significativos para graficar")
            return None
            
        # Preparar datos
        assets = list(significant_weights.keys())
        weights = list(significant_weights.values())
        percentages = [w * 100 for w in weights]
        
        # Crear pie chart
        fig = go.Figure(
            go.Pie(
                labels=assets,
                values=percentages,
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b><br>Peso: %{percent}<br><extra></extra>'
            )
        )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'font': {'size': 16}
            },
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        fig.show()
        return fig
    
    def plot_rolling_metrics(self, returns_series, window=63, save_path=None):
        """Gráfico de métricas móviles (Sharpe, volatilidad)"""
        
        print(f"📊 Creando métricas móviles ({window} días)...")
        
        if len(returns_series) < window:  # Solo requerir window observaciones
            print("❌ Pocos datos para métricas móviles")
            return None
            
        # Calcular métricas móviles
        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252) * 100
        rolling_return = returns_series.rolling(window=window).mean() * 252 * 100
        rolling_sharpe = rolling_return / rolling_vol
        
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Retorno Anualizado Móvil (%)', 'Volatilidad Móvil (%)', 'Sharpe Ratio Móvil'),
            vertical_spacing=0.08
        )
        
        # Retorno
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values,
                mode='lines',
                name='Retorno',
                line=dict(color='#2ecc71')
            ),
            row=1, col=1
        )
        
        # Volatilidad
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatilidad',
                line=dict(color='#e74c3c')
            ),
            row=2, col=1
        )
        
        # Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe',
                line=dict(color='#3498db')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'Métricas Móviles ({window} días)',
            template=self.theme,
            height=800,
            showlegend=False
        )
        
        # Mostrar
        fig.show()
        
        # Guardar
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=1000, height=800, scale=2)
                print(f"✅ Métricas móviles guardadas en: {save_file_path}")
            except:
                print("⚠️ No se pudo guardar el gráfico")
                
        return fig

    def plot_positions_value(self, weights_matrix, total_value_series, assets=None, top_n=10, stacked=False, save_path=None):
        """Grafica el valor absoluto en ARS de cada posición a lo largo del tiempo.

        Parámetros:
        - weights_matrix: DataFrame (index fechas, columnas activos, valores = pesos diarios)
        - total_value_series: Series (valor total diario de la cartera en ARS)
        - assets: lista opcional de activos a incluir. Si None se toman los top_n por peso promedio
        - top_n: número de activos a seleccionar si no se pasa 'assets'
        - stacked: si True crea área apilada, si False líneas independientes
        - save_path: nombre base para guardar imagen en outputs/graficos

        Retorna: figura Plotly o None
        """
        print("📊 Creando gráfico de valor absoluto de posiciones (ARS)...")

        if weights_matrix is None or total_value_series is None:
            print("❌ Datos insuficientes (weights_matrix o total_value_series faltante)")
            return None

        # Alinear índices
        common_index = weights_matrix.index.intersection(total_value_series.index)
        if common_index.empty:
            print("❌ No hay fechas comunes entre pesos y valor total")
            return None

        wm = weights_matrix.loc[common_index]
        tv = total_value_series.loc[common_index]

        # Selección de activos
        if assets is None:
            avg_weights = wm.mean().sort_values(ascending=False)
            assets = avg_weights.head(top_n).index.tolist()
        else:
            assets = [a for a in assets if a in wm.columns]

        if not assets:
            print("❌ Lista de activos vacía para graficar")
            return None

        # Calcular valores absolutos en ARS
        values_df = wm[assets].multiply(tv, axis=0)

        fig = go.Figure()

        if stacked:
            cumulative = None
            for i, asset in enumerate(assets):
                y_values = values_df[asset].fillna(0)
                color = self.color_palette[i % len(self.color_palette)]
                fig.add_trace(go.Scatter(
                    x=values_df.index,
                    y=y_values,
                    mode='lines',
                    name=asset,
                    line=dict(width=1.5, color=color),
                    stackgroup='one',
                    hovertemplate=f"<b>{asset}</b><br>Fecha: %{{x}}<br>Valor: $%{{y:,.0f}} ARS<extra></extra>"
                ))
        else:
            for i, asset in enumerate(assets):
                color = self.color_palette[i % len(self.color_palette)]
                fig.add_trace(go.Scatter(
                    x=values_df.index,
                    y=values_df[asset].values,
                    mode='lines',
                    name=asset,
                    line=dict(width=2, color=color),
                    hovertemplate=f"<b>{asset}</b><br>Fecha: %{{x}}<br>Valor: $%{{y:,.0f}} ARS<extra></extra>"
                ))

        fig.update_layout(
            title={
                'text': 'Valor de Posiciones en ARS',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Fecha',
            yaxis_title='Valor (ARS)',
            template=self.theme,
            height=600,
            hovermode='x unified'
        )

        # Mostrar
        fig.show()

        # Guardar si corresponde
        if save_path:
            save_file_path = self._get_save_path(save_path)
            try:
                fig.write_image(save_file_path, width=1200, height=600, scale=2)
                print(f"✅ Gráfico guardado en: {save_file_path}")
            except Exception as e:
                print(f"⚠️ No se pudo guardar: {e}")

        # Estadísticas rápidas
        latest_values = values_df.iloc[-1].sort_values(ascending=False)
        top_total = latest_values.head(3).sum()
        print(f"📌 Top 3 posiciones actuales suman: ${top_total:,.0f} ARS")

        return fig

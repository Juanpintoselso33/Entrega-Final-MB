# Portfolio Composition Analysis System

## Overview
This system analyzes **real portfolio composition over time** using daily portfolio data. The primary focus is tracking how portfolio weights change day-by-day, not theoretical optimization.

## Key Architecture

### Core Data Flow
1. **Primary Data Source**: `outputs/portfolio_consolidado_completo.csv` - Daily portfolio holdings with columns:
   - `instrumento`: Asset name (full Spanish names like "BANCO HIPOTECARIO ESCRIT...")
   - `cantidad`: Quantity held
   - `precio`: Price per unit in ARS
   - `total`: Total value in ARS
   - `fecha`: Date

2. **Secondary Data**: `movements_report_2025-01-01_2025-09-08.csv` - Transaction history (optional)
3. **Auxiliary Data**: `precios_completos_ars.csv` - Benchmark prices (only if needed)

### Main Classes

#### `PortfolioCompositionLoader` (Primary)
- **Core method**: `load_portfolio_daily_data()` - loads the main CSV
- **Key calculation**: `_calculate_daily_composition()` - converts daily holdings to portfolio weights
- **Output**: `portfolio_weights_history` - DataFrame with dates as index, instruments as columns, weights as values

#### `DataLoader` (Wrapper)
- Delegates to `PortfolioCompositionLoader`
- Maintains backward compatibility
- **Main workflow**: `load_main_portfolio_data()` → loads daily data → calculates composition

## Critical Patterns

### Spanish Asset Names
The system handles full Spanish instrument names from broker data:
```python
instrument_mapping = {
    'CENTRAL PUERTO S.A. ORD. 1 VOTO ESCRIT (CEPU)': 'CEPU',
    'BANCO HIPOTECARIO ESCRIT.  D  (CATEGORIAS 1 Y 2) 3 VOTOS (BHIP)': 'BHIL',
    'FCI ALAMERICA RENTA MIXTA CL.A ESC (COCORMA)': 'COCORMA'
}
```

### Semicolon-Separated CSV Files
All CSV files use `;` as separator, not commas. Always use `pd.read_csv(file, sep=';')`.

### Date Handling
- Portfolio data: `YYYY-MM-DD` format, use `pd.to_datetime()`
- Movements data: `DD-MM-YYYY` format, use `pd.to_datetime(format='%d-%m-%Y')`

### Output Structure
```
outputs/
├── portfolio_consolidado_completo.csv  # PRIMARY - daily portfolio data
├── precios_completos_ars.csv          # AUXILIARY - benchmark prices
├── graficos/                           # Generated charts
├── reportes/                           # Analysis reports
└── datos/                              # Processed datasets
```

## Development Workflow

### Essential Commands
1. **Load and analyze portfolio**:
   ```python
   from portfolio_analyzer import PortfolioCompositionLoader
   loader = PortfolioCompositionLoader()
   loader.load_portfolio_daily_data()  # Loads main CSV
   weights_matrix = loader.get_portfolio_weights_matrix()  # Gets daily weights
   ```

2. **Get composition for analysis**:
   ```python
   evolution = loader.get_portfolio_evolution()  # Full composition with totals
   current_weights = loader.get_current_composition()  # Latest weights
   total_value = loader.get_total_value_series()  # Daily portfolio value
   ```

### Module Focus Areas
- **data_loader**: Load real portfolio data, calculate daily composition weights
- **risk_analysis**: Apply metrics to actual portfolio composition (not theoretical)
- **visualization**: Chart composition evolution over time (area charts, weight changes)
- **optimization**: Analyze efficiency of current real portfolio vs alternatives
- **simulation**: Project future performance based on current real composition

## Project Conventions

### Data Priority Hierarchy
1. **PRIMARY**: Real daily portfolio composition (`portfolio_consolidado_completo.csv`)
2. **Secondary**: Transaction history for position reconciliation  
3. **Auxiliary**: Benchmark data for comparison (only if specifically needed)

### Error Handling Pattern
Functions return boolean success/failure and print detailed status with emojis:
```python
print(f"✅ CARTERA DIARIA CARGADA:")
print(f"❌ No se encontró {filename}")
print(f"⚠️ No se encontró {filename} (opcional)")
```

### Naming Conventions
- Spanish field names from broker data are preserved
- Internal variables use English (`portfolio_weights_history`)
- File outputs use Spanish descriptors (`portfolio_consolidado_completo`)

## Key Integration Points

The system is **composition-focused**, not optimization-focused. When implementing features:
- Start with real portfolio weights, not theoretical optimal weights
- Analyze how composition changed over time (June: only COCORMA → July: added BHIP, METR, CEPU → August: added AAPL, IBM → later: added EWZ, SPY)
- Use benchmark data only for comparison, never as primary analysis target

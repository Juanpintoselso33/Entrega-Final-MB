import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("🔍 VERIFICANDO TICKERS FALTANTES:")
print("")

# Verificar BHIL
print("BHIL:")
try:
    bhil = yf.download('BHIL', period='6mo', progress=False)
    print(f"✅ Datos encontrados: {len(bhil)} registros")
    if len(bhil) > 0:
        print(bhil.head(2))
    else:
        print("Sin datos")
except Exception as e:
    print(f"❌ Error: {e}")

print("")

# Verificar METRO
print("METRO:")
try:
    metro = yf.download('METRO', period='6mo', progress=False)
    print(f"✅ Datos encontrados: {len(metro)} registros")
    if len(metro) > 0:
        print(metro.head(2))
    else:
        print("Sin datos")
except Exception as e:
    print(f"❌ Error: {e}")

print("")

# Probar variaciones
print("🔄 PROBANDO VARIACIONES:")
variaciones = ['BHIL.BA', 'BHIL.NYSE', 'METRO.BA', 'METRO.NYSE', 'METR', 'BHIP']

for ticker in variaciones:
    try:
        data = yf.download(ticker, period='1mo', progress=False)
        if len(data) > 0:
            print(f"✅ {ticker}: {len(data)} registros encontrados")
        else:
            print(f"❌ {ticker}: Sin datos")
    except:
        print(f"❌ {ticker}: Error de conexión")

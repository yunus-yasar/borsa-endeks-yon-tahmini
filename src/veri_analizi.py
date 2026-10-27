
import pandas as pd
import os

dosyalar = {
    "BIST 100": "data/raw/BIST100.csv",
    "S&P 500": "data/raw/SP500.csv",
    "NASDAQ-100": "data/raw/NASDAQ100.csv"
}

os.makedirs("data/processed", exist_ok=True)

for endeks, dosya in dosyalar.items():

    df = pd.read_csv(dosya)

    print("\nEndeks:", endeks)
    print("Veri seti boyutu:", df.shape)

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nSütunlar:")
    print(df.columns.tolist())

    print("\nEksik değerler:")
    print(df.isnull().sum().to_string())

    print("\nTekrar eden satır sayısı:")
    print(df.duplicated().sum())

    print("\nTekrar eden tarih sayısı:")
    print(df["Date"].duplicated().sum())

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    print("\nİlk tarih:", df["Date"].min())
    print("Son tarih:", df["Date"].max())

    fiyat_sutunlari = ["Price", "Open", "High", "Low"]

    for sutun in fiyat_sutunlari:
        df[sutun] = df[sutun].astype(str).str.replace(",", "", regex=False)
        df[sutun] = pd.to_numeric(df[sutun], errors="coerce")

    print("\nFiyat sütunlarının veri tipleri:")
    print(df[fiyat_sutunlari].dtypes)

    print("\nFiyat sütunlarındaki eksik değerler:")
    print(df[fiyat_sutunlari].isnull().sum().to_string())

    print("\nSıfır veya negatif fiyat sayıları:")
    print((df[fiyat_sutunlari] <= 0).sum().to_string())

    print("\nEn düşük fiyat değerleri:")
    print(df[fiyat_sutunlari].min())

    df = df.drop(columns=["Vol.", "Change %"])

    print("\nTemizlenmiş veri setinin sütunları:")
    print(df.columns.tolist())

    print("\nTemizlenmiş veri seti:")
    print(df.head())

    df["Change"] = df["Price"].pct_change() * 100

    print("\nGünlük değişim hesaplandı:")
    print(df[["Date", "Price", "Change"]].head())

    df = df.dropna(subset=["Change"])

    df = df.reset_index(drop=True)

    df["MA5"] = df["Price"].rolling(window=5).mean()
    df["MA10"] = df["Price"].rolling(window=10).mean()
    df["MA20"] = df["Price"].rolling(window=20).mean()

    print("\nHareketli ortalamalar hesaplandı:")
    print(df[["Date", "Price", "MA5", "MA10", "MA20"]].tail())

    delta = df["Price"].diff()

    kazanc = delta.clip(lower=0)
    kayip = -delta.clip(upper=0)

    ortalama_kazanc = kazanc.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    ortalama_kayip = kayip.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = ortalama_kazanc / ortalama_kayip

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[
        (ortalama_kayip == 0) & (ortalama_kazanc > 0), "RSI"
    ] = 100

    df.loc[
        (ortalama_kayip == 0) & (ortalama_kazanc == 0), "RSI"
    ] = 50

    print("\nRSI hesaplandı:")
    print(df[["Date", "Price", "RSI"]].tail())

    ema12 = df["Price"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["Price"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    print("\nMACD hesaplandı:")
    print(
        df[
            ["Date", "Price", "MACD", "MACD_Signal", "MACD_Hist"]
        ].tail()
    )

    print("\nTeknik göstergelerdeki eksik değerler:")
    print(
        df[
            ["MA5", "MA10", "MA20", "RSI",
             "MACD", "MACD_Signal", "MACD_Hist"]
        ].isnull().sum().to_string()
    )

    dosya_adi = os.path.basename(dosya)

    df.to_csv("data/processed/" + dosya_adi, index=False)

    print("\nİşlenmiş veri kaydedildi:", dosya_adi)
    print("Yeni veri seti boyutu:", df.shape)

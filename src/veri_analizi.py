
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
    print("Ham veri seti boyutu:", df.shape)

    print("\nHam verideki eksik değerler:")
    print(df.isnull().sum().to_string())

    print("\nTekrar eden satır sayısı:", df.duplicated().sum())
    print("Tekrar eden tarih sayısı:", df["Date"].duplicated().sum())

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    print("\nİlk tarih:", df["Date"].min())
    print("Son tarih:", df["Date"].max())

    fiyat_sutunlari = ["Price", "Open", "High", "Low"]

    for sutun in fiyat_sutunlari:
        df[sutun] = df[sutun].astype(str).str.replace(",", "", regex=False)
        df[sutun] = pd.to_numeric(df[sutun], errors="coerce")

    print("\nSıfır veya negatif fiyat sayısı:",
          (df[fiyat_sutunlari] <= 0).sum().sum())

    hatali_fiyat = (
        (df["High"] < df["Low"]) |
        (df["Open"] > df["High"]) |
        (df["Open"] < df["Low"]) |
        (df["Price"] > df["High"]) |
        (df["Price"] < df["Low"])
    )

    print("\nTemizlenen hatalı fiyat kaydı:", hatali_fiyat.sum())

    df = df.loc[~hatali_fiyat].copy()

    df = df.reset_index(drop=True)

    df = df.drop(columns=["Vol.", "Change %"])

    df["Change"] = df["Price"].pct_change() * 100

    df = df.dropna(subset=["Change"])

    df = df.reset_index(drop=True)

    df["MA5"] = df["Price"].rolling(window=5).mean()
    df["MA10"] = df["Price"].rolling(window=10).mean()
    df["MA20"] = df["Price"].rolling(window=20).mean()

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

    print("\nTeknik göstergelerdeki eksik değerler:")
    print(
        df[
            ["MA5", "MA10", "MA20", "RSI",
             "MACD", "MACD_Signal", "MACD_Hist"]
        ].isnull().sum().to_string()
    )

    df["Target"] = (df["Price"].shift(-1) > df["Price"]).astype(int)

    df = df.iloc[:-1].copy()

    print("\nEksik değer temizliği öncesi satır sayısı:", len(df))

    df = df.dropna()

    df = df.reset_index(drop=True)

    print("Eksik değer temizliği sonrası satır sayısı:", len(df))

    print("\nSon veri doğrulama:")

    print("Tarihler sıralı mı:", df["Date"].is_monotonic_increasing)

    print("Tekrar eden tarih sayısı:", df["Date"].duplicated().sum())

    hatali_fiyat = (
        (df["High"] < df["Low"]) |
        (df["Open"] > df["High"]) |
        (df["Open"] < df["Low"]) |
        (df["Price"] > df["High"]) |
        (df["Price"] < df["Low"])
    )

    print("Hatalı fiyat aralığı sayısı:", hatali_fiyat.sum())

    print(
        "RSI aralık hatası:",
        ((df["RSI"] < 0) | (df["RSI"] > 100)).sum()
    )

    beklenen_target = (
        df["Price"].shift(-1) > df["Price"]
    ).astype(int)

    target_hatalari = (
        df["Target"].iloc[:-1] != beklenen_target.iloc[:-1]
    ).sum()

    print("Hatalı Target sayısı:", target_hatalari)

    print("Toplam eksik değer sayısı:", df.isnull().sum().sum())

    print("\nTarget sınıf dağılımı:")
    print(df["Target"].value_counts().sort_index())

    print("\nTarget yüzdelik dağılımı (%):")
    print(
        (df["Target"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    print("\nSon veri setinin ilk 5 satırı:")
    print(df.head().to_string(index=False))

    dosya_adi = os.path.basename(dosya)

    df.to_csv("data/processed/" + dosya_adi, index=False)

    print("\nİşlenmiş veri kaydedildi:", dosya_adi)
    print("Yeni veri seti boyutu:", df.shape)

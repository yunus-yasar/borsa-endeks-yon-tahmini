
import pandas as pd

dosyalar = {
    "BIST 100": "BIST100.csv",
    "S&P 500": "SP500.csv",
    "NASDAQ-100": "NASDAQ100.csv"
}

for endeks, dosya in dosyalar.items():

    ham_df = pd.read_csv("data/raw/" + dosya)
    df = pd.read_csv("data/processed/" + dosya)

    ham_df["Date"] = pd.to_datetime(ham_df["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    fiyat_sutunlari = ["Price", "Open", "High", "Low"]

    for sutun in fiyat_sutunlari:
        ham_df[sutun] = pd.to_numeric(
            ham_df[sutun].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    def hatali_fiyat_sayisi(veri):
        hatali = (
            (veri["High"] < veri["Low"]) |
            (veri["Open"] > veri["High"]) |
            (veri["Open"] < veri["Low"]) |
            (veri["Price"] > veri["High"]) |
            (veri["Price"] < veri["Low"])
        )
        return hatali.sum()

    print("\nEndeks:", endeks)

    print("Ham veri sayısı:", len(ham_df))
    print("İşlenmiş veri sayısı:", len(df))
    print("Ham sütun sayısı:", len(ham_df.columns))
    print("İşlenmiş sütun sayısı:", len(df.columns))

    print("İlk tarih:", df["Date"].min().date())
    print("Son tarih:", df["Date"].max().date())

    print("\nHam verideki eksik değerler:")
    print(ham_df.isnull().sum().to_string())

    print("\nİşlenmiş verideki eksik değerler:")
    print(df.isnull().sum().to_string())

    print("\nTekrarlanan satır sayısı:", df.duplicated().sum())
    print("Tekrarlanan tarih sayısı:", df["Date"].duplicated().sum())

    print("\nHatalı fiyat kayıtları:")
    print("Ön işleme öncesi:", hatali_fiyat_sayisi(ham_df))
    print("Ön işleme sonrası:", hatali_fiyat_sayisi(df))

    print("\nTarget sınıf dağılımı:")
    print(df["Target"].value_counts().sort_index().to_string())

    print("\nTarget sınıf yüzdeleri (%):")
    print(
        (df["Target"].value_counts(normalize=True).sort_index() * 100)
        .round(2).to_string()
    )

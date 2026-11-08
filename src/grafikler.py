
import os
import pandas as pd
import matplotlib.pyplot as plt

dosyalar = {
    "BIST 100": "BIST100.csv",
    "S&P 500": "SP500.csv",
    "NASDAQ-100": "NASDAQ100.csv"
}

os.makedirs("reports/figures", exist_ok=True)

eksik_veriler = {}

for endeks, dosya in dosyalar.items():

    ham_df = pd.read_csv("data/raw/" + dosya)
    df = pd.read_csv("data/processed/" + dosya)

    df["Date"] = pd.to_datetime(df["Date"])
    dosya_adi = dosya.replace(".csv", "")

    eksik_veriler[endeks] = ham_df.isnull().sum()

    siniflar = df["Target"].value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(8, 5))
    plt.bar(["0 - Yukselis yok", "1 - Yukselis var"], siniflar.values)

    for i, sayi in enumerate(siniflar.values):
        plt.text(i, sayi, str(sayi), ha="center", va="bottom")

    plt.title(endeks + " - Target Sinif Dagilimi")
    plt.ylabel("Veri Sayisi")
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_target.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(df["Date"], df["Price"], linewidth=1)
    plt.title(endeks + " - Kapanis Fiyatlari")
    plt.xlabel("Tarih")
    plt.ylabel("Kapanis Fiyati")
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_fiyat.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(df["Change"], bins=100)
    plt.title(endeks + " - Gunluk Getiri Dagilimi")
    plt.xlabel("Gunluk Getiri (%)")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_getiri.png", dpi=300)
    plt.close()

    son_veriler = df.tail(250)

    plt.figure(figsize=(12, 6))
    plt.plot(son_veriler["Date"], son_veriler["Price"], label="Kapanis")
    plt.plot(son_veriler["Date"], son_veriler["MA5"], label="MA5")
    plt.plot(son_veriler["Date"], son_veriler["MA10"], label="MA10")
    plt.plot(son_veriler["Date"], son_veriler["MA20"], label="MA20")
    plt.title(endeks + " - Hareketli Ortalamalar")
    plt.xlabel("Tarih")
    plt.ylabel("Endeks Degeri")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_ma.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(son_veriler["Date"], son_veriler["RSI"], label="RSI")
    plt.axhline(70, linestyle="--", label="70")
    plt.axhline(30, linestyle="--", label="30")
    plt.title(endeks + " - RSI (14)")
    plt.xlabel("Tarih")
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_rsi.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(son_veriler["Date"], son_veriler["MACD"], label="MACD")
    plt.plot(
        son_veriler["Date"],
        son_veriler["MACD_Signal"],
        label="Sinyal"
    )
    plt.bar(
        son_veriler["Date"],
        son_veriler["MACD_Hist"],
        label="Histogram"
    )
    plt.axhline(0, linewidth=0.8)
    plt.title(endeks + " - MACD (12, 26, 9)")
    plt.xlabel("Tarih")
    plt.ylabel("MACD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/" + dosya_adi + "_macd.png", dpi=300)
    plt.close()

    print(endeks, "grafikleri kaydedildi.")

eksik_df = pd.DataFrame(eksik_veriler).T

plt.figure(figsize=(10, 5))
eksik_df.plot(kind="bar", ax=plt.gca())
plt.title("Ham Veri Setlerinde Eksik Degerler")
plt.xlabel("Endeks")
plt.ylabel("Eksik Veri Sayisi")
plt.xticks(rotation=0)
plt.legend(title="Sutunlar")
plt.tight_layout()
plt.savefig("reports/figures/eksik_veriler.png", dpi=300)
plt.close()

print("\nToplam 19 grafik reports/figures klasorune kaydedildi.")

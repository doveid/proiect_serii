import pandas as pd

# === 1. Citire fișiere originale ===
df_somaj = pd.read_csv("../proiectSeriiRevizuit/exportPivot_AMG130A.csv", encoding="utf-8")
df_pib = pd.read_csv("exportPivot_CON106J.csv", encoding="utf-8")

# === 2. Curățare nume coloane ===
df_somaj.columns = df_somaj.columns.str.strip()
df_pib.columns = df_pib.columns.str.strip()

# === 3. Conversie dată din text în datetime ===
def extrage_data(trimestru_str):
    p = trimestru_str.strip().split()
    if len(p) == 3:
        trimestru, anul = p[1], p[2]
        luna_start = {"I": "01", "II": "04", "III": "07", "IV": "10"}[trimestru]
        return pd.to_datetime(f"{anul}-{luna_start}-01")
    return None

df_somaj["Data"] = df_somaj["Perioade"].apply(extrage_data)
df_pib["Data"] = df_pib["Trimestre"].apply(extrage_data)

# === 4. Selectare și redenumire coloane relevante ===
df_somaj = df_somaj[["Data", "Valoare"]].rename(columns={"Valoare": "Somaj"})
df_pib = df_pib[["Data", "Valoare"]].rename(columns={"Valoare": "PIB"})

# === 5. Conversie PIB în procente (scădem 100) ===
df_pib["PIB"] = df_pib["PIB"] - 100  # ex: 102.1 → 2.1%

# === 6. Aliniere după Data (merge) ===
df_final = pd.merge(df_somaj, df_pib, on="Data", how="inner")
df_final = df_final.sort_values("Data").reset_index(drop=True)

# === 7. Salvare rezultat final ===
df_final.to_csv("serii_multivariate.csv", index=False)
print("Fișierul 'serii_multivariate.csv' a fost salvat cu succes.")

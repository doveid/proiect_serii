import pandas as pd

# Încarcă fișierul CSV cu șomajul (ajustează calea dacă e necesar)
df_somaj = pd.read_csv("../proiectSeriiRevizuit/exportPivot_AMG130A.csv", encoding="utf-8")

# Curăță numele coloanelor (elimină spațiile)
df_somaj.columns = df_somaj.columns.str.strip()

# Extrage data calendaristică din coloana "Perioade"
def extrage_data_somaj(perioada_str):
    p = perioada_str.strip().split()
    if len(p) == 3:
        trimestru, anul = p[1], p[2]
        luna_start = {"I": "01", "II": "04", "III": "07", "IV": "10"}[trimestru]
        return pd.to_datetime(f"{anul}-{luna_start}-01")
    return None

df_somaj["Data"] = df_somaj["Perioade"].apply(extrage_data_somaj)

# Convertire valori în număr întreg (șomeri)
df_somaj["Numar someri"] = pd.to_numeric(df_somaj["Valoare"], errors="coerce")

# Selectează doar coloanele relevante și sortează cronologic
df_somaj_clean = df_somaj[["Data", "Numar someri"]].sort_values("Data").reset_index(drop=True)

# Afișează primele rânduri (opțional)
print(df_somaj_clean.head())

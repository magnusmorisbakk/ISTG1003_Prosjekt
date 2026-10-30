# =============================================================================
# Klassifisering av overlevelse på Titanic med k-nærmeste nabo (KNN)
# =============================================================================

# --- Importer ---------------------------------------------------------------
import pandas as pd                                   # datahåndtering (tabeller)
import numpy as np                                    # tallrekker og numeriske operasjoner
import matplotlib.pyplot as plt                       # plotting
from sklearn.model_selection import train_test_split  # deling i trening/validering/test
from sklearn.metrics import accuracy_score            # andel riktige prediksjoner
from sklearn.neighbors import KNeighborsClassifier    # KNN-klassifikatoren
from sklearn.preprocessing import StandardScaler      # standardisering av forklaringsvariabler

# --- Last inn data ----------------------------------------------------------
df = pd.read_csv("Jupyterhub/Titanic-Dataset.csv")

'''
Problemstilling (definert FØR vi så resultatene):
Jeg ønsker å forsøke og klassifiere om en passasjer overlevde eller ikke.
Først vil jeg sjekke basert på Kjønn og alder.
Så vil jeg også sjekke om status (Klasse) har en påvirkning
Så sjekker jeg om antall familiemedlemmer om bord har noe å si
'''

# --- Forbehandling av data --------------------------------------------------

# Lager en ny variabel: antall familiemedlemmer om bord.
# SibSp = søsken/ektefelle, Parch = foreldre/barn. Summen gir total familiestørrelse
# (passasjeren selv er ikke talt med).
df["FamilySize"] = df["SibSp"] + df["Parch"]

# KNN regner avstander, så alle forklaringsvariabler må være tall.
# Sex gjøres om til 0/1 (male = 0, female = 1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# --- Deling av data ---------------------------------------------------------

# 60% trening, 20% validering, 20% testing.
# Steg 1: 60% trening / 40% "temp". Steg 2: temp deles 50/50 -> 20% validering og 20% test.
# - stratify=... gir lik andel "Survived" i alle tre sett (viktig siden klassene er skjevfordelt)
# - random_state=1 gjør delingen reproduserbar
# Treningssettet brukes til å tilpasse modellen, valideringssettet til å velge modell og k,
# og testsettet brukes KUN til slutt for et ærlig estimat av ytelsen.
df_trening, temp = train_test_split(df, test_size=0.4, random_state=1, stratify=df["Survived"])
df_val, df_test = train_test_split(temp, test_size=0.5, random_state=1, stratify=temp["Survived"])

# --- Definisjon av modellene ------------------------------------------------

# Tre modeller med gradvis flere forklaringsvariabler (hver bygger på den forrige),
# slik at vi kan se om hver ekstra variabel forbedrer prediksjonen.
modeller_def = {
    "Modell 1: Sex + Age": ['Sex', 'Age'],
    "Modell 2: + Pclass": ['Sex', 'Age', 'Pclass'],
    "Modell 3: + FamilySize": ['Sex', 'Age', 'Pclass', 'FamilySize'],
}

# Samme valideringsrader for alle modeller, slik at de er sammenlignbare.
# Vi bruker alle kolonner fra modell 3 og fjerner rader med manglende verdi (hovedsakelig Age).
# Da er valideringssettet identisk uansett hvilken modell som evalueres.
alle_kolonner = ['Survived', 'Sex', 'Age', 'Pclass', 'FamilySize']
df_val_clean = df_val[alle_kolonner].dropna()
y_val = df_val_clean['Survived']

# Verdier av k vi prøver (kun odde tall, slik at vi unngår uavgjort i flertallsavstemningen)
knaboer = np.arange(1, 199, step=2)

# Beholdere for resultater
resultater = {}        # beste k og tilhørende valideringsfeilrate per modell
val_feilrater = {}     # hele feilrate-kurven (mot k) per modell, brukes til plotting
beste_modeller = {}    # lagrer scaler, beste k, kolonner og treningsdata per modell

# --- Tilpasning og valg av k for hver modell --------------------------------
for navn, x_kolonner in modeller_def.items():
    # Treningsdata (fjern rader med manglende verdier)
    df_tren = df_trening[['Survived'] + x_kolonner].dropna()

    # Standardiser slik at Age (0-80) ikke dominerer avstanden over Sex (0/1).
    # Skaleringen læres kun fra treningsdata (fit_transform), og valideringsdata
    # skaleres med samme parametre (transform) for å unngå "data leakage".
    scaler = StandardScaler()
    X_tren = scaler.fit_transform(df_tren[x_kolonner])
    X_val = scaler.transform(df_val_clean[x_kolonner])
    y_tren = df_tren['Survived']

    # Feilrate på valideringssettet for hver k
    val_feilrate = np.empty(len(knaboer))
    for i, k in enumerate(knaboer):
        # p=2 betyr euklidsk avstand
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(X_tren, y_tren)
        # score() gir andel riktige, så feilrate = 1 - score
        val_feilrate[i] = 1 - knn.score(X_val, y_val)

    val_feilrater[navn] = val_feilrate

    # Velg k med lavest feilrate på valideringssettet
    i_beste = np.argmin(val_feilrate)
    beste_modeller[navn] = (scaler, knaboer[i_beste], x_kolonner, df_tren)
    resultater[navn] = {
        "Beste k": knaboer[i_beste],
        "Feilrate (val)": val_feilrate[i_beste],
    }

# --- Baseline og oppsummering -----------------------------------------------

# Baseline: alltid gjette den vanligste klassen.
# Modellene våre må slå denne for å være nyttige.
baseline = 1 - y_val.value_counts(normalize=True).max()

print(pd.DataFrame(resultater).T)
print("Baseline feilrate:", baseline)

# --- Plot: feilrate mot k for hver modell -----------------------------------
# Viser hvordan fleksibiliteten (liten k = fleksibel, stor k = stiv) påvirker feilraten.
plt.figure(figsize=(8, 5))
for navn, feilrate in val_feilrater.items():
    plt.plot(knaboer, feilrate, label=navn)
plt.axhline(baseline, color="gray", linestyle="--", label="Baseline")
plt.xlabel("k (antall naboer)")
plt.ylabel("Feilrate (validering)")
plt.legend()
plt.show()

# --- Endelig modellvalg -----------------------------------------------------

# 1. Velg beste modell: lavest feilrate på valideringssettet
resultater_df = pd.DataFrame(resultater).T
beste_navn = resultater_df["Feilrate (val)"].idxmin()
scaler, beste_k, x_kolonner, df_tren = beste_modeller[beste_navn]

print("Beste modell:", beste_navn)
print("Beste k:", beste_k)
print("Feilrate (val):", resultater_df.loc[beste_navn, "Feilrate (val)"])

# 2. Tren endelig modell på treningsdata med beste k
# (scaleren er den som ble tilpasset treningsdata for akkurat denne modellen)
X_tren = scaler.transform(df_tren[x_kolonner])
y_tren = df_tren['Survived']

knn_best = KNeighborsClassifier(n_neighbors=int(beste_k), p=2)
knn_best.fit(X_tren, y_tren)

# --- Evaluering på testsettet (brukes kun én gang) --------------------------

# 3. Predikasjon på testsettet (fjern rader med manglende verdier, bruk samme skalering)
df_test_clean = df_test[['Survived'] + x_kolonner].dropna()
X_test = scaler.transform(df_test_clean[x_kolonner])
y_test = df_test_clean['Survived']

y_testpred = knn_best.predict(X_test)

# Feilrate på testsett og tilhørende baseline, for et ærlig estimat på ukjente data
test_feilrate = 1 - accuracy_score(y_true=y_test, y_pred=y_testpred)
baseline_test = 1 - y_test.value_counts(normalize=True).max()

print("Feilrate (test):", test_feilrate)
print("Baseline feilrate (test):", baseline_test)

# Forvekslingsmatrise: viser hvor mange som ble klassifisert riktig/feil i hver klasse
# (rader = faktisk, kolonner = predikert). Nyttig for å se sensitivitet og spesifisitet.
from sklearn.metrics import confusion_matrix
print(pd.DataFrame(
    confusion_matrix(y_test, y_testpred),
    index=["Faktisk: døde", "Faktisk: overlevde"],
    columns=["Pred: døde", "Pred: overlevde"],
))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_testpred)

# Tegner forvekslingsmatrisen som et fargekart (rader = faktisk, kolonner = predikert)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Døde", "Overlevde"],
)
disp.plot(cmap="Blues")
plt.title("Forvekslingsmatrise (testsett)")
plt.xlabel("Predikert")
plt.ylabel("Faktisk")
plt.show()
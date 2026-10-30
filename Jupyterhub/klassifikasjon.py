import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Jupyterhub/Titanic-Dataset.csv")

''' 
Jeg ønsker å forsøke og klassifiere om en passasjer overlevde eller ikke.
Først vil jeg sjekke basert på Kjønn og alder. 
Så vil jeg også sjekke om status(Klasse) har en påvirkning
Så sjekker jeg om antall familiemedlemmer om bord har noe å si 
'''
df["FamilySize"] = df["SibSp"] + df["Parch"]


# https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/
# 60% til trenging, 20% til validering og 20% til testing
# Stratisfy sørger for at forholdet av "Survived" er tilsvarende likt i alle sett. Det er altså ikke helt tilfeldig
df_trening, temp = train_test_split(df, test_size=0.4, random_state=1, stratify=df["Survived"])
df_val, df_test = train_test_split(temp, test_size=0.5, random_state=1, stratify=temp["Survived"])


'''
Her kan vi se att kjønn har en kjempestor påvirknig på sannsynligheten (git att alder er låst)
Vi ser også at Alder ikke har en signifikant påvirkning på sannsynligheten (Gitt at kjønn er låst)

'''
# Treningsdata per modell (fjern rader med manglende verdier)
df_1 = df_trening[['Survived', 'Sex', 'Age']].dropna()
df_2 = df_trening[['Survived', 'Sex', 'Age', 'Pclass']].dropna()
df_3 = df_trening[['Survived', 'Sex', 'Age', 'Pclass', 'FamilySize']].dropna()

# Modellnavn: (formel, treningsdata, kolonner som trengs i val/test)
modeller_def = {
    "Modell 1: Sex + Age": ('Survived ~ Sex + Age', df_1),
    "Modell 2: + Pclass": ('Survived ~ Sex + Age + C(Pclass)', df_2),
    "Modell 3: + FamilySize": ('Survived ~ Sex + Age + C(Pclass) + FamilySize', df_3),
}

# Bruk samme valideringsrader for alle modeller, slik at de er sammenlignbare
df_val_clean = df_val[['Survived', 'Sex', 'Age', 'Pclass', 'FamilySize']].dropna()

cutoff = 0.5
resultater = {}
modeller = {}

for navn, (formel, data) in modeller_def.items():
    modell = smf.logit(formel, data=data)
    resultat = modell.fit(disp=0)
    modeller[navn] = resultat
    print(navn)
    print(resultat.summary())

    val_pred = resultat.predict(df_val_clean)
    y_valpred = np.where(val_pred > cutoff, 1, 0)
    y_valobs = df_val_clean['Survived']

    resultater[navn] = {
        "Feilrate (val)": 1 - accuracy_score(y_true=y_valobs, y_pred=y_valpred),
        "AIC (trening)": resultat.aic,
    }

# Baseline: alltid gjette den vanligste klassen
baseline = 1 - df_val_clean['Survived'].value_counts(normalize=True).max()

print(pd.DataFrame(resultater).T)
print("Baseline feilrate:", baseline)

from sklearn.metrics import confusion_matrix

# --- Velg beste modell: lavest feilrate på valideringsdata ---
# Ved lik feilrate velges modellen med lavest AIC (trening)
tabell = pd.DataFrame(resultater).T
beste_navn = tabell.sort_values(["Feilrate (val)", "AIC (trening)"]).index[0]
beste_modell = modeller[beste_navn]

print("\nBeste modell:", beste_navn)
print("Feilrate (val):", round(tabell.loc[beste_navn, "Feilrate (val)"], 4))

# --- Test den valgte modellen på testdatasettet (brukes kun én gang) ---
df_test_clean = df_test[['Survived', 'Sex', 'Age', 'Pclass', 'FamilySize']].dropna()

test_pred = beste_modell.predict(df_test_clean)
y_testpred = np.where(test_pred > cutoff, 1, 0)
y_testobs = df_test_clean['Survived']

feilrate_test = 1 - accuracy_score(y_true=y_testobs, y_pred=y_testpred)
baseline_test = 1 - y_testobs.value_counts(normalize=True).max()

print("\nFeilrate (test):", round(feilrate_test, 4))
print("Baseline feilrate (test):", round(baseline_test, 4))

# Forvirringsmatrise (rader = faktisk, kolonner = predikert)
print(pd.DataFrame(
    confusion_matrix(y_testobs, y_testpred),
    index=["Faktisk: døde", "Faktisk: overlevde"],
    columns=["Pred: døde", "Pred: overlevde"],
))

import pandas as pd
import matplotlib as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

df = pd.read_csv("Jupyterhub/Titanic-Dataset.csv")

''' 
Jeg ønsker å forsøke og klassifiere om en passasjer overlevde eller ikke.
Først vil jeg sjekke basert på Kjønn og alder. 
Så vil jeg også sjekke om status(Klasse) har en påvirkning
Så sjekker jeg om antall familiemedlemmer om bord har noe å si 
'''

#Hentet fra lego oppgave
df_1 = df[['Survived','Sex', 'Age']]

#Fjern rader med ukjente vriabler (F.eks alder)
df_1 = df_1.dropna()

# https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/
# 60% til trenging, 20% til validering og 20% til testing
# Stratisfy sørger for at forholdet av "Survived" er tilsvarende likt i alle sett. Det er altså ikke helt tilfeldig
df_trening, temp = train_test_split(df_1, test_size=0.4, random_state=1, stratify=df_1["Survived"])
df_val, df_test = train_test_split(temp, test_size=0.5, random_state=1, stratify=temp["Survived"])

formel = 'Survived ~ Sex + Age'
modell = smf.logit(formel, data=df_trening)
resultat = modell.fit()
print(resultat.summary())
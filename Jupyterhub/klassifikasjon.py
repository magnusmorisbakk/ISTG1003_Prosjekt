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

# https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/
# 60% til trenging, 20% til validering og 20% til testing
# Stratisfy sørger for at forholdet av "Survived" er tilsvarende likt i alle sett. Det er altså ikke helt tilfeldig
df_trening, temp = train_test_split(df, test_size=0.4, random_state=1, stratify=df["Survived"])
df_val, df_test = train_test_split(temp, test_size=0.5, random_state=1, stratify=temp["Survived"])


df_1 = df_trening[['Survived','Sex', 'Age']]

#Fjern rader med ukjente vriabler (F.eks alder)
df_1 = df_1.dropna()

df_2 = df_trening[['Survived','Sex', 'Age', 'Pclass']]
df_2 = df_2.dropna()

df_3 = df_trening[['Survived','Sex', 'Age', 'Pclass', 'SibSp', 'Parch']]
#Add family size and drop un-needed variables
df_3['FamilySize'] = df_trening['SibSp'] + df_trening['Parch']
df_3 = df_3.drop(columns=['SibSp', 'Parch'])
df_3 = df_3.dropna()


'''
Her kan vi se att kjønn har en kjempestor påvirknig på sannsynligheten (git att alder er låst)
Vi ser også at Alder ikke har en signifikant påvirkning på sannsynligheten (Gitt at kjønn er låst)

'''
formel = 'Survived ~ Sex + Age'
modell = smf.logit(formel, data=df_1)
resultat = modell.fit()
print(resultat.summary())

formel = 'Survived ~ Sex + Age + Pclass'
modell = smf.logit(formel, data=df_2)
resultat = modell.fit()
print(resultat.summary())

formel = 'Survived ~ Sex + Age + Pclass + FamilySize'
modell = smf.logit(formel, data=df_3)
resultat = modell.fit()
print(resultat.summary())
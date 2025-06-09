import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Carica dati e pulizia
df = pd.read_csv('Salary_Data.csv')
df.dropna(inplace=True)

# Dividi train/test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Definisci feature e target
X_train = df_train[["Years of Experience"]]
y_train = df_train["Salary"]

X_test = df_test[["Years of Experience"]]
y_test = df_test["Salary"]

# Allena modello
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Imposta parametro positive=True se ti serve
linreg.set_params(positive=True)

# Previsione su test
y_test_pred = linreg.predict(X_test)

# Crea cartella se non esiste
os.makedirs('boiocchi', exist_ok=True)

# Salva modello con joblib
dump(linreg, 'boiocchi/artifact.joblib')

print("Modello salvato correttamente in 'boiocchi/artifact.joblib'")

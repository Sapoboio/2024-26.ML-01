import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer, r2_score, mean_squared_error
import os
from joblib import dump



csv_path = os.path.join(os.path.dirname(__file__), 'Salary_Data.csv')
df = pd.read_csv(csv_path)
df[df.isnull().any(axis=1)]
df.dropna(inplace=True)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
X_train = df_train[["Years of Experience"]]
y_train = df_train["Salary"]

X_test = df_test[["Years of Experience"]]
y_test = df_test["Salary"]
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.get_params()
linreg.set_params(positive=True)
y_test_pred = linreg.predict(X_test)
y_test_pred

os.makedirs('boiocchi', exist_ok=True)
dump(linreg, 'boiocchi/artifact.joblib')
print("Modello salvato in 'boiocchi/artifact.joblib'")
print("sddyt")
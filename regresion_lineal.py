import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




df = pd.read_csv('datos_procesados.csv')


X = df.drop(columns=['DEATH_EVENT', 'age'])


y = df['age']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


modelo_regresion = LinearRegression()


modelo_regresion.fit(X_train, y_train)


y_pred = modelo_regresion.predict(X_test)


error_cuadratico_medio = mean_squared_error(y_test, y_pred)


print(f"Error cuadrático medio: {error_cuadratico_medio}")


print("\nEdades reales vs. Edades predichas:")
comparacion_edades = pd.DataFrame({'Edad Real': y_test, 'Edad Predicha': y_pred})
print(comparacion_edades)
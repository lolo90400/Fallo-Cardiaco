#PARTE 1


from datasets import load_dataset
import numpy as np 

dataset = load_dataset("mstz/heart_failure")

data = dataset["train"]

edades = np.array(data["age"])
promedio_de_edad = np.mean(edades)

print(f"El promedio de la edad de los participantes es,{promedio_de_edad}")

#PARTE 2 

import pandas as pd 

df = pd.DataFrame(data)
df_1 = df.query("is_dead==1")
print(f"La cantidad de pacientes muertos es: {df_1.shape[0]}")
df_2 = df.query("is_dead==0")
print(f"La cantidad de pacientes vivos es: {df_2.shape[0]}")

promedio_df_1 = df_1["age"].mean()
print(F"El promedio de la edad de las  personas muertas es: {round(promedio_df_1,2)}")
promedio_df_2 = df_2["age"].mean()
print(f" El promedio de la edad de las personas vivas es: {round(promedio_df_2,2)}")

# PARTE 3

print(df.dtypes)

df['is_dead'] = df['is_dead'].astype(bool)

print(df.dtypes)
print(df["is_dead"].head())

df_agrupado = df.groupby(['is_male', 'is_smoker']).size().reset_index(name='count')

print(df_agrupado)

conteo_fumadoras = df_agrupado.loc[(df_agrupado['is_male'] == False) & (df_agrupado['is_smoker'] == True), 'count'].values
conteo_fumadores = df_agrupado.loc[(df_agrupado['is_male'] == True) & (df_agrupado['is_smoker'] == True), 'count'].values

print("Cantidad de mujeres fumadoras:", conteo_fumadoras[0] if len(conteo_fumadoras) > 0 else 0)
print("Cantidad de hombres fumadores:", conteo_fumadores[0] if len(conteo_fumadores) > 0 else 0)

# PARTE 4

import requests

def descargar_y_guardar_csv(url, nombre_archivo):
    
    response = requests.get(url)

    
    if response.status_code == 200:
       
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(response.text)
        print(f"Datos descargados y guardados en {nombre_archivo}")
    else:
        print(f"Error en la solicitud: {response.status_code}")


url_datos = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"


descargar_y_guardar_csv(url_datos, "datos_descargados.csv")

# PARTE 5

def limpieza_preparacion_datos(df : pd.DataFrame):

  
  if df.isnull().any().any():
    print("Hay valores faltantes en el DataFrame. Realiza la imputación o eliminación según sea necesario.")
    
    df = df.fillna(df.mean()) 
  else:
    print("No hay valores faltantes en el DataFrame.")

  
  if df.duplicated().any():
    print("Hay valores duplicados en el DataFrame. se relizarán las correcciones necesarias")
    
    df = df.drop_duplicates() 
  else:
    print("No hay valores repetidos en el DataFrame.")  

  
  Q1 = df.quantile(0.25)  
  Q3 = df.quantile(0.75)  

  IQR = Q3 - Q1   

  limite_min = Q1 - (1.5 * IQR) 
  limite_max = Q3 + (1.5 * IQR) 

    
  df = df[(df >= limite_min) & (df <= limite_max)]

  
  df['categoria_edad'] = pd.cut(df['age'],
                                  bins=[-float("inf"), 12, 19, 39, 59, float('inf')],
                                  labels=['Niño', 'Adolescente', 'Joven adulto', 'Adulto', 'Adulto mayor'],
                                  right=True
                                  )
    # Guardar el resultado como CSV
  df.to_csv("datos_corregidos.csv", index=False)
  print("Datos limpios y preparados guardados como 'datos_corregidos.csv'.")

df = pd.read_csv("datos_descargados.csv")

limpieza_preparacion_datos(df)





 
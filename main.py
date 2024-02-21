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





 
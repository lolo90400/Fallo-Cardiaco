from datasets import  load_dataset
from main import descargar_y_guardar_csv, limpieza_preparacion_datos
import numpy as np
import pandas as pd

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

ages = np.array(data['age'])
has_anemia = np.array(data['has_anaemia'])

average_age = np.mean(ages)

sum_has_anemia = np.sum(has_anemia)
# TODO: print(has_anemia) 
# TODO: print(data)
# TODO: print(ages)
print(f"\n{sum_has_anemia} personas tienen anemia.")
print(f"El promedio de edad de las personas participantes en el estudio es: {average_age:.2f}\n" )

df = pd.DataFrame(data)
dead_df = df[df['is_dead'] == 1]
survived_df = df[df['is_dead'] == 0]
av_age_dead = dead_df['age'].mean()
av_age_survived = survived_df['age'].mean()
print(f"Promedio de edad de personas que perecieron: {av_age_dead:.2f}")
print(f"Promedio de edad de personas que sobrevivieron: {av_age_survived:.2f}")

smoker_counts = df.groupby(['is_male', 'is_smoker']).size().unstack()

smoker_counts.columns = ['No Fumador', 'Fumador']

smoker_counts.index = ['Mujer', 'Hombre']
print("\nCantidad de hombres fumadores vs mujeres fumadoras:")
print(f"{smoker_counts}\n")

url = 'https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv'


descargar_y_guardar_csv(url, 'datos5.csv')
descargar_y_guardar_csv(url, 'datos_procesados.csv')

df = pd.read_csv('datos5.csv')
df = pd.read_csv('datos_procesados.csv')
df_cleaned = limpieza_preparacion_datos(df)
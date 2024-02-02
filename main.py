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





 
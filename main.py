#PARTE 1


from datasets import load_dataset
import numpy as np 
dataset = load_dataset("mstz/heart_failure")

data = dataset["train"]

edades = np.array(data["age"])
promedio_de_edad = np.mean(edades)

print(f"El promedio de la edad de los participantes es,{promedio_de_edad}")


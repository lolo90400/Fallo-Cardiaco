import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from sklearn.manifold import TSNE 



df = pd.read_csv('datos_procesados.csv')


plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black', align='mid')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()




m = df[df['sex'] == 1]  
f = df[df['sex'] == 0]  


smoking_m = m['smoking'].sum()  
diabetes_m = m['diabetes'].sum()
anemia_m = m['anaemia'].sum()
deaths_m = m['DEATH_EVENT'].sum()


smoking_f = f['smoking'].sum()
diabetes_f = f['diabetes'].sum()
anemia_f = f['anaemia'].sum() 
deaths_f = f['DEATH_EVENT'].sum()


categorias = ['Anemicos', 'Diabeticos', 'Fumadores', 'Muertos']
datos_m = [anemia_m, diabetes_m, smoking_m, deaths_m]
datos_f = [anemia_f, diabetes_f, smoking_f, deaths_f]

index = np.arange(len(datos_m))
width = 0.30


fig, ax = plt.subplots()

ax.bar(index-width/2, datos_m, width)
ax.bar(index+width/2, datos_f, width)

for i, j in zip(index, datos_m):
        ax.annotate(j, xy=(i-0.2,j+0.2))

for i, j in zip(index, datos_f):
        ax.annotate(j, xy=(i+0.1,j+0.2))

ax.set_title('Gráfico de comparación por género')
ax.set_xticks(index)
ax.set_xticklabels(categorias)
ax.set_xlabel('Categorias')
fig.legend(['Hombres', 'Mujeres'], loc='upper right', fontsize='small')

fig.savefig('comparacion_por_genero_.png')
plt.show()
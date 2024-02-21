import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Parte 11

def graficar_distribucion_clases_1(df, columna_clases='DEATH_EVENT'):
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=columna_clases, data=df)
    plt.title('Distribución de Clases')
    plt.xlabel(columna_clases)
    plt.ylabel('Cantidad de Observaciones')
    plt.savefig('distribucion_clases.png')  
    plt.show()

def particion_estratificada_1(df, columna_clases='DEATH_EVENT'):
    
    X = df.drop(columns=[columna_clases])
    y = df[columna_clases]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def ajustar_arbol_decision_1(X_train, X_test, y_train, y_test):
    
    modelo_arbol = DecisionTreeClassifier(random_state=42)
    modelo_arbol.fit(X_train, y_train)
    y_pred = modelo_arbol.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo de árbol de decisión: {precision}")

    
    from sklearn.metrics import confusion_matrix
    matriz_confusion = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(matriz_confusion)
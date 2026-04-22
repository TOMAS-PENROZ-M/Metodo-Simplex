import numpy as np

tabla = np.array([
    [0, 1, 0, 0, 0, 1, 100],
    [1, 0, 0, 0, 1, 0, 45],
    [1, 3, 0, 1, 0, 0, 80],   
    [2, 1, 1, 0, 0, 0, 100],   
    [-2, -2, 0, 0, 0, 0, 0]  # función objetivo
], dtype=float)

columnas_nombres = ["x1", "x2", "s1", "s2", "s3", "s4", "b"]
filas_nombres = ["s1", "s2", "s3", "s4", "Z"]

def print_tabla(tabla, paso, columnas, filas):
    print(f"Iteracion:  {paso}")
    
    print("     ", end="")
    for col in columnas:
        print(f"{col:>8}", end="")
    print()

    for i, fila in enumerate(tabla):
        print(f"{filas[i]:>4}", end="")
        for val in fila:
            print(f"{val:>8.2f}", end="")
        print() 
    
def simplex(tabla, columnas, filas):
    n_filas, n_cols = tabla.shape
    paso = 0

    while True:
        print_tabla(tabla, paso, columnas, filas)

        # 1. Elegir columna pivote (el más negativo en Z)
        col_pivote = np.argmin(tabla[-1, :-1])
        
        if tabla[-1, col_pivote] >= 0:
            break  # ya es óptimo

        print(f"Columna pivote: {columnas[col_pivote]}")
        
        # 2. Elegir fila pivote (mínima razón positiva)
        ratios = []
        for i in range(n_filas - 1):
            if tabla[i, col_pivote] > 0:
                ratios.append(tabla[i, -1] / tabla[i, col_pivote])
            else:
                ratios.append(np.inf)

        fila_pivote = np.argmin(ratios)

        if ratios[fila_pivote] == np.inf:
            print("No hay solución factible.")
            return None

        print(f"Fila pivote: {filas[fila_pivote]}")

        filas[fila_pivote] = columnas[col_pivote]

        # 3. Normalizar fila pivote
        pivote = tabla[fila_pivote, col_pivote]
        print(f"Pivote: {pivote}")
        tabla[fila_pivote] /= pivote

        # 4. Hacer ceros en la columna pivote
        for i in range(n_filas):
            if i != fila_pivote:
                tabla[i] -= tabla[i, col_pivote] * tabla[fila_pivote]

        paso +=1
        print_tabla(tabla, paso, columnas, filas)

    return tabla

resultado = simplex(tabla, columnas_nombres, filas_nombres)
print("Tabla final:\n", resultado)

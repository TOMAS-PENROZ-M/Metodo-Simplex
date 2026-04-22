import json
import os
from .model import ProblemaLineal

def get_int(msg, min_val=1):
    while True:
        try:
            val = int(input(msg))
            if val < min_val:
                print(f"Error: El valor debe ser mayor o igual a {min_val}.")
            else:
                return val
        except ValueError:
            print("Error: Por favor ingresa un número entero válido.")

def get_float_list(msg, expected_qty):
    while True:
        raw = input(msg)
        # Dividir por comas o espacios
        raw_list = raw.replace(',', ' ').split()
        if len(raw_list) != expected_qty:
            print(f"Error: Se esperaban exactamente {expected_qty} valores. Ingresaste {len(raw_list)}.")
            continue
            
        try:
            float_list = [float(x) for x in raw_list]
            return float_list
        except ValueError:
            print("Error: Todos los valores deben ser numéricos (enteros o decimales).")

def build_problem_from_console():
    """
    Guía interactiva para ingresar el problema de optimización.
    """
    print("\n--- INGRESO INTERACTIVO DE DATOS ---")
    
    # 1. Minimizar o Maximizar
    while True:
        tipo = input("¿Tipo de optimización? (max/min): ").strip().lower()
        if tipo in ['max', 'min']:
            break
        print("Error: Por favor ingresa 'max' o 'min'.")
        
    problem = ProblemaLineal(tipo_objetivo=tipo)
    
    # 2. Cantidad de variables
    num_vars = get_int("Ingresa la CANTIDAD de variables de decisión (ej. 2): ")
    
    # 3. Coeficientes Función Objetivo
    print(f"\nIngresa los {num_vars} coeficientes de la Función Objetivo separados por espacios:")
    c_coeffs = get_float_list("Coeficientes (ej. 3 2): ", num_vars)
    problem.definir_funcion_objetivo(c_coeffs)
    
    # 4. Cantidad de restricciones
    num_restr = get_int("\nIngresa la CANTIDAD de restricciones (excluyendo no negatividad): ")
    
    # 5. Restricciones
    for i in range(num_restr):
        print(f"\n--- Restricción {i+1} ---")
        a_coeffs = get_float_list(f"Coeficientes de las {num_vars} variables (ej. 2 1): ", num_vars)
        ld = get_float_list(f"Lado derecho de la restricción (RHS): ", 1)[0]
        # De acuerdo al plan original se evalúan inecuaciones '<=' por ahora en el conversor a forma estándar estándar.
        problem.agregar_restriccion(a_coeffs, ld, tipo_inecuacion='<=')
        
    return problem

def build_problem_from_json(filepath):
    """
    Opcional: Lee un PPL desde un archivo JSON.
    Ejemplo JSON:
    {
       "type": "max",
       "objective": [3, 2],
       "constraints": [
           {"coeffs": [2, 1], "bound": 18},
           {"coeffs": [2, 3], "bound": 42}
       ]
    }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    problem = ProblemaLineal(tipo_objetivo=data.get("type", "max"))
    problem.definir_funcion_objetivo(data["objective"])
    
    for cnst in data.get("constraints", []):
        problem.agregar_restriccion(cnst["coeffs"], cnst["bound"], cnst.get("type", "<="))
        
    return problem

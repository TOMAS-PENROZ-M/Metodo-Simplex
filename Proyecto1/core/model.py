import numpy as np

class ProblemaLineal:
    """
    Clase para modelar un Problema de Programación Lineal (PPL).
    """
    def __init__(self, tipo_objetivo='max'):
        """
        Inicializa un nuevo problema de programación lineal.
        
        Args:
            tipo_objetivo (str): 'max' para maximizar, 'min' para minimizar.
        """
        self.tipo_objetivo = tipo_objetivo.lower()
        self.coeficientes_funcion_objetivo = []  # Coeficientes de la función objetivo (C)
        self.matriz_tecnologica_restricciones = [] # Matriz de coeficientes de restricciones (A)
        self.vector_lado_derecho_restricciones = [] # Lado derecho de las restricciones (LD o b)
        self.tipos_inecuacion_restricciones = []  # Tipo de restricción: '<=', '>=', '='
        self.cantidad_variables_decision = 0

    def definir_funcion_objetivo(self, coeficientes):
        """
        Define la función objetivo.
        
        Args:
            coeficientes (list o np.array): Coeficientes de las variables de decisión.
        """
        self.coeficientes_funcion_objetivo = np.array(coeficientes, dtype=float)
        self.cantidad_variables_decision = len(self.coeficientes_funcion_objetivo)

    def agregar_restriccion(self, coeficientes, lado_derecho, tipo_inecuacion='<='):
        """
        Añade una restricción al problema.
        
        Args:
            coeficientes (list o np.array): Coeficientes de las variables en la restricción.
            lado_derecho (float): Límite o lado derecho de la restricción.
            tipo_inecuacion (str): '<=', '>=', o '='. Por defecto es '<='.
        """
        if len(coeficientes) != self.cantidad_variables_decision and self.cantidad_variables_decision != 0:
            raise ValueError("El número de coeficientes debe coincidir con el de la función objetivo.")
        
        self.matriz_tecnologica_restricciones.append(coeficientes)
        self.vector_lado_derecho_restricciones.append(lado_derecho)
        self.tipos_inecuacion_restricciones.append(tipo_inecuacion)
        
        if self.cantidad_variables_decision == 0:
            self.cantidad_variables_decision = len(coeficientes)

    def generar_vectores_forma_estandar(self):
        """
        Convierte el problema a la forma estándar para el método Simplex.
        Maneja '<=', '>=' y '=' agregando variables de holgura, exceso y artificiales.
        
        Returns:
            vector_z (np.array): Coeficientes de la F.O. ajustados (tamaño variable).
            matriz_a (np.array): Matriz de restricciones con holguras/excesos/artificiales.
            vector_b (np.array): Lado derecho.
            indices_base_inicial (list): Índices de las variables básicas para iniciar Simplex.
            identificadores_var (list): Nombre textual de cada variable (ej. 'X1', 'H1', 'A1').
        """
        cantidad_restricciones = len(self.matriz_tecnologica_restricciones)
        
        # Copiar y asegurar que b >= 0
        matriz_tech = np.array(self.matriz_tecnologica_restricciones, dtype=float)
        b = np.array(self.vector_lado_derecho_restricciones, dtype=float)
        tipos_ineq = list(self.tipos_inecuacion_restricciones)
        
        for i in range(cantidad_restricciones):
            if b[i] < 0:
                b[i] = -b[i]
                matriz_tech[i, :] = -matriz_tech[i, :]
                if tipos_ineq[i] == '<=':
                    tipos_ineq[i] = '>='
                elif tipos_ineq[i] == '>=':
                    tipos_ineq[i] = '<='

        # Contar tipos de variables nuevas
        n_holguras = sum(1 for t in tipos_ineq if t == '<=')
        n_excesos = sum(1 for t in tipos_ineq if t == '>=')
        n_artificiales = sum(1 for t in tipos_ineq if t in ['>=', '='])
        
        total_vars = self.cantidad_variables_decision + n_holguras + n_excesos + n_artificiales
        
        matriz_a = np.zeros((cantidad_restricciones, total_vars))
        matriz_a[:, :self.cantidad_variables_decision] = matriz_tech
        
        identificadores_variables = [f"X{i+1}" for i in range(self.cantidad_variables_decision)]
        indices_base_inicial = [0] * cantidad_restricciones
        
        idx_var_actual = self.cantidad_variables_decision
        c_holg = 1
        c_exc = 1
        c_art = 1
        
        for i in range(cantidad_restricciones):
            if tipos_ineq[i] == '<=':
                matriz_a[i, idx_var_actual] = 1.0
                identificadores_variables.append(f"H{c_holg}")
                indices_base_inicial[i] = idx_var_actual
                idx_var_actual += 1
                c_holg += 1
            elif tipos_ineq[i] == '>=':
                # Exceso (-1)
                matriz_a[i, idx_var_actual] = -1.0
                identificadores_variables.append(f"E{c_exc}")
                idx_var_actual += 1
                c_exc += 1
                # Artificial (+1)
                matriz_a[i, idx_var_actual] = 1.0
                identificadores_variables.append(f"A{c_art}")
                indices_base_inicial[i] = idx_var_actual
                idx_var_actual += 1
                c_art += 1
            elif tipos_ineq[i] == '=':
                matriz_a[i, idx_var_actual] = 1.0
                identificadores_variables.append(f"A{c_art}")
                indices_base_inicial[i] = idx_var_actual
                idx_var_actual += 1
                c_art += 1
                
        # Vector Z (con tamaño total, rellenado con ceros)
        vector_z = np.zeros(total_vars)
        vector_z[:self.cantidad_variables_decision] = self.coeficientes_funcion_objetivo
        
        return vector_z, matriz_a, b, indices_base_inicial, identificadores_variables


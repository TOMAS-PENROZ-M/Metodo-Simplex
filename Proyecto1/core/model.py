import numpy as np
class ProblemaLineal:
    def __init__(self, tipo_objetivo='max'):
        self.tipo_objetivo = tipo_objetivo.lower()
        self.coeficientes_funcion_objetivo = []
        self.matriz_tecnologica_restricciones = []
        self.vector_lado_derecho_restricciones = []
        self.tipos_inecuacion_restricciones = []
        self.cantidad_variables_decision = 0

    def definir_funcion_objetivo(self, coeficientes):
        self.coeficientes_funcion_objetivo = np.array(coeficientes, dtype=float)
        self.cantidad_variables_decision = len(self.coeficientes_funcion_objetivo)

    def agregar_restriccion(self, coeficientes, lado_derecho, tipo_inecuacion='<='):
        if len(coeficientes) != self.cantidad_variables_decision and self.cantidad_variables_decision != 0:
            raise ValueError("El número de coeficientes debe coincidir con el de la función objetivo.")
        self.matriz_tecnologica_restricciones.append(coeficientes)
        self.vector_lado_derecho_restricciones.append(lado_derecho)
        self.tipos_inecuacion_restricciones.append(tipo_inecuacion)
        if self.cantidad_variables_decision == 0:
            self.cantidad_variables_decision = len(coeficientes)
            
    def generar_vectores_forma_estandar(self):
        cantidad_restricciones = len(self.matriz_tecnologica_restricciones)
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
                matriz_a[i, idx_var_actual] = -1.0
                identificadores_variables.append(f"E{c_exc}")
                idx_var_actual += 1
                c_exc += 1
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
        vector_z = np.zeros(total_vars)
        vector_z[:self.cantidad_variables_decision] = self.coeficientes_funcion_objetivo
        return vector_z, matriz_a, b, indices_base_inicial, identificadores_variables

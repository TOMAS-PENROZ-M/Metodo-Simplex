import numpy as np
class SolucionadorSimplex:
    def __init__(self, problema):
        self.problema = problema
        self.matriz_tabla_simplex = None
        self.indices_variables_basicas = []
        self.cantidad_variables_decision = problema.cantidad_variables_decision
        self.cantidad_restricciones = len(problema.matriz_tecnologica_restricciones)
        self.estado_solucion = "No Resuelto"
        self.solucion_optima_variables = None
        self.valor_optimo_z = 0.0
        self.logs = []
        self.identificadores_variables = []
    def _log(self, mensaje):
        self.logs.append(mensaje)
    def ejecutar_algoritmo_simplex(self):
        self.logs = []
        vector_z, matriz_a, vector_b, indices_base_inicial, identificadores_var = self.problema.generar_vectores_forma_estandar()
        self.identificadores_variables = identificadores_var
        self.indices_variables_basicas = indices_base_inicial.copy()
        indices_artificiales = [i for i, nombre in enumerate(identificadores_var) if nombre.startswith('A')]
        if indices_artificiales:
            self._log("\n" + "="*50)
            self._log(" INICIANDO ALGORITMO SIMPLEX - FASE 1")
            self._log("="*50)
            total_vars = len(vector_z)
            self.matriz_tabla_simplex = np.zeros((self.cantidad_restricciones + 1, total_vars + 1))
            for i in range(self.cantidad_restricciones):
                self.matriz_tabla_simplex[i, :total_vars] = matriz_a[i, :]
                self.matriz_tabla_simplex[i, -1] = vector_b[i]
            fila_w = np.zeros(total_vars + 1)
            for j in indices_artificiales:
                fila_w[j] = 1.0 
            self.matriz_tabla_simplex[-1, :] = fila_w
            self._imprimir_estado_tabla_simplex(0, "Fase 1 (Planteamiento)")
            for i in range(self.cantidad_restricciones):
                if self.indices_variables_basicas[i] in indices_artificiales:
                    self.matriz_tabla_simplex[-1, :] -= self.matriz_tabla_simplex[i, :]
            estado = self._iterar_simplex("Fase 1")
            if estado != "Óptimo":
                return None, None
            if self.matriz_tabla_simplex[-1, -1] < -1e-8:
                self.estado_solucion = "Infactible"
                self._log("\n>>> ERROR: El problema es infactible (W óptimo < 0). <<<")
                return None, None
            self._log("\n" + "="*50)
            self._log(" INICIANDO ALGORITMO SIMPLEX - FASE 2")
            self._log("="*50)
            if self.problema.tipo_objetivo == 'max':
                fila_z = -vector_z
            else:
                fila_z = vector_z
            self.matriz_tabla_simplex[-1, :total_vars] = fila_z
            self.matriz_tabla_simplex[-1, -1] = 0.0
            self._imprimir_estado_tabla_simplex(0, "Fase 2 (Planteamiento)")
            for i in range(self.cantidad_restricciones):
                idx_basica = self.indices_variables_basicas[i]
                if self.matriz_tabla_simplex[-1, idx_basica] != 0:
                    factor = self.matriz_tabla_simplex[-1, idx_basica]
                    self.matriz_tabla_simplex[-1, :] -= factor * self.matriz_tabla_simplex[i, :]
            self.indices_ignorados = indices_artificiales
            estado = self._iterar_simplex("Fase 2")
        else:
            self._log("\n" + "="*50)
            self._log(" INICIANDO ALGORITMO SIMPLEX")
            self._log("="*50)
            self.indices_ignorados = []
            total_vars = len(vector_z)
            self.matriz_tabla_simplex = np.zeros((self.cantidad_restricciones + 1, total_vars + 1))
            for i in range(self.cantidad_restricciones):
                self.matriz_tabla_simplex[i, :total_vars] = matriz_a[i, :]
                self.matriz_tabla_simplex[i, -1] = vector_b[i]
            if self.problema.tipo_objetivo == 'max':
                self.matriz_tabla_simplex[-1, :total_vars] = -vector_z
            else:
                self.matriz_tabla_simplex[-1, :total_vars] = vector_z
            self._imprimir_estado_tabla_simplex(0, "Fase Única (Planteamiento)")
            estado = self._iterar_simplex("Fase 2 (Única)")
        if estado == "Óptimo" or self.estado_solucion == "Óptimo":
            self.estado_solucion = "Óptimo"
            self._extraer_resultados_finales()
            self._log("\n" + "*"*50)
            self._log(">>> ¡Solución Óptima Alcanzada! <<<")
            for i, valor_variable in enumerate(self.solucion_optima_variables[:self.cantidad_variables_decision]):
                self._log(f"    X{i+1} = {valor_variable:.2f}")
            self._log(f"    Z  = {self.valor_optimo_z:.2f}")
            self._log("*"*50 + "\n")
            return self.solucion_optima_variables, self.valor_optimo_z
        return None, None
    def _iterar_simplex(self, nombre_fase):
        limite_iteraciones = 1000
        for iteracion in range(limite_iteraciones):
            if iteracion == 0 and ("Fase 1" in nombre_fase or nombre_fase == "Fase 2"):
                self._imprimir_estado_tabla_simplex(iteracion, f"{nombre_fase} (Ajustada)")
            else:
                self._imprimir_estado_tabla_simplex(iteracion, nombre_fase)
            costos_reducidos = self.matriz_tabla_simplex[-1, :-1].copy()
            if hasattr(self, 'indices_ignorados'):
                for idx in self.indices_ignorados:
                    costos_reducidos[idx] = np.inf
            if np.all(costos_reducidos >= -1e-10):
                self.estado_solucion = "Óptimo"
                return "Óptimo"
            indice_columna_pivote_entrante = np.argmin(costos_reducidos)
            self._log(f"> Identificando columna pivote: {self.identificadores_variables[indice_columna_pivote_entrante]} (valor: {costos_reducidos[indice_columna_pivote_entrante]:.2f})")
            valores_columna_pivote = self.matriz_tabla_simplex[:-1, indice_columna_pivote_entrante]
            valores_lado_derecho = self.matriz_tabla_simplex[:-1, -1]
            filas_divisibles_validas = valores_columna_pivote > 1e-10
            if not np.any(filas_divisibles_validas):
                self.estado_solucion = "No Acotado"
                self._log(">>> ERROR: Solución no acotada. <<<")
                return "No Acotado"
            resultados_prueba_cociente_minimo = np.full(self.cantidad_restricciones, np.inf)
            resultados_prueba_cociente_minimo[filas_divisibles_validas] = valores_lado_derecho[filas_divisibles_validas] / valores_columna_pivote[filas_divisibles_validas]
            indice_fila_pivote_saliente = np.argmin(resultados_prueba_cociente_minimo)
            self._log(f"> Fila de salida seleccionada: Índice {indice_fila_pivote_saliente}")
            self._log(f"> Elemento pivote: {self.matriz_tabla_simplex[indice_fila_pivote_saliente, indice_columna_pivote_entrante]:.2f}\n")
            self._aplicar_gauss_jordan_al_pivote(indice_fila_pivote_saliente, indice_columna_pivote_entrante)
        self.estado_solucion = "Iteraciones Excedidas"
        return "Iteraciones Excedidas"
    def _aplicar_gauss_jordan_al_pivote(self, indice_fila_pivote_saliente, indice_columna_pivote_entrante):
        self.indices_variables_basicas[indice_fila_pivote_saliente] = indice_columna_pivote_entrante
        valor_elemento_pivote_central = self.matriz_tabla_simplex[indice_fila_pivote_saliente, indice_columna_pivote_entrante]
        self.matriz_tabla_simplex[indice_fila_pivote_saliente, :] /= valor_elemento_pivote_central
        for indice_fila_actual in range(self.matriz_tabla_simplex.shape[0]):
            if indice_fila_actual != indice_fila_pivote_saliente:
                factor_multiplicador = self.matriz_tabla_simplex[indice_fila_actual, indice_columna_pivote_entrante]
                self.matriz_tabla_simplex[indice_fila_actual, :] -= factor_multiplicador * self.matriz_tabla_simplex[indice_fila_pivote_saliente, :]
    def _extraer_resultados_finales(self):
        cantidad_total_variables = self.matriz_tabla_simplex.shape[1] - 1
        self.solucion_optima_variables = np.zeros(cantidad_total_variables)
        for indice_fila, indice_variable_basica in enumerate(self.indices_variables_basicas):
            self.solucion_optima_variables[indice_variable_basica] = self.matriz_tabla_simplex[indice_fila, -1]
        self.valor_optimo_z = self.matriz_tabla_simplex[-1, -1]
        if self.problema.tipo_objetivo == 'min':
            self.valor_optimo_z = -self.valor_optimo_z
    def _imprimir_estado_tabla_simplex(self, numeracion_iteracion, nombre_fase="Iteración"):
        self._log(f"\n--- {nombre_fase} - Iteración {numeracion_iteracion} ---")
        encabezados_columnas = self.identificadores_variables + ["LD"]
        texto_cabecera_columnas = "".join([f"{encabezado:>8}" for encabezado in encabezados_columnas])
        self._log(texto_cabecera_columnas)
        self._log("-" * len(texto_cabecera_columnas))
        indice_z = self.matriz_tabla_simplex.shape[0] - 1
        texto_fila_z = "".join([f"{valor:8.2f}" for valor in self.matriz_tabla_simplex[indice_z, :]])
        self._log(texto_fila_z + "  (Fila Z/W)")
        self._log("-" * len(texto_cabecera_columnas))
        for indice_fila in range(self.matriz_tabla_simplex.shape[0] - 1):
            texto_fila_numerica = "".join([f"{valor:8.2f}" for valor in self.matriz_tabla_simplex[indice_fila, :]])
            idx_var = self.indices_variables_basicas[indice_fila]
            nombre_var = self.identificadores_variables[idx_var]
            self._log(texto_fila_numerica + f"  ({nombre_var})")
        self._log("-" * len(texto_cabecera_columnas))

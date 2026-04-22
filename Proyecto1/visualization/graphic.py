import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class SolucionadorGrafico:
    """
    Implementación del Método Gráfico para 2 dimensiones.
    """
    def __init__(self, problema):
        self.problema = problema
        if problema.cantidad_variables_decision != 2:
            raise ValueError("El método gráfico solo soporta 2 variables.")
            
        self.valor_optimo_z = None
        self.solucion_optima_variables = None
        self.vertices_validos_region_factible = []
        self.todas_las_intersecciones_calculadas = []
        self.estado_solucion = "No Resuelto"
        self.logs = []
        
    def _log(self, mensaje):
        self.logs.append(mensaje)

    def _calcular_intersecciones_sistemas_2x2(self):
        """Calcula las intersecciones entre restricciones y ejes."""
        # Convertir todo a igualdades A x = b
        matriz_a = np.array(self.problema.matriz_tecnologica_restricciones)
        vector_b = np.array(self.problema.vector_lado_derecho_restricciones)
        
        # Añadir restricciones de no negatividad x1 >= 0, x2 >= 0
        matriz_a = np.vstack([matriz_a, [1, 0], [0, 1]])
        vector_b = np.append(vector_b, [0, 0])
        
        cantidad_ecuaciones = len(vector_b)
        vertices_temporales = []
        
        # Combinar 2 a 2 para hallar intersecciones en el plano
        for indice_ecuacion_1, indice_ecuacion_2 in combinations(range(cantidad_ecuaciones), 2):
            submatriz_a = np.array([matriz_a[indice_ecuacion_1], matriz_a[indice_ecuacion_2]])
            subvector_b = np.array([vector_b[indice_ecuacion_1], vector_b[indice_ecuacion_2]])
            
            # Resolver sistema de 2x2 si no son paralelas
            try:
                coordenadas_interseccion = np.linalg.solve(submatriz_a, subvector_b)
                vertices_temporales.append(coordenadas_interseccion)
                self._log(f"  [Intersección hallada] -> x1={coordenadas_interseccion[0]:.2f}, x2={coordenadas_interseccion[1]:.2f}")
            except np.linalg.LinAlgError:
                continue # Sistema singular (rectas paralelas o coincidentes)
                
        self.todas_las_intersecciones_calculadas = vertices_temporales

    def _filtrar_puntos_fuera_de_region_factible(self):
        """Filtra los vértices evaluándolos en todas las inecuaciones."""
        matriz_a = np.array(self.problema.matriz_tecnologica_restricciones)
        vector_b = np.array(self.problema.vector_lado_derecho_restricciones)
        tipos_inecuacion = self.problema.tipos_inecuacion_restricciones
        
        vertices_factibles_temporales = []
        for coordenadas_punto in self.todas_las_intersecciones_calculadas:
            # Primero checar no negatividad (usando una ligera tolerancia de precisión de flotantes)
            if coordenadas_punto[0] < -1e-8 or coordenadas_punto[1] < -1e-8:
                continue
                
            punto_cumple_todas_las_restricciones = True
            for indice_restriccion in range(len(vector_b)):
                valor_evaluado = np.dot(matriz_a[indice_restriccion], coordenadas_punto)
                if tipos_inecuacion[indice_restriccion] == '<=' and valor_evaluado > vector_b[indice_restriccion] + 1e-8:
                    punto_cumple_todas_las_restricciones = False
                    break
                elif tipos_inecuacion[indice_restriccion] == '>=' and valor_evaluado < vector_b[indice_restriccion] - 1e-8:
                    punto_cumple_todas_las_restricciones = False
                    break
                elif tipos_inecuacion[indice_restriccion] == '=' and abs(valor_evaluado - vector_b[indice_restriccion]) > 1e-8:
                    punto_cumple_todas_las_restricciones = False
                    break
            
            if punto_cumple_todas_las_restricciones:
                vertices_factibles_temporales.append(coordenadas_punto)
                
        # Eliminar duplicados
        if len(vertices_factibles_temporales) > 0:
            vertices_factibles_temporales = np.unique(np.round(vertices_factibles_temporales, 8), axis=0)
            
        self.vertices_validos_region_factible = vertices_factibles_temporales
        self._log(f"  Se filtraron y verificaron {len(self.vertices_validos_region_factible)} vértices netamente factibles en total.")

    def resolver_metodo_grafico(self, generar_grafica=False):
        self.logs = []
        self._log("\n" + "="*50)
        self._log(" INICIANDO ALGORITMO GRÁFICO")
        self._log("="*50)
        
        self._log("> Paso 1: Calculando intersecciones entre restricciones y ejes...")
        self._calcular_intersecciones_sistemas_2x2()
        self._log(f"\n> Paso 2: Filtrando puntos atrapados en la región factible...")
        self._filtrar_puntos_fuera_de_region_factible()
        
        if len(self.vertices_validos_region_factible) == 0:
            self._log(">>> ERROR: No se encontró una región factible. <<<")
            self.estado_solucion = "Infactible"
            return None, None
            
        self._log("\n> Paso 3: Evaluando Función Objetivo (Z) en los vértices factibles...")
        # Evaluar F.O.
        vector_z = self.problema.coeficientes_funcion_objetivo
        mejor_valor_z = -float('inf') if self.problema.tipo_objetivo == 'max' else float('inf')
        mejores_coordenadas = None
        
        for coordenadas_punto in self.vertices_validos_region_factible:
            valor_z_actual = np.dot(vector_z, coordenadas_punto)
            self._log(f"  Vértice ({coordenadas_punto[0]:.2f}, {coordenadas_punto[1]:.2f}) -> Z = {valor_z_actual:.2f}")
            if self.problema.tipo_objetivo == 'max' and valor_z_actual > mejor_valor_z:
                mejor_valor_z = valor_z_actual
                mejores_coordenadas = coordenadas_punto
            elif self.problema.tipo_objetivo == 'min' and valor_z_actual < mejor_valor_z:
                mejor_valor_z = valor_z_actual
                mejores_coordenadas = coordenadas_punto
                
        self.solucion_optima_variables = mejores_coordenadas
        self.valor_optimo_z = mejor_valor_z
        self.estado_solucion = "Óptimo"
        
        self._log("\n" + "*"*50)
        self._log(">>> ¡Solución Gráfica Óptima Alcanzada! <<<")
        self._log(f"    Vértice Óptimo : ({self.solucion_optima_variables[0]:.2f}, {self.solucion_optima_variables[1]:.2f})")
        self._log(f"    Z             : {self.valor_optimo_z:.2f}")
        self._log("*"*50 + "\n")
        
        if generar_grafica:
            self.dibujar_solucion_grafica()
            
        return self.solucion_optima_variables, self.valor_optimo_z

    def dibujar_solucion_grafica(self):
        if self.estado_solucion != "Óptimo":
            self._log("No se puede graficar porque el problema no es óptimo/no factible.")
            return None
            
        figura, ejes = plt.subplots(figsize=(10, 8))
        
        # Encontrar los límites para graficar
        maximo_x = max([v[0] for v in self.todas_las_intersecciones_calculadas if v[0] >= 0]) * 1.5 if self.todas_las_intersecciones_calculadas else 10
        valores_recta_x = np.linspace(0, maximo_x, 400)
        
        matriz_a = self.problema.matriz_tecnologica_restricciones
        vector_b = self.problema.vector_lado_derecho_restricciones
        
        for indice_restriccion in range(len(vector_b)):
            signo = self.problema.tipos_inecuacion_restricciones[indice_restriccion]
            if matriz_a[indice_restriccion][1] != 0:
                valores_recta_y = (vector_b[indice_restriccion] - matriz_a[indice_restriccion][0] * valores_recta_x) / matriz_a[indice_restriccion][1]
                ejes.plot(valores_recta_x, valores_recta_y, label=f'R{indice_restriccion+1}: {matriz_a[indice_restriccion][0]}x1 + {matriz_a[indice_restriccion][1]}x2 {signo} {vector_b[indice_restriccion]}', linestyle='--')
            else:
                ejes.axvline(x=vector_b[indice_restriccion]/matriz_a[indice_restriccion][0], label=f'R{indice_restriccion+1}: {matriz_a[indice_restriccion][0]}x1 {signo} {vector_b[indice_restriccion]}', linestyle='--')
        # Ordenar vértices conveccionales para pintar el polígono
        if len(self.vertices_validos_region_factible) > 2:
            centro_region = np.mean(self.vertices_validos_region_factible, axis=0)
            angulos = np.arctan2(self.vertices_validos_region_factible[:, 1] - centro_region[1], self.vertices_validos_region_factible[:, 0] - centro_region[0])
            indices_ordenados = np.argsort(angulos)
            vertices_ordenados_convexa = self.vertices_validos_region_factible[indices_ordenados]
            
            ejes.fill(vertices_ordenados_convexa[:, 0], vertices_ordenados_convexa[:, 1], color='gray', alpha=0.3, label='Región Factible')

        # Graficar puntos factibles
        for punto_vertice in self.vertices_validos_region_factible:
            ejes.plot(punto_vertice[0], punto_vertice[1], 'ko')
            
        # Graficar el punto óptimo
        ejes.plot(self.solucion_optima_variables[0], self.solucion_optima_variables[1], 'r*', markersize=15, label=f'Óptimo ({self.solucion_optima_variables[0]:.2f}, {self.solucion_optima_variables[1]:.2f})')
        
        # Graficar recta de isocosto
        vector_z = self.problema.coeficientes_funcion_objetivo
        if vector_z[1] != 0:
            valores_recta_y_isocosto = (self.valor_optimo_z - vector_z[0] * valores_recta_x) / vector_z[1]
            ejes.plot(valores_recta_x, valores_recta_y_isocosto, color='red', label=f'Isocosto Z = {self.valor_optimo_z:.2f}')
            
        ejes.set_xlim(0, maximo_x)
        ejes.set_ylim(0, maximo_x)
        ejes.set_xlabel('X1')
        ejes.set_ylabel('X2')
        ejes.set_title('Método Gráfico para Optimización Lineal')
        ejes.legend()
        ejes.grid(True)
        return figura

import streamlit as st
import numpy as np
from core.model import ProblemaLineal
from solvers.simplex import SolucionadorSimplex
from visualization.graphic import SolucionadorGrafico

st.set_page_config(page_title="Optimizador Lineal", layout="wide")

st.title("LinealOpt - Solver Interactivo 🚀")

if 'cantidad_variables' not in st.session_state:
    st.session_state.cantidad_variables = 2
if 'cantidad_restricciones' not in st.session_state:
    st.session_state.cantidad_restricciones = 3

def agregar_restriccion_ui():
    st.session_state.cantidad_restricciones += 1

def remover_restriccion_ui():
    if st.session_state.cantidad_restricciones > 1:
        st.session_state.cantidad_restricciones -= 1

columna_inputs, columna_visualizacion = st.columns([1, 1.2])

with columna_inputs:
    st.header("1. Definición del Problema")
    
    # Parámetros básicos
    col_tipo, col_cant_vars = st.columns(2)
    with col_tipo:
        tipo_optimizacion = st.selectbox("Optimizar:", ["Maximizar", "Minimizar"])
    with col_cant_vars:
        st.session_state.cantidad_variables = st.number_input("Cant. Variables (X)", min_value=1, max_value=10, value=st.session_state.cantidad_variables)
        
    cantidad_actual_variables = st.session_state.cantidad_variables
    
    st.subheader("Función Objetivo (Z)")
    columnas_variables_objetivo = st.columns(cantidad_actual_variables)
    lista_coeficientes_z = []
    for indice_variable in range(cantidad_actual_variables):
        with columnas_variables_objetivo[indice_variable]:
            valor_coeficiente = st.number_input(f"X{indice_variable+1}", value=3.0 if indice_variable==0 else 2.0 if indice_variable==1 else 1.0, key=f"obj_{indice_variable}")
            lista_coeficientes_z.append(valor_coeficiente)
            
    st.subheader("Restricciones")
    
    # Valores de relleno automático para el ejemplo estándar (para 2 variables, 3 restricciones)
    matriz_demo_a = [[2.0, 1.0], [2.0, 3.0], [3.0, 1.0]]
    vector_demo_b = [18.0, 42.0, 24.0]
    
    lista_matriz_coeficientes_a = []
    lista_vector_b = []
    lista_signos = []
    
    for indice_restriccion in range(st.session_state.cantidad_restricciones):
        st.markdown(f"**Restricción {indice_restriccion+1}**")
        columnas_restriccion = st.columns(cantidad_actual_variables + 2)
        
        fila_coeficientes_a = []
        for indice_variable in range(cantidad_actual_variables):
            with columnas_restriccion[indice_variable]:
                valor_por_defecto_a = matriz_demo_a[indice_restriccion][indice_variable] if indice_restriccion < len(matriz_demo_a) and indice_variable < len(matriz_demo_a[indice_restriccion]) else 1.0
                valor_coef_a = st.number_input(f"X{indice_variable+1}##{indice_restriccion}", value=float(valor_por_defecto_a), key=f"a_{indice_restriccion}_{indice_variable}", label_visibility="collapsed")
                fila_coeficientes_a.append(valor_coef_a)
        
        with columnas_restriccion[-2]:
            signo = st.selectbox("Signo", ["<=", ">=", "="], key=f"sign_{indice_restriccion}", label_visibility="collapsed")
            
        with columnas_restriccion[-1]:
            valor_por_defecto_b = vector_demo_b[indice_restriccion] if indice_restriccion < len(vector_demo_b) else 10.0
            valor_ld = st.number_input("LD", value=float(valor_por_defecto_b), key=f"b_{indice_restriccion}", label_visibility="collapsed")
        
        lista_matriz_coeficientes_a.append(fila_coeficientes_a)
        lista_vector_b.append(valor_ld)
        lista_signos.append(signo)
        
    boton_col_1, boton_col_2 = st.columns(2)
    with boton_col_1:
        st.button("➕ Añadir restricción", on_click=agregar_restriccion_ui)
    with boton_col_2:
        st.button("➖ Quitar restricción", on_click=remover_restriccion_ui)
        
    st.markdown("---")
    boton_calcular_presionado = st.button("🚀 Calcular Óptimos", type="primary", use_container_width=True)

with columna_visualizacion:
    st.header("2. Resultados y Visualización")
    
    if not boton_calcular_presionado:
        st.info("Presiona **Calcular Óptimos** en el panel izquierdo para ejecutar el Simplex y renderizar el gráfico.")
    else:
        # 1. Construir Problema Matemático
        codigo_tipo_problema = 'max' if tipo_optimizacion == "Maximizar" else 'min'
        problema_lineal = ProblemaLineal(codigo_tipo_problema)
        problema_lineal.definir_funcion_objetivo(lista_coeficientes_z)
        
        for indice_restriccion in range(st.session_state.cantidad_restricciones):
            problema_lineal.agregar_restriccion(lista_matriz_coeficientes_a[indice_restriccion], lista_vector_b[indice_restriccion], tipo_inecuacion=lista_signos[indice_restriccion])
            
        # 2. Ejecutar Simplex (Desacoplado)
        solucionador_simplex = SolucionadorSimplex(problema_lineal)
        solucionador_simplex.ejecutar_algoritmo_simplex()
        
        st.subheader("💡 Solución Analítica (Simplex)")
        if solucionador_simplex.estado_solucion == "Óptimo":
            st.success(f"**Estado:** {solucionador_simplex.estado_solucion} | **Z Óptimo:** {solucionador_simplex.valor_optimo_z:.4f}")
            columnas_resultados = st.columns(cantidad_actual_variables)
            for indice_variable, valor_variable in enumerate(solucionador_simplex.solucion_optima_variables[:cantidad_actual_variables]):
                columnas_resultados[indice_variable].metric(label=f"Variable X{indice_variable+1}", value=f"{valor_variable:.4f}")
        else:
            st.error(f"El Simplex falló, es no-acotado o es infactible. Estado: {solucionador_simplex.estado_solucion}")
            
        with st.expander("🖥️ Consola de Logs del Simplex (Paso a paso)", expanded=False):
            st.code("\n".join(solucionador_simplex.logs), language="plaintext")

        st.markdown("---")
        st.subheader("📈 Método Gráfico")
        
        # 3. Ejecutar Método Gráfico si hay solo 2 variables
        if cantidad_actual_variables == 2:
            solucionador_grafico = SolucionadorGrafico(problema_lineal)
            solucionador_grafico.resolver_metodo_grafico(generar_grafica=False)
            
            if solucionador_grafico.estado_solucion == "Óptimo":
                # Rescatar la figura para la GUI sin bloquear hilos.
                figura_generada = solucionador_grafico.dibujar_solucion_grafica()
                if figura_generada:
                    st.pyplot(figura_generada)
                
                with st.expander("👁️ Logs Matemáticos del Método Gráfico", expanded=False):
                    st.code("\n".join(solucionador_grafico.logs), language="plaintext")
            else:
                st.error("El problema es infactible o no acotado geométricamente.")
        else:
            st.warning("⚠️ El método gráfico solo puede graficar planos de **exactamente** 2 variables.")

import streamlit as st
import numpy as np
from core.model import ProblemaLineal
from solvers.simplex import SolucionadorSimplex

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
            valor_coeficiente = st.number_input(f"X{indice_variable+1}", value=0.0, key=f"obj_{indice_variable}")
            lista_coeficientes_z.append(valor_coeficiente)
    st.subheader("Restricciones")
    lista_matriz_coeficientes_a = []
    lista_vector_b = []
    lista_signos = []
    for indice_restriccion in range(st.session_state.cantidad_restricciones):
        st.markdown(f"**Restricción {indice_restriccion+1}**")
        columnas_restriccion = st.columns(cantidad_actual_variables + 2)
        fila_coeficientes_a = []
        for indice_variable in range(cantidad_actual_variables):
            with columnas_restriccion[indice_variable]:
                valor_coef_a = st.number_input(f"X{indice_variable+1}##{indice_restriccion}", value=0.0, key=f"a_{indice_restriccion}_{indice_variable}", label_visibility="collapsed")
                fila_coeficientes_a.append(valor_coef_a)
        with columnas_restriccion[-2]:
            signo = st.selectbox("Signo", ["<=", ">=", "="], key=f"sign_{indice_restriccion}", label_visibility="collapsed")
        with columnas_restriccion[-1]:
            valor_ld = st.number_input("LD", value=0.0, key=f"b_{indice_restriccion}", label_visibility="collapsed")
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
        st.info("Presiona **Calcular Óptimos** en el panel izquierdo para ejecutar el Simplex.")
    else:
        codigo_tipo_problema = 'max' if tipo_optimizacion == "Maximizar" else 'min'
        problema_lineal = ProblemaLineal(codigo_tipo_problema)
        problema_lineal.definir_funcion_objetivo(lista_coeficientes_z)
        for indice_restriccion in range(st.session_state.cantidad_restricciones):
            problema_lineal.agregar_restriccion(lista_matriz_coeficientes_a[indice_restriccion], lista_vector_b[indice_restriccion], tipo_inecuacion=lista_signos[indice_restriccion])
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
        with st.expander("🖥️ Consola de Logs del Simplex (Paso a paso)", expanded=True):
            st.code("\n".join(solucionador_simplex.logs), language="plaintext")

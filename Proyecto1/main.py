import sys
import textwrap
from core.model import ProblemaLineal
from core.utils import build_problem_from_console, build_problem_from_json
from solvers.simplex import SolucionadorSimplex
from visualization.graphic import SolucionadorGrafico

def p_demo():
    # Construcción del problema predeterminado como demostración
    problema = ProblemaLineal(tipo_objetivo='max')
    problema.definir_funcion_objetivo([3.0, 2.0])
    problema.agregar_restriccion([2.0, 1.0], 18.0)
    problema.agregar_restriccion([2.0, 3.0], 42.0)
    problema.agregar_restriccion([3.0, 1.0], 24.0)
    return problema

def solve_and_show(problema):
    print("\n--- [1] RESOLVIENDO CON MÉTODO SIMPLEX ---")
    simplex = SolucionadorSimplex(problema)
    sol_simplex, z_simplex = simplex.ejecutar_algoritmo_simplex()
    
    print("\n".join(simplex.logs))
    print(f"Estado Simplex: {simplex.estado_solucion}")

    print("\n--- [2] RESOLVIENDO CON MÉTODO GRÁFICO ---")
    try:
        graphic = SolucionadorGrafico(problema)
        sol_graph, z_graph = graphic.resolver_metodo_grafico(generar_grafica=False)
        print("\n".join(graphic.logs))
        print(f"Estado Gráfico: {graphic.estado_solucion}")
        
    except ValueError as e:
        print(f"\n[Aviso Gráfico]: {e} (Recuerda que solo grafica 2 variables)")

def main():
    while True:
        print("\n" + "=" * 60)
        print(" HERRAMIENTA DE OPTIMIZACIÓN LINEAL (Versión CMD)")
        print(" Nota: Si buscas la Interfaz Gráfica, corre 'streamlit run app.py'")
        print("=" * 60)
        print("SELECCIONE UNA OPCIÓN:")
        print("  [1] Ingresar datos interactivamente")
        print("  [2] Cargar JSON")
        print("  [3] Ejecutar problema por defecto")
        print("  [0] Salir")
        
        op = input("\nIngrese su opción: ").strip()
        
        problema = None
        if op == "1":
            problema = build_problem_from_console()
        elif op == "2":
            filepath = input("Ingrese la ruta del archivo JSON (ej. problem.json): ").strip()
            try:
                problema = build_problem_from_json(filepath)
            except Exception as e:
                print(f"\nError al cargar JSON: {e}")
                continue
        elif op == "3":
            print(textwrap.dedent("""
                Maximizar Z = 3X1 + 2X2
                Sujeto a:
                  2X1 + 1X2 <= 18
                  2X1 + 3X2 <= 42
                  3X1 + 1X2 <= 24
                  X1, X2 >= 0
            """))
            problema = p_demo()
        elif op == "0":
            print("Saliendo...")
            sys.exit(0)
        else:
            continue
            
        if problema:
            solve_and_show(problema)
            input("\n--- Presione ENTER ---")

if __name__ == '__main__':
    main()

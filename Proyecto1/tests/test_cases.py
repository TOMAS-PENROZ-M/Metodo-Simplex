import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import ProblemaLineal
from solvers.simplex import SolucionadorSimplex
from visualization.graphic import SolucionadorGrafico

class TestLinearOptimization(unittest.TestCase):

    def setUp(self):
        # Problema de prueba estándar (Maximización 2D)
        # Max Z = 3X1 + 2X2
        # s.a:
        #   2X1 + X2 <= 18
        #   2X1 + 3X2 <= 42
        #   3X1 + X2 <= 24
        
        self.problema = ProblemaLineal(tipo_objetivo='max')
        self.problema.definir_funcion_objetivo([3.0, 2.0])
        self.problema.agregar_restriccion([2.0, 1.0], 18.0)
        self.problema.agregar_restriccion([2.0, 3.0], 42.0)
        self.problema.agregar_restriccion([3.0, 1.0], 24.0)

    def test_simplex_solver(self):
        solucionador = SolucionadorSimplex(self.problema)
        solucion, z = solucionador.ejecutar_algoritmo_simplex()
        
        self.assertEqual(solucionador.estado_solucion, "Óptimo")
        self.assertAlmostEqual(z, 33.0, places=4)
        self.assertAlmostEqual(solucion[0], 3.0, places=4)  # X1 = 3
        self.assertAlmostEqual(solucion[1], 12.0, places=4) # X2 = 12
        
        # Holguras H1=0, H2=0, H3=3
        self.assertAlmostEqual(solucion[2], 0.0, places=4)
        self.assertAlmostEqual(solucion[3], 0.0, places=4)
        self.assertAlmostEqual(solucion[4], 3.0, places=4)

    def test_graphic_solver(self):
        solucionador = SolucionadorGrafico(self.problema)
        solucion, z = solucionador.resolver_metodo_grafico(generar_grafica=False)
        
        self.assertEqual(solucionador.estado_solucion, "Óptimo")
        self.assertAlmostEqual(z, 33.0, places=4)
        self.assertAlmostEqual(solucion[0], 3.0, places=4)
        self.assertAlmostEqual(solucion[1], 12.0, places=4)

if __name__ == '__main__':
    unittest.main()

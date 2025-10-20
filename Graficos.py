import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

def min_eliminaciones_palindromos_tabular(a):
    n = len(a)
    if n == 0:
        return 0
    dp = [[0]*n for _ in range(n)]

    # Subarreglos de longitud 1
    for i in range(n):
        dp[i][i] = 1

    # Longitudes crecientes
    for length in range(2, n+1):
        for l in range(0, n-length+1):
            r = l + length - 1
            res = 1 + dp[l+1][r]

            # Caso palindrómico directo
            if a[l] == a[r]:
                if l+1 <= r-1:
                    res = min(res, dp[l+1][r-1])
                else:
                    res = 1

            # División del subarreglo en dos partes
            for k in range(l, r):
                res = min(res, dp[l][k] + dp[k+1][r])

            dp[l][r] = res
    return dp[0][n-1]


# ==========================================================
# 2. VALIDACIÓN FUNCIONAL (comprobación de correctitud)
# ==========================================================
def validar_correctitud():
    casos = [
        # (entrada, valor_esperado)
        ([1, 4, 4, 2, 3, 2, 1], 2),
        ([1, 2, 3], 3),
        ([1, 2, 1], 1),
        ([2, 4, 3, 3, 4, 2], 1),
        ([2, 1, 3, 3, 2], 2)
    ]

    print("\n=== Validación de correctitud del algoritmo ===")
    for arr, esperado in casos:
        obtenido = min_eliminaciones_palindromos_tabular(arr)
        estado = "Correcto" if esperado == obtenido else " Incorrecto"
        print(f"Entrada: {arr} -> Esperado: {esperado}, Obtenido: {obtenido} --> {estado}")


# ==========================================================
# 3. GENERADORES DE CASOS DE PRUEBA
# ==========================================================
def generar_caso(tipo, n):
    if tipo == "palindromo":
        base = [2, 4, 3, 3, 4, 2]
        return (base * (n // len(base) + 1))[:n]
    elif tipo == "no_palindromo":
        return list(range(1, n+1))
    elif tipo == "mixto":
        base = [2, 1, 3, 3, 2]
        return (base * (n // len(base) + 1))[:n]
    else:
        raise ValueError("tipo desconocido")


# ==========================================================
# 4. EXPERIMENTACIÓN (tiempo vs tamaño)
# ==========================================================
def ejecutar_experimentos():
    all_sizes = [10, 50, 100, 200, 300, 400, 500]
    max_run_n = 500
    repeats = 3
    tipos = ["palindromo", "no_palindromo", "mixto"]

    results = {t: [] for t in tipos}

    print("\n=== Experimentos de rendimiento ===")
    for tipo in tipos:
        print(f"\n--- Caso: {tipo} ---")
        for n in all_sizes:
            if n > max_run_n:
                results[tipo].append(np.nan)
                continue

            tiempos = []
            for _ in range(repeats):
                arr = generar_caso(tipo, n)
                start = time.perf_counter()
                _ = min_eliminaciones_palindromos_tabular(arr)
                end = time.perf_counter()
                tiempos.append(end - start)

            t_avg = mean(tiempos)
            results[tipo].append(t_avg)
            print(f"n={n:5d} -> tiempo promedio = {t_avg:.5f} s")

    return all_sizes, results


# ==========================================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================================
def graficar_resultados(all_sizes, results):
    colors = {'palindromo': 'tab:blue', 'no_palindromo': 'tab:orange', 'mixto': 'tab:green'}
    markers = {'palindromo': 'o', 'no_palindromo': 's', 'mixto': '^'}

    plt.figure(figsize=(10, 7))
    all_x, all_y = [], []

    for tipo in results.keys():
        x_n = [n for i, n in enumerate(all_sizes) if not np.isnan(results[tipo][i])]
        y_t = [results[tipo][i] for i, n in enumerate(all_sizes) if not np.isnan(results[tipo][i])]

        # Ajuste lineal con respecto a n^3
        X = (np.array(x_n) ** 3).reshape(-1, 1)
        y = np.array(y_t)
        k, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        k = k[0]
        print(f"{tipo}: k ~ {k:.3e}")

        plt.scatter(x_n, y_t, marker=markers[tipo], color=colors[tipo], label=f"{tipo} datos")
        n_plot = np.linspace(min(x_n), max(x_n), 100)
        t_fit = k * (n_plot ** 3)
        plt.plot(n_plot, t_fit, '-', color=colors[tipo])

        all_x.extend(x_n)
        all_y.extend(y_t)

    # Ajuste global O(n³)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    X_global = (all_x ** 3).reshape(-1, 1)
    k_global, _, _, _ = np.linalg.lstsq(X_global, all_y, rcond=None)
    k_global = k_global[0]
    print(f"Ajuste global: k ~ {k_global:.3e}")

    n_plot_global = np.linspace(min(all_x), max(all_x), 100)
    t_fit_global = k_global * (n_plot_global ** 3)
    plt.plot(n_plot_global, t_fit_global, 'k--', linewidth=2, label="Ajuste global $O(n^3)$")

    plt.xlabel("Tamaño n")
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de tiempos y ajuste cúbico global")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 6. PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    # Paso 1: Validar correctitud del algoritmo
    validar_correctitud()

    # Paso 2: Ejecutar los experimentos de rendimiento
    all_sizes, results = ejecutar_experimentos()

    # Paso 3: Visualizar resultados
    graficar_resultados(all_sizes, results)

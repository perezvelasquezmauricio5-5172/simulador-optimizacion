import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Any

# ==========================================================
# 1. CONFIGURACIÓN GLOBAL DE STREAMLIT & ESTILOS VISUALES
# ==========================================================
st.set_page_config(
    page_title="Laboratorio de Optimización Numérica",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo oscuro integral con soporte para Matplotlib
plt.style.use('dark_background')

# ==========================================================
# 2. MODELO DE DATOS Y POBLACIÓN DE ESCENARIOS
# ==========================================================
@dataclass
class EscenarioTopografico:
    f: Callable[[np.ndarray], float]
    g: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    limites: Tuple[float, float, float, float]
    optimo: Tuple[float, float]
    tipo: str
    descripcion: str
    maximizar: bool

@st.cache_resource
def cargar_base_conocimiento() -> Dict[str, EscenarioTopografico]:
    """Carga y almacena en caché la definición matemática de las superficies."""
    return {
        "1. Cuenco convexo": EscenarioTopografico(
            f=lambda x: (x[0] - 2)**2 + 2*(x[1] - 1)**2,
            g=lambda x: np.array([2*(x[0] - 2), 4*(x[1] - 1)], dtype=float),
            h=lambda x: np.array([[2., 0.], [0., 4.]]),
            limites=(-2., 6., -2., 4.),
            optimo=(2., 1.),
            tipo="Mínimo global",
            descripcion="Función convexa con mínimo global en (2, 1). Ideal para comparar velocidad pura de algoritmos.",
            maximizar=False
        ),
        "2. Función del libro": EscenarioTopografico(
            f=lambda x: 2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2,
            g=lambda x: np.array([2*x[1] + 2 - 2*x[0], 2*x[0] - 4*x[1]], dtype=float),
            h=lambda x: np.array([[-2., 2.], [2., -4.]]),
            limites=(-3., 5., -3., 3.),
            optimo=(2., 1.),
            tipo="Máximo global",
            descripcion="La función original del libro tiene un máximo en (2, 1). El motor aplica inversión de gradiente automáticamente.",
            maximizar=True
        ),
        "3. Elipse alargada (Cresta)": EscenarioTopografico(
            f=lambda x: x[0]**2 + 10*x[1]**2,
            g=lambda x: np.array([2*x[0], 20*x[1]], dtype=float),
            h=lambda x: np.array([[2., 0.], [0., 20.]]),
            limites=(-4., 4., -2., 2.),
            optimo=(0., 0.),
            tipo="Mínimo global",
            descripcion="Diferencia de curvatura severa entre ejes X e Y. Permite evaluar el patrón de zigzag en primera mano.",
            maximizar=False
        ),
        "4. Punto silla": EscenarioTopografico(
            f=lambda x: x[0]**2 - x[1]**2,
            g=lambda x: np.array([2*x[0], -2*x[1]], dtype=float),
            h=lambda x: np.array([[2., 0.], [0., -2.]]),
            limites=(-4., 4., -4., 4.),
            optimo=(0., 0.),
            tipo="Punto silla",
            descripcion="El gradiente se anula en el origen pero la curvatura es indefinida: la función crece en una dirección y decrece en otra.",
            maximizar=False
        )
    }

# ==========================================================
# 3. MOTOR MATEMÁTICO & ALGORITMOS DE OPTIMIZACIÓN
# ==========================================================
class MotorOptimizacion:
    
    @staticmethod
    def clasificar_hessiano(H: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, str]:
        valores = np.linalg.eigvalsh(H)
        if np.all(valores > tol):
            texto = "Definido positivo: curvatura de mínimo local."
        elif np.all(valores < -tol):
            texto = "Definido negativo: curvatura de máximo local."
        elif np.any(valores > tol) and np.any(valores < -tol):
            texto = "Indefinido: compatible con un punto silla."
        else:
            texto = "Semidefinido o degenerado: se requiere más análisis."
        return valores, texto

    @staticmethod
    def busqueda_armijo(f: Callable, x: np.ndarray, p: np.ndarray, g: np.ndarray, alpha_inicial: float = 1.0) -> float:
        alpha = alpha_inicial
        c, rho = 1e-4, 0.5
        fx = f(x)
        pendiente = float(np.dot(g, p))

        if not np.isfinite(pendiente) or pendiente >= 0:
            return 0.0

        for _ in range(40):
            xn = x + alpha * p
            fn = f(xn)
            if np.isfinite(fn) and fn <= fx + c * alpha * pendiente:
                return alpha
            alpha *= rho
        return 0.0

    @classmethod
    def ejecutar(cls, f: Callable, grad: Callable, hess: Callable, x0: List[float], 
                 metodo: str, max_iter: int, tol: float, paso_fijo: float = 0.05, 
                 busqueda: bool = True) -> Tuple[np.ndarray, pd.DataFrame, str, np.ndarray]:
        
        x = np.array(x0, dtype=float)
        puntos = [x.copy()]
        historial = []
        estado = "Se alcanzó el límite de iteraciones."
        
        B = np.eye(2) # Aproximación inicial del Hessiano inverso para BFGS
        g_anterior, p_anterior = None, None

        for k in range(max_iter + 1):
            fx = float(f(x))
            g = np.asarray(grad(x), dtype=float)
            ng = float(np.linalg.norm(g))

            historial.append({
                "Iteracion": k,
                "x": x[0],
                "y": x[1],
                "f(x,y)": fx,
                "Norma gradiente": ng,
            })

            if not np.isfinite(fx) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)):
                estado = "Cálculo no finito: el algoritmo se detuvo."
                break

            if ng < tol:
                estado = "Gradiente pequeño: punto crítico aproximado."
                break

            if k == max_iter:
                break

            # Cálculo de la dirección de descenso
            if metodo == "Descenso del gradiente":
                p = -g
            elif metodo == "Gradiente conjugado":
                if g_anterior is None:
                    p = -g
                else:
                    denominador = float(np.dot(g_anterior, g_anterior))
                    beta = max(0.0, float(np.dot(g, g - g_anterior)) / denominador) if denominador > 1e-20 else 0.0
                    p = -g + beta * p_anterior
                    if float(np.dot(p, g)) >= 0:
                        p = -g
            elif metodo == "Newton":
                H = np.asarray(hess(x), dtype=float)
                try:
                    p = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    p = -g
                if not np.all(np.isfinite(p)) or float(np.dot(p, g)) >= 0:
                    p = -g
            elif metodo == "BFGS (cuasi-Newton)":
                p = -B @ g
                if not np.all(np.isfinite(p)) or float(np.dot(p, g)) >= 0:
                    B = np.eye(2)
                    p = -g
            else:
                p = -g

            # Tamaño de paso
            alpha = cls.busqueda_armijo(f, x, p, g) if busqueda else paso_fijo

            if alpha <= 1e-14:
                estado = "No se encontró un paso útil; posible estancamiento."
                break

            xn = x + alpha * p

            if not np.all(np.isfinite(xn)) or np.linalg.norm(xn) > 1e8:
                estado = "El método se volvió inestable y se detuvo."
                break

            gn = np.asarray(grad(xn), dtype=float)

            if metodo == "BFGS (cuasi-Newton)":
                s = xn - x
                y = gn - g
                ys = float(np.dot(y, s))
                if ys > 1e-12:
                    rho = 1.0 / ys
                    I = np.eye(2)
                    V = I - rho * np.outer(s, y)
                    B = V @ B @ V.T + rho * np.outer(s, s)
                else:
                    B = np.eye(2)

            g_anterior = g.copy()
            p_anterior = p.copy()
            x = xn
            puntos.append(x.copy())

        return np.asarray(puntos), pd.DataFrame(historial), estado, x

# ==========================================================
# 4. INTERFAZ DE USUARIO & COMPONENTES DE VISUALIZACIÓN
# ==========================================================
def main():
    st.title("📈 Laboratorio Interactivo de Optimización Multivariable")
    st.markdown("##### Capítulo 14.2: Gradientes, Hessianos, Máxima Inclinación y Métodos Cuasi-Newton")

    base_escenarios = cargar_base_conocimiento()

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        nombre_escenario = st.selectbox("Selecciona una función:", list(base_escenarios.keys()))
        escenario = base_escenarios[nombre_escenario]

        metodo = st.selectbox(
            "Algoritmo Numérico:",
            ["Descenso del gradiente", "Gradiente conjugado", "Newton", "BFGS (cuasi-Newton)"]
        )

        xmin, xmax, ymin, ymax = escenario.limites

        st.markdown("---")
        st.markdown("**Vector de Posición Inicial (X₀)**")
        x0 = st.slider("Coordenada X", float(xmin), float(xmax), float(np.clip(xmin + 0.8, xmin, xmax)))
        y0 = st.slider("Coordenada Y", float(ymin), float(ymax), float(np.clip(ymin + 0.7, ymin, ymax)))

        st.markdown("---")
        max_iter = st.slider("Máximo de iteraciones", 5, 200, 60, 5)
        tol = st.select_slider(
            "Tolerancia del gradiente",
            options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            value=1e-5,
            format_func=lambda v: f"{v:.0e}"
        )

        busqueda = st.toggle("Búsqueda lineal automática (Armijo)", value=True)
        paso_fijo = st.slider("Paso fijo (α)", 0.001, 0.5, 0.05, 0.001, disabled=busqueda)

    # Adaptación dinámica si la función original es de maximización
    f_orig, g_orig, h_orig = escenario.f, escenario.g, escenario.h
    if escenario.maximizar:
        f = lambda x: -f_orig(x)
        grad = lambda x: -g_orig(x)
        hess = lambda x: -h_orig(x)
        st.info("💡 **Nota:** Esta función busca un máximo global. El motor optimiza $-f(x,y)$ mediante minimización.")
    else:
        f, grad, hess = f_orig, g_orig, h_orig

    st.caption(escenario.descripcion)

    # Cálculo principal
    puntos, historial, estado, xf = MotorOptimizacion.ejecutar(
        f, grad, hess, [x0, y0], metodo, max_iter, tol, paso_fijo, busqueda
    )

    # --- PESTAÑAS DE NAVEGACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌌 Simulador Interactivo",
        "🧮 Análisis de Gradiente y Hessiano",
        "📊 Comparativa de Algoritmos",
        "📝 Cuestionario Evaluativo"
    ])

    # ---------------- TAB 1: SIMULADOR ----------------
    with tab1:
        col_graf, col_res = st.columns([1.8, 1])

        with col_graf:
            xg, yg = np.meshgrid(np.linspace(xmin, xmax, 120), np.linspace(ymin, ymax, 120))
            zg = np.vectorize(lambda a, b: f(np.array([a, b])))(xg, yg)

            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")

            niveles = np.linspace(np.nanmin(zg), np.nanmax(zg), 25)
            curvas = ax.contour(xg, yg, zg, levels=niveles, cmap="viridis", linewidths=0.9, alpha=0.8)
            ax.clabel(curvas, inline=True, fontsize=7, colors="#A0A0A0")

            # Dibujo de trayectorias con colores de alto contraste neón
            ax.plot(puntos[:, 0], puntos[:, 1], "-o", color="#FFC107", linewidth=2.2, markersize=4, label="Ruta del algoritmo")
            ax.scatter([x0], [y0], s=100, color="#00BCD4", edgecolor="white", label="Punto Inicial X₀", zorder=5)
            ax.scatter([xf[0]], [xf[1]], s=120, marker="X", color="#FF5722", edgecolor="white", label="Punto Final Alcanzado", zorder=6)

            ax.set_xlabel("Eje X")
            ax.set_ylabel("Eje Y")
            ax.set_title("Superficie Topográfica y Trayectoria Numérica")
            ax.legend(facecolor="#171A21", labelcolor="white", edgecolor="#333333")
            ax.grid(alpha=0.15)

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with col_res:
            st.markdown("### Telemetría")
            st.metric("Iteraciones Computadas", len(historial) - 1)
            st.metric("Coordenada X Final", f"{xf[0]:.6f}")
            st.metric("Coordenada Y Final", f"{xf[1]:.6f}")
            st.metric("Valor f(x,y) Evaluación", f"{f_orig(xf):.6f}")
            st.metric("Norma Gradiente ||∇f||", f"{np.linalg.norm(g_orig(xf)):.3e}")

            st.markdown("**Estado del algoritmo:**")
            st.info(estado)

            _, clasif = MotorOptimizacion.clasificar_hessiano(h_orig(xf))
            st.markdown("**Matriz Hessiana en el punto final:**")
            st.caption(clasif)

        st.markdown("---")
        st.markdown("### Registros Históricos")
        st.dataframe(
            historial.style.format({
                "x": "{:.6f}",
                "y": "{:.6f}",
                "f(x,y)": "{:.8f}",
                "Norma gradiente": "{:.3e}",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "💾 Descargar reporte en CSV",
            historial.to_csv(index=False).encode("utf-8"),
            file_name="reporte_optimizacion.csv",
            mime="text/csv"
        )

    # ---------------- TAB 2: GRADIENTE Y HESSIANO ----------------
    with tab2:
        st.markdown("### 1. Vector Gradiente: Derivadas de Primer Orden")
        g0 = g_orig(np.array([x0, y0], dtype=float))
        
        st.latex(r"\nabla f(x,y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}")
        st.write(f"Evaluación exacta en el punto de origen **({x0:.3f}, {y0:.3f})**:")
        st.code(f"∇f = [{g0[0]:.6f}, {g0[1]:.6f}]\n||∇f|| = {np.linalg.norm(g0):.6f}")

        st.markdown("""
        * **Crecimiento:** El vector gradiente apunta directamente hacia la dirección donde la función crece más rápidamente.
        * **Descenso:** El vector opuesto ($-\nabla f$) marca la ruta de máxima inclinación de descenso.
        * **Ortogonalidad:** Es perpendicular a las curvas de nivel en cada punto.
        """)

        st.divider()
        st.markdown("### 2. Matriz Hessiana: Derivadas de Segundo Orden")
        H0 = h_orig(np.array([x0, y0], dtype=float))
        
        st.latex(rf"H(x,y) = \begin{{bmatrix}} {H0[0,0]:.4f} & {H0[0,1]:.4f} \\ {H0[1,0]:.4f} & {H0[1,1]:.4f} \end{bmatrix}")
        
        val, clas = MotorOptimizacion.clasificar_hessiano(H0)
        st.write(f"**Autovalores computados:** $\lambda_1 = {val[0]:.5f}, \lambda_2 = {val[1]:.5f}$")
        st.success(clas)

    # ---------------- TAB 3: COMPARATIVA ----------------
    with tab3:
        st.markdown("### Rendimiento de Algoritmos desde la misma posición X₀")
        
        nombres = ["Descenso del gradiente", "Gradiente conjugado", "Newton", "BFGS (cuasi-Newton)"]
        filas = []
        historico_comparativo = {}

        for n in nombres:
            pts_c, hist_c, est_c, xf_c = MotorOptimizacion.ejecutar(
                f, grad, hess, [x0, y0], n, max_iter, tol, paso_fijo, busqueda
            )
            historico_comparativo[n] = hist_c
            filas.append({
                "Método": n,
                "Iteraciones": len(hist_c) - 1,
                "X final": xf_c[0],
                "Y final": xf_c[1],
                "Valor Objetivo f(x,y)": f_orig(xf_c),
                "Norma ||∇f||": np.linalg.norm(g_orig(xf_c)),
                "Diagnóstico": est_c
            })

        st.dataframe(
            pd.DataFrame(filas).style.format({
                "X final": "{:.6f}",
                "Y final": "{:.6f}",
                "Valor Objetivo f(x,y)": "{:.8f}",
                "Norma ||∇f||": "{:.3e}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Gráfica de Convergencia
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#0E1117")

        for n, h_comp in historico_comparativo.items():
            ax2.semilogy(
                np.arange(len(h_comp)),
                np.maximum(h_comp["Norma gradiente"].to_numpy(), 1e-16),
                marker="o", markersize=3, label=n
            )

        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Norma del Gradiente ||∇f|| (Escala Logarítmica)")
        ax2.set_title("Curva de Velocidad de Convergencia")
        ax2.legend(facecolor="#171A21", labelcolor="white", edgecolor="#333333")
        ax2.grid(alpha=0.2)

        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    # ---------------- TAB 4: CUESTIONARIO ----------------
    with tab4:
        st.markdown("### Comprueba lo aprendido en la presentación")

        r1 = st.radio(
            "1. Si el gradiente vale exactamente cero (∇f = 0), ¿qué podemos afirmar?",
            ["Siempre es un mínimo global.", "Es un punto crítico, pero se requiere evaluar el Hessiano para clasificarlo.", "La función vale cero en ese punto.", "El algoritmo no ha comenzado."],
            key="p1"
        )

        r2 = st.radio(
            "2. ¿Qué información proporciona la Matriz Hessiana?",
            ["La curvatura local de la función mediante derivadas de segundo orden.", "Solamente el punto de partida.", "Asegura siempre convergencia en 1 paso.", "Solamente el número de variables."],
            key="p2"
        )

        r3 = st.radio(
            "3. ¿Cuál es la característica principal del método BFGS?",
            ["Construye y actualiza iterativamente una aproximación de la inversa de la matriz Hessiana.", "Únicamente evalúa la función sin usar derivadas.", "No requiere calcular gradientes.", "Siempre encuentra el óptimo global en cualquier función."],
            key="p3"
        )

        if st.button("Comprobar respuestas"):
            st.success("1. Correcto: Es un punto crítico (puede ser mínimo, máximo o punto silla).") if r1.startswith("Es un punto") else st.error("1. Incorrecto. Revisa el concepto de punto crítico.")
            st.success("2. Correcto: La matriz Hessiana contiene las derivadas de segundo orden y mide la curvatura.") if r2.startswith("La curvatura") else st.error("2. Incorrecto. Repasa las derivadas parciales de segundo orden.")
            st.success("3. Correcto: BFGS aproxima B ≈ H⁻¹ utilizando solo información de gradientes de pasos previos.") if r3.startswith("Construye") else st.error("3. Incorrecto. Revisa la definición de métodos Cuasi-Newton.")

if __name__ == "__main__":
    main()

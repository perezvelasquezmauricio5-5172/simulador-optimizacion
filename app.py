
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================================
# LABORATORIO INTERACTIVO DE OPTIMIZACION NUMERICA
# Capitulo 14.2
# Gradientes, Hessianos, maxima inclinacion y cuasi-Newton
# ==========================================================

st.set_page_config(
    page_title="Laboratorio de Optimizacion",
    page_icon="📈",
    layout="wide",
)

# ---------------- FUNCIONES DE PRUEBA ----------------

def f_cuenco(x):
    a, b = x
    return (a - 2)**2 + 2*(b - 1)**2

def g_cuenco(x):
    a, b = x
    return np.array([2*(a - 2), 4*(b - 1)], dtype=float)

def h_cuenco(x):
    return np.array([[2., 0.], [0., 4.]])

def f_libro(x):
    a, b = x
    return 2*a*b + 2*a - a**2 - 2*b**2

def g_libro(x):
    a, b = x
    return np.array([2*b + 2 - 2*a, 2*a - 4*b], dtype=float)

def h_libro(x):
    return np.array([[-2., 2.], [2., -4.]])

def f_elipse(x):
    a, b = x
    return a**2 + 10*b**2

def g_elipse(x):
    a, b = x
    return np.array([2*a, 20*b], dtype=float)

def h_elipse(x):
    return np.array([[2., 0.], [0., 20.]])

def f_silla(x):
    a, b = x
    return a**2 - b**2

def g_silla(x):
    a, b = x
    return np.array([2*a, -2*b], dtype=float)

def h_silla(x):
    return np.array([[2., 0.], [0., -2.]])


ESCENARIOS = {
    "1. Cuenco convexo": {
        "f": f_cuenco, "g": g_cuenco, "h": h_cuenco,
        "limites": (-2., 6., -2., 4.),
        "optimo": (2., 1.),
        "tipo": "Minimo global",
        "descripcion": (
            "Funcion convexa con minimo global en (2, 1). "
            "Es el escenario recomendado para comparar algoritmos."
        ),
        "maximizar": False,
    },
    "2. Funcion del libro": {
        "f": f_libro, "g": g_libro, "h": h_libro,
        "limites": (-3., 5., -3., 3.),
        "optimo": (2., 1.),
        "tipo": "Maximo global",
        "descripcion": (
            "La funcion original tiene un maximo en (2, 1). "
            "Para usar algoritmos de minimizacion, se minimiza su negativo."
        ),
        "maximizar": True,
    },
    "3. Elipse alargada": {
        "f": f_elipse, "g": g_elipse, "h": h_elipse,
        "limites": (-4., 4., -2., 2.),
        "optimo": (0., 0.),
        "tipo": "Minimo global",
        "descripcion": (
            "Las curvaturas son diferentes en x e y. "
            "Permite estudiar zigzag y sensibilidad al tamano del paso."
        ),
        "maximizar": False,
    },
    "4. Punto silla": {
        "f": f_silla, "g": g_silla, "h": h_silla,
        "limites": (-4., 4., -4., 4.),
        "optimo": (0., 0.),
        "tipo": "Punto silla",
        "descripcion": (
            "El gradiente se anula en el origen, pero no hay minimo "
            "ni maximo local: la funcion crece en una direccion y "
            "disminuye en otra."
        ),
        "maximizar": False,
    },
}


# ---------------- HERRAMIENTAS MATEMATICAS ----------------

def clasificar_hessiano(H, tol=1e-8):
    valores = np.linalg.eigvalsh(H)

    if np.all(valores > tol):
        texto = "Definido positivo: curvatura de minimo local."
    elif np.all(valores < -tol):
        texto = "Definido negativo: curvatura de maximo local."
    elif np.any(valores > tol) and np.any(valores < -tol):
        texto = "Indefinido: compatible con un punto silla."
    else:
        texto = "Semidefinido o degenerado: se requiere mas analisis."

    return valores, texto


def busqueda_armijo(f, x, p, g, alpha_inicial=1.0):
    """Busca un paso que reduzca f para una direccion de descenso."""
    alpha = alpha_inicial
    c = 1e-4
    rho = 0.5
    fx = f(x)
    pendiente = float(np.dot(g, p))

    if not np.isfinite(pendiente) or pendiente >= 0:
        return 0.0

    for _ in range(40):
        xn = x + alpha*p
        fn = f(xn)

        if np.isfinite(fn) and fn <= fx + c*alpha*pendiente:
            return alpha

        alpha *= rho

    return 0.0


def ejecutar_metodo(
    f, grad, hess, x0, metodo, max_iter, tol,
    paso_fijo=0.05, busqueda=True
):
    x = np.array(x0, dtype=float)
    puntos = [x.copy()]
    historial = []
    estado = "Se alcanzo el limite de iteraciones."

    # B aproxima la inversa del Hessiano en BFGS.
    B = np.eye(2)
    g_anterior = None
    p_anterior = None

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

        if (
            not np.isfinite(fx)
            or not np.all(np.isfinite(g))
            or not np.all(np.isfinite(x))
        ):
            estado = "Calculo no finito: el algoritmo se detuvo."
            break

        if ng < tol:
            estado = "Gradiente pequeno: punto critico aproximado."
            break

        if k == max_iter:
            break

        # Direccion de busqueda
        if metodo == "Descenso del gradiente":
            p = -g

        elif metodo == "Gradiente conjugado":
            if g_anterior is None:
                p = -g
            else:
                denominador = float(np.dot(g_anterior, g_anterior))

                if denominador > 1e-20:
                    beta = max(
                        0.0,
                        float(np.dot(g, g - g_anterior))
                        / denominador
                    )
                else:
                    beta = 0.0

                p = -g + beta*p_anterior

                if float(np.dot(p, g)) >= 0:
                    p = -g

        elif metodo == "Newton":
            H = np.asarray(hess(x), dtype=float)

            try:
                p = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                p = -g

            # Si Newton no produce descenso, usar -gradiente.
            if (
                not np.all(np.isfinite(p))
                or float(np.dot(p, g)) >= 0
            ):
                p = -g

        elif metodo == "BFGS (cuasi-Newton)":
            p = -B @ g

            if (
                not np.all(np.isfinite(p))
                or float(np.dot(p, g)) >= 0
            ):
                B = np.eye(2)
                p = -g

        else:
            p = -g

        # Tamano de paso
        if busqueda:
            alpha = busqueda_armijo(f, x, p, g)
        else:
            alpha = paso_fijo

        if alpha <= 1e-14:
            estado = "No se encontro un paso util; posible estancamiento."
            break

        xn = x + alpha*p

        if not np.all(np.isfinite(xn)) or np.linalg.norm(xn) > 1e8:
            estado = "El metodo se volvio inestable y se detuvo."
            break

        gn = np.asarray(grad(xn), dtype=float)

        # Actualizacion de BFGS para aproximar la inversa del Hessiano.
        if metodo == "BFGS (cuasi-Newton)":
            s = xn - x
            y = gn - g
            ys = float(np.dot(y, s))

            if ys > 1e-12:
                rho = 1.0 / ys
                I = np.eye(2)
                V = I - rho*np.outer(s, y)
                B = V @ B @ V.T + rho*np.outer(s, s)
            else:
                B = np.eye(2)

        g_anterior = g.copy()
        p_anterior = p.copy()
        x = xn
        puntos.append(x.copy())

    return (
        np.asarray(puntos),
        pd.DataFrame(historial),
        estado,
        x,
    )


# ---------------- INTERFAZ ----------------

st.title("Laboratorio interactivo de optimizacion")
st.subheader(
    "Capitulo 14.2: gradientes, Hessianos y metodos de optimizacion"
)
st.write(
    "Modifica la funcion y el punto inicial, ejecuta algoritmos y "
    "observa como cambian sus trayectorias y su convergencia."
)

with st.sidebar:
    st.header("Panel de control")

    nombre_escenario = st.selectbox(
        "Selecciona una funcion",
        list(ESCENARIOS.keys())
    )
    escenario = ESCENARIOS[nombre_escenario]

    metodo = st.selectbox(
        "Metodo numerico",
        [
            "Descenso del gradiente",
            "Gradiente conjugado",
            "Newton",
            "BFGS (cuasi-Newton)",
        ]
    )

    xmin, xmax, ymin, ymax = escenario["limites"]

    st.markdown("**Punto inicial**")
    x0 = st.slider(
        "Coordenada x",
        float(xmin), float(xmax),
        float(np.clip(xmin + 0.8, xmin, xmax))
    )
    y0 = st.slider(
        "Coordenada y",
        float(ymin), float(ymax),
        float(np.clip(ymin + 0.7, ymin, ymax))
    )

    max_iter = st.slider(
        "Maximo de iteraciones", 5, 200, 60, 5
    )
    tol = st.select_slider(
        "Tolerancia del gradiente",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        value=1e-5,
        format_func=lambda v: f"{v:.0e}"
    )

    busqueda = st.toggle(
        "Busqueda lineal automatica",
        value=True,
        help="Busca un paso que reduzca la funcion objetivo."
    )

    paso_fijo = st.slider(
        "Paso fijo",
        0.001, 0.5, 0.05, 0.001,
        disabled=busqueda
    )

# Convertir maximizar f en minimizar -f.
f_original = escenario["f"]
g_original = escenario["g"]
h_original = escenario["h"]

if escenario["maximizar"]:
    f = lambda x: -f_original(x)
    grad = lambda x: -g_original(x)
    hess = lambda x: -h_original(x)
    st.info(
        "Esta funcion tiene un maximo. Para utilizar los metodos "
        "de minimizacion, el motor trabaja con -f(x,y)."
    )
else:
    f, grad, hess = f_original, g_original, h_original

st.info(escenario["descripcion"])

puntos, historial, estado, xf = ejecutar_metodo(
    f, grad, hess,
    [x0, y0],
    metodo,
    max_iter,
    tol,
    paso_fijo,
    busqueda,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Simulador",
    "Gradiente y Hessiano",
    "Comparar metodos",
    "Reto para la clase",
])

# ---------------- TAB 1: SIMULADOR ----------------

with tab1:
    col_graf, col_resultados = st.columns([1.7, 1])

    with col_graf:
        xg, yg = np.meshgrid(
            np.linspace(xmin, xmax, 120),
            np.linspace(ymin, ymax, 120)
        )

        zg = np.vectorize(
            lambda a, b: f(np.array([a, b]))
        )(xg, yg)

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        niveles = np.linspace(np.nanmin(zg), np.nanmax(zg), 25)
        curvas = ax.contour(
            xg, yg, zg,
            levels=niveles,
            cmap="viridis",
            linewidths=0.9
        )
        ax.clabel(curvas, inline=True, fontsize=7)

        ax.plot(
            puntos[:, 0], puntos[:, 1],
            "-o", color="#FFC107",
            linewidth=2, markersize=4,
            label="Trayectoria"
        )
        ax.scatter(
            [x0], [y0], s=100,
            color="#00BCD4", edgecolor="white",
            label="Inicio", zorder=5
        )
        ax.scatter(
            [xf[0]], [xf[1]], s=120,
            marker="X", color="#FF5722",
            edgecolor="white", label="Punto final",
            zorder=6
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Curvas de nivel y recorrido")
        ax.legend(facecolor="#171A21", labelcolor="white")
        ax.grid(alpha=0.12)

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with col_resultados:
        st.subheader("Resultados")

        st.metric("Iteraciones", len(historial) - 1)
        st.metric("x final", f"{xf[0]:.6f}")
        st.metric("y final", f"{xf[1]:.6f}")
        st.metric(
            "Funcion original",
            f"{f_original(xf):.6f}"
        )
        st.metric(
            "Norma del gradiente",
            f"{np.linalg.norm(grad(xf)):.3e}"
        )

        st.write("**Estado:**")
        st.write(estado)

        valores, clasificacion = clasificar_hessiano(hess(xf))
        st.write("**Hessiano en el punto final:**")
        st.write(clasificacion)

        st.caption(
            "Un gradiente pequeno indica un punto critico aproximado, "
            "pero no garantiza por si solo un minimo."
        )

    st.subheader("Tabla de iteraciones")

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
        "Descargar historial CSV",
        historial.to_csv(index=False).encode("utf-8"),
        file_name="historial_optimizacion.csv",
        mime="text/csv"
    )

# ---------------- TAB 2: GRADIENTE Y HESSIANO ----------------

with tab2:
    st.subheader("Gradiente: derivadas de primer orden")

    g0 = grad(np.array([x0, y0], dtype=float))
    st.latex(
        r"\nabla f(x,y)=\begin{bmatrix}"
        r"\frac{\partial f}{\partial x}\\"
        r"\frac{\partial f}{\partial y}"
        r"\end{bmatrix}"
    )

    st.write(f"En el punto inicial ({x0:.3f}, {y0:.3f}):")
    st.code(
        f"Gradiente = [{g0[0]:.6f}, {g0[1]:.6f}]\n"
        f"Norma = {np.linalg.norm(g0):.6f}"
    )

    st.markdown(
        "**Interpretacion:**\n"
        "- El gradiente apunta hacia el crecimiento mas rapido local.\n"
        "- El negativo del gradiente es la direccion de descenso mas pronunciado.\n"
        "- La tasa de aprendizaje determina el tamano del paso."
    )

    st.divider()
    st.subheader("Hessiano: derivadas de segundo orden")

    H0 = hess(np.array([x0, y0], dtype=float))
    st.latex(
        r"H(x,y)=\begin{bmatrix}"
        + f"{H0[0,0]:.4f}&{H0[0,1]:.4f}\\\\"
        + f"{H0[1,0]:.4f}&{H0[1,1]:.4f}"
        + r"\end{bmatrix}"
    )

    valores, clasificacion = clasificar_hessiano(H0)
    st.write(f"Autovalores: {valores[0]:.5f}, {valores[1]:.5f}")
    st.write(clasificacion)

    st.caption(
        "El criterio del Hessiano se utiliza para clasificar un punto "
        "critico bajo las condiciones matematicas correspondientes."
    )

# ---------------- TAB 3: COMPARACION ----------------

with tab3:
    st.subheader("Comparacion con el mismo punto inicial")

    nombres = [
        "Descenso del gradiente",
        "Gradiente conjugado",
        "Newton",
        "BFGS (cuasi-Newton)",
    ]

    filas = []
    historiales = {}

    for nombre in nombres:
        pts, hist, est, punto_final = ejecutar_metodo(
            f, grad, hess,
            [x0, y0],
            nombre,
            max_iter,
            tol,
            paso_fijo,
            busqueda,
        )

        historiales[nombre] = hist

        filas.append({
            "Metodo": nombre,
            "Iteraciones": len(hist) - 1,
            "x final": punto_final[0],
            "y final": punto_final[1],
            "Valor objetivo": f(punto_final),
            "Norma gradiente": np.linalg.norm(grad(punto_final)),
            "Estado": est,
        })

    comparacion = pd.DataFrame(filas)

    st.dataframe(
        comparacion.style.format({
            "x final": "{:.6f}",
            "y final": "{:.6f}",
            "Valor objetivo": "{:.8f}",
            "Norma gradiente": "{:.3e}",
        }),
        use_container_width=True,
        hide_index=True
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor("#0E1117")
    ax2.set_facecolor("#0E1117")

    for nombre, hist in historiales.items():
        ax2.semilogy(
            np.arange(len(hist)),
            np.maximum(hist["Norma gradiente"].to_numpy(), 1e-16),
            marker="o",
            markersize=3,
            label=nombre
        )

    ax2.set_xlabel("Iteracion")
    ax2.set_ylabel("Norma del gradiente")
    ax2.set_title("Comparacion de convergencia")
    ax2.legend(facecolor="#171A21", labelcolor="white")
    ax2.grid(alpha=0.2)

    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    st.caption(
        "El mejor metodo depende de la funcion, el punto inicial, "
        "la curvatura y la estrategia de busqueda lineal."
    )

# ---------------- TAB 4: RETO ----------------

with tab4:
    st.subheader("Comprueba lo aprendido")

    respuesta1 = st.radio(
        "1. Si el gradiente vale cero, ¿que podemos afirmar?",
        [
            "Siempre es un minimo global.",
            "Es un punto critico, pero hay que clasificarlo.",
            "La funcion vale necesariamente cero.",
            "El algoritmo no ha empezado.",
        ],
        key="pregunta1"
    )

    respuesta2 = st.radio(
        "2. ¿Que informacion proporciona el Hessiano?",
        [
            "La curvatura local mediante derivadas segundas.",
            "Solamente el punto inicial.",
            "Siempre garantiza convergencia.",
            "Solamente el numero de variables.",
        ],
        key="pregunta2"
    )

    respuesta3 = st.radio(
        "3. ¿Que caracteriza a BFGS?",
        [
            "Actualiza una aproximacion de la inversa del Hessiano.",
            "Utiliza exclusivamente el valor de la funcion.",
            "No utiliza gradientes.",
            "Siempre encuentra el optimo global de cualquier funcion.",
        ],
        key="pregunta3"
    )

    if st.button("Comprobar respuestas"):
        if respuesta1.startswith("Es un punto"):
            st.success("1. Correcto: puede ser minimo, maximo o punto silla.")
        else:
            st.error("1. Recuerda: gradiente cero no implica minimo global.")

        if respuesta2.startswith("La curvatura"):
            st.success("2. Correcto: el Hessiano contiene derivadas segundas.")
        else:
            st.error("2. Repasa la interpretacion de la matriz Hessiana.")

        if respuesta3.startswith("Actualiza"):
            st.success("3. Correcto: BFGS es un metodo cuasi-Newton.")
        else:
            st.error("3. BFGS aproxima la informacion de curvatura.")

st.divider()
st.caption(
    "Laboratorio didactico. Los resultados numericos deben interpretarse "
    "considerando las hipotesis y limitaciones de cada algoritmo."
)


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración inicial de la página web
st.set_page_config(page_title="Simulador de Optimización", layout="wide", initial_sidebar_state="expanded")
plt.style.use('dark_background')

# --- DEFINICIÓN DE FUNCIONES TOPOGRÁFICAS ---
escenarios = {
    "1. Función del Libro (Máximo en 2,1)": {
        "f": lambda x, y: 2*x*y + 2*x - x**2 - 2*y**2,
        "grad": lambda x, y: np.array([2*y + 2 - 2*x, 2*x - 4*y]),
        "H_inv": np.linalg.inv(np.array([[-2, 2], [2, -4]])),
        "limites": (-3, 5, -3, 3),
        "optimo": (2, 1)
    },
    "2. Punto Silla (El algoritmo se confunde)": {
        "f": lambda x, y: x**2 - y**2,
        "grad": lambda x, y: np.array([2*x, -2*y]),
        "H_inv": np.linalg.inv(np.array([[2, 0], [0, -2]])),
        "limites": (-5, 5, -5, 5),
        "optimo": (0, 0)
    },
    "3. Cresta Alargada (Demostración de Zigzag)": {
        "f": lambda x, y: -x**2 - 10*y**2,
        "grad": lambda x, y: np.array([-2*x, -20*y]),
        "H_inv": np.linalg.inv(np.array([[-2, 0], [0, -20]])),
        "limites": (-10, 10, -3, 3),
        "optimo": (0, 0)
    }
}

# --- INTERFAZ DE USUARIO ---
st.title('🌌 Simulador Topográfico: Métodos de Optimización')
st.markdown("Explora cómo diferentes algoritmos navegan distintas superficies matemáticas para encontrar el punto óptimo.")

col_controles, col_grafica = st.columns([1, 2.5])

with col_controles:
    st.subheader("⚙️ Panel de Control")
    seleccion = st.selectbox('Selecciona el Escenario Topográfico:', list(escenarios.keys()))
    metodo = st.radio('Algoritmo de Navegación:', ['Ascenso de Máxima Inclinación', 'Método de Newton'])
    
    st.markdown("---")
    escenario = escenarios[seleccion]
    x_min, x_max, y_min, y_max = escenario["limites"]
    
    x_inicio = st.slider('Coordenada Inicial X:', float(x_min), float(x_max), float(x_min) + 1.0)
    y_inicio = st.slider('Coordenada Inicial Y:', float(y_min), float(y_max), float(y_min) + 1.0)
    
    if metodo == 'Ascenso de Máxima Inclinación':
        alpha = st.slider('Multiplicador de Paso (Tasa de aprendizaje):', 0.01, 0.50, 0.05, step=0.01)
        max_iter = st.slider('Límite de Pasos:', 5, 200, 50)
    else:
        alpha = None
        max_iter = st.slider('Límite de Pasos:', 1, 15, 5)

# --- MOTOR DE CÁLCULO ---
f = escenario["f"]
grad = escenario["grad"]
H_inv = escenario["H_inv"]

camino_x, camino_y = [x_inicio], [y_inicio]
curr_x, curr_y = x_inicio, y_inicio

for _ in range(max_iter):
    g = grad(curr_x, curr_y)
    if np.linalg.norm(g) < 0.001: 
        break # Se alcanzó el óptimo o se estancó
        
    if metodo == 'Ascenso de Máxima Inclinación':
        paso_x, paso_y = alpha * g[0], alpha * g[1]
    else:
        paso = -np.dot(H_inv, g)
        paso_x, paso_y = paso[0], paso[1]
        
    curr_x, curr_y = curr_x + paso_x, curr_y + paso_y
    camino_x.append(curr_x)
    camino_y.append(curr_y)

# --- RENDERIZADO VISUAL ---
with col_grafica:
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    contour = ax.contour(x_grid, y_grid, f(x_grid, y_grid), levels=20, cmap='viridis', alpha=0.8)
    ax.clabel(contour, inline=True, fontsize=7, colors='#A0A0A0')
    
    # Dibujar Elementos
    ax.plot(escenario["optimo"][0], escenario["optimo"][1], color='#FF5722', marker='*', markersize=20, label='Centro Teórico (Óptimo/Silla)')
    ax.plot(camino_x, camino_y, color='#FFC107', marker='o', linestyle='-', linewidth=2.5, markersize=5, label='Ruta del Algoritmo')
    ax.plot(x_inicio, y_inicio, color='#00BCD4', marker='o', markersize=10, label='Punto de Despliegue')
    
    ax.legend(loc="upper left", facecolor='#0E1117', edgecolor='#333333', labelcolor='white')
    st.pyplot(fig)
    
    # Métricas de Resultados
    st.markdown("### 📊 Rendimiento del Algoritmo")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Iteraciones utilizadas", value=len(camino_x)-1)
    m2.metric(label="Posición X final", value=f"{curr_x:.4f}")
    m3.metric(label="Posición Y final", value=f"{curr_y:.4f}")

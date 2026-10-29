import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Activar el estilo oscuro para que coincida con el video
plt.style.use('dark_background')

def f(x, y): return 2*x*y + 2*x - x**2 - 2*y**2
def gradiente(x, y): return np.array([2*y + 2 - 2*x, 2*x - 4*y])
H_inv = np.linalg.inv(np.array([[-2, 2], [2, -4]]))

st.set_page_config(layout="wide")
st.title('Simulador: Métodos con Gradiente')
st.markdown("Encuentra el punto máximo de la función $f(x,y) = 2xy + 2x - x^2 - 2y^2$")

col_controles, col_grafica = st.columns([1, 2])

with col_controles:
    st.subheader("Configuración")
    metodo = st.selectbox('Algoritmo de Optimización:', ['Ascenso de Máxima Inclinación', 'Método de Newton'])
    x_inicio = st.slider('Punto de Inicio X:', -3.0, 4.0, -2.0)
    y_inicio = st.slider('Punto de Inicio Y:', -3.0, 3.0, -1.0)
    
    if metodo == 'Ascenso de Máxima Inclinación':
        alpha = st.slider('Tamaño del paso (Multiplicador):', 0.01, 0.50, 0.15, step=0.01)
        max_iter = st.slider('Límite de pasos:', 10, 100, 30)
    else:
        alpha = None
        max_iter = st.slider('Límite de pasos:', 1, 10, 5)

camino_x, camino_y = [x_inicio], [y_inicio]
curr_x, curr_y = x_inicio, y_inicio

if metodo == 'Ascenso de Máxima Inclinación':
    for _ in range(max_iter):
        g = gradiente(curr_x, curr_y)
        curr_x, curr_y = curr_x + alpha * g[0], curr_y + alpha * g[1]
        camino_x.append(curr_x)
        camino_y.append(curr_y)
        if np.linalg.norm(g) < 0.01: break
else:
    for _ in range(max_iter):
        g = gradiente(curr_x, curr_y)
        paso = -np.dot(H_inv, g)
        curr_x, curr_y = curr_x + paso[0], curr_y + paso[1]
        camino_x.append(curr_x)
        camino_y.append(curr_y)
        if np.linalg.norm(g) < 0.01: break

with col_grafica:
    x_grid, y_grid = np.meshgrid(np.linspace(-3, 5, 100), np.linspace(-3, 3, 100))
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ajustar colores del fondo para que coincida con el video
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Mapa de colores 'viridis' para imitar los tonos del video
    contour = ax.contour(x_grid, y_grid, f(x_grid, y_grid), levels=15, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8, colors='white')
    
    # Colores neón idénticos a los de la presentación
    ax.plot(2, 1, color='#FF5722', marker='*', markersize=18, label='Máximo Óptimo (2,1)')
    ax.plot(camino_x, camino_y, color='#FFC107', marker='o', linestyle='-', linewidth=2, markersize=6, label='Trayectoria')
    ax.plot(x_inicio, y_inicio, color='#00BCD4', marker='o', markersize=10, label='Punto de Inicio')
    
    ax.legend(loc="upper left", facecolor='#0E1117', edgecolor='white')
    st.pyplot(fig)
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="Pasos realizados", value=len(camino_x)-1)
    col_res2.metric(label="Coordenada final alcanzada", value=f"({curr_x:.2f}, {curr_y:.2f})")

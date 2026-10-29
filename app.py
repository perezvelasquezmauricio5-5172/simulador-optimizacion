import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict

# 1. CONFIGURACIÓN GLOBAL (Debe ser el primer comando de Streamlit)
st.set_page_config(page_title="Simulador de Optimización", layout="wide", initial_sidebar_state="expanded")
plt.style.use('dark_background')

# 2. DEFINICIÓN DE ESTRUCTURAS DE DATOS (Data Models)
@dataclass
class EscenarioTopografico:
    """Modelo de datos para representar una superficie matemática."""
    f: Callable[[float, float], float]
    grad: Callable[[float, float], np.ndarray]
    H_inv: np.ndarray
    limites: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    optimo: Tuple[float, float]

# 3. BASE DE CONOCIMIENTO Y CACHÉ
@st.cache_resource
def cargar_escenarios() -> Dict[str, EscenarioTopografico]:
    """Genera y almacena en caché los escenarios para no recalcularlos en cada interacción."""
    return {
        "1. Función del Libro (Máximo en 2,1)": EscenarioTopografico(
            f=lambda x, y: 2*x*y + 2*x - x**2 - 2*y**2,
            grad=lambda x, y: np.array([2*y + 2 - 2*x, 2*x - 4*y]),
            H_inv=np.linalg.inv(np.array([[-2, 2], [2, -4]])),
            limites=(-3.0, 5.0, -3.0, 3.0),
            optimo=(2.0, 1.0)
        ),
        "2. Punto Silla (El algoritmo se confunde)": EscenarioTopografico(
            f=lambda x, y: x**2 - y**2,
            grad=lambda x, y: np.array([2*x, -2*y]),
            H_inv=np.linalg.inv(np.array([[2, 0], [0, -2]])),
            limites=(-5.0, 5.0, -5.0, 5.0),
            optimo=(0.0, 0.0)
        ),
        "3. Cresta Alargada (Demostración de Zigzag)": EscenarioTopografico(
            f=lambda x, y: -x**2 - 10*y**2,
            grad=lambda x, y: np.array([-2*x, -20*y]),
            H_inv=np.linalg.inv(np.array([[-2, 0], [0, -20]])),
            limites=(-10.0, 10.0, -3.0, 3.0),
            optimo=(0.0, 0.0)
        )
    }

# 4. MOTOR DE OPTIMIZACIÓN (Lógica de negocio separada de la UI)
class Optimizador:
    """Clase estática encargada de ejecutar los algoritmos numéricos."""
    
    @staticmethod
    def calcular_trayectoria(escenario: EscenarioTopografico, metodo: str, 
                             x_ini: float, y_ini: float, alpha: float, 
                             max_iter: int, tol: float = 0.001) -> Tuple[List[float], List[float], float, float]:
        camino_x, camino_y = [x_ini], [y_ini]
        curr_x, curr_y = x_ini, y_ini
        
        for _ in range(max_iter):
            g = escenario.grad(curr_x, curr_y)
            if np.linalg.norm(g) < tol:
                break
                
            if metodo == 'Ascenso de Máxima Inclinación':
                paso_x, paso_y = alpha * g[0], alpha * g[1]
            else: # Método de Newton
                paso = -np.dot(escenario.H_inv, g)
                paso_x, paso_y = paso[0], paso[1]
                
            curr_x, curr_y = curr_x + paso_x, curr_y + paso_y
            camino_x.append(curr_x)
            camino_y.append(curr_y)
            
        return camino_x, camino_y, curr_x, curr_y

# 5. RENDERIZADO DE INTERFAZ DE USUARIO (Vista)
def renderizar_ui() -> None:
    st.title('🌌 Simulador Topográfico: Análisis Numérico')
    st.markdown("Plataforma interactiva para la evaluación de algoritmos de optimización multivariable.")

    escenarios_db = cargar_escenarios()
    
    col_controles, col_grafica = st.columns([1, 2.5])

    with col_controles:
        with st.expander("⚙️ Configuración del Escenario", expanded=True):
            seleccion = st.selectbox('Topología de la Superficie:', list(escenarios_db.keys()))
            metodo = st.radio('Algoritmo de Optimización:', ['Ascenso de Máxima Inclinación', 'Método de Newton'])
        
        escenario = escenarios_db[seleccion]
        x_min, x_max, y_min, y_max = escenario.limites
        
        with st.expander("🎛️ Parámetros de Inicialización", expanded=True):
            x_inicio = st.slider('Vector Inicial X₀:', x_min, x_max, x_min + 1.0)
            y_inicio = st.slider('Vector Inicial Y₀:', y_min, y_max, y_min + 1.0)
            
            if metodo == 'Ascenso de Máxima Inclinación':
                alpha = st.slider('Tasa de Aprendizaje (α):', 0.01, 0.50, 0.05, step=0.01)
                max_iter = st.slider('Iteraciones Máximas:', 5, 200, 50)
            else:
                alpha = 0.0
                max_iter = st.slider('Iteraciones Máximas:', 1, 15, 5)

    # Ejecución del motor numérico
    camino_x, camino_y, final_x, final_y = Optimizador.calcular_trayectoria(
        escenario, metodo, x_inicio, y_inicio, alpha, max_iter
    )

    with col_grafica:
        # Optimización de renderizado de la malla
        x_linspace = np.linspace(x_min, x_max, 100)
        y_linspace = np.linspace(y_min, y_max, 100)
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
        z_grid = escenario.f(x_grid, y_grid)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        contour = ax.contour(x_grid, y_grid, z_grid, levels=20, cmap='viridis', alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=7, colors='#A0A0A0')
        
        # Trazado de trayectorias
        ax.plot(escenario.optimo[0], escenario.optimo[1], color='#FF5722', marker='*', markersize=20, label='Punto Crítico Analítico')
        ax.plot(camino_x, camino_y, color='#FFC107', marker='o', linestyle='-', linewidth=2.5, markersize=5, label='Trayectoria del Algoritmo')
        ax.plot(x_inicio, y_inicio, color='#00BCD4', marker='o', markersize=10, label='Punto de Inicio')
        
        ax.legend(loc="upper left", facecolor='#0E1117', edgecolor='#333333', labelcolor='white')
        
        # Eliminar bordes para un look más limpio
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            
        st.pyplot(fig)
        
        # Panel de Telemetría
        st.markdown("### 📊 Telemetría de Convergencia")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Iteraciones Computadas", value=len(camino_x)-1)
        m2.metric(label="Vector Final X", value=f"{final_x:.4f}")
        m3.metric(label="Vector Final Y", value=f"{final_y:.4f}")

# 6. PUNTO DE ENTRADA (Entry point estándar en Python)
if __name__ == "__main__":
    renderizar_ui()

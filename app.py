import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MAT201: Gradient & Steepest Ascent", layout="wide")

# --- TITLE & STUDENT INFO ---
st.title("Calculus MAT201: Gradient & Direction of Steepest Ascent")
st.markdown("""
**Topic:** Visualizing Functions of Several Variables & Gradients
*Designed for MAT201 Assignment 2*
""")
st.divider()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Input Function & Point")
st.sidebar.info("Use standard Python syntax: e.g., x**2 + y**2, sin(x)*cos(y), exp(-x**2)")

# Input for the function
func_input = st.sidebar.text_input("Enter a function f(x, y):", value="x**2 + y**2")
# Inputs for the point (a, b)
val_a = st.sidebar.number_input("x-coordinate (a):", value=1.0, step=0.1)
val_b = st.sidebar.number_input("y-coordinate (b):", value=1.0, step=0.1)

# --- MATHEMATICAL COMPUTATION (SYMPY) ---
try:
    x, y = sp.symbols('x y')
    f_expr = sp.sympify(func_input)
    
    # Calculate Partial Derivatives
    fx = sp.diff(f_expr, x)
    fy = sp.diff(f_expr, y)
    
    # Calculate Numerical Values at (a,b)
    f_val = f_expr.subs({x: val_a, y: val_b})
    fx_val = fx.subs({x: val_a, y: val_b})
    fy_val = fy.subs({x: val_a, y: val_b})
    
    # Gradient Magnitude
    grad_mag = sp.sqrt(fx_val**2 + fy_val**2)

    # --- SECTION 1: CONCEPTS & CALCULATIONS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Step 1: Mathematical Analysis")
        st.write("We analyze the function at the specific point to find the gradient.")
        
        # Display Math nicely
        st.latex(r"f(x, y) = " + sp.latex(f_expr))
        st.markdown(f"**Point:** $P({val_a}, {val_b})$")
        
        st.subheader("Key Concepts:")
        st.markdown("""
        1. **Partial Derivatives:** Measuring the rate of change in x and y directions.
        2. **The Gradient Vector ($\nabla f$):** A vector composed of partial derivatives $\langle f_x, f_y \\rangle$.
        3. **Steepest Ascent:** The gradient vector points in the direction of the greatest rate of increase of the function.
        """)

    with col2:
        st.header("Step 2: Computational Results")
        st.success("Calculations executed successfully.")
        
        st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx) + r" \quad \Rightarrow \quad " + f"{float(fx_val):.4f}")
        st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy) + r" \quad \Rightarrow \quad " + f"{float(fy_val):.4f}")
        
        st.markdown("### The Gradient Vector:")
        st.latex(r"\nabla f(" + str(val_a) + "," + str(val_b) + r") = \langle " + f"{float(fx_val):.4f}, {float(fy_val):.4f}" + r" \rangle")
        
        st.markdown(f"**Max Rate of Ascent (Magnitude):** {float(grad_mag):.4f}")

    st.divider()

    # --- SECTION 2: VISUALIZATION (PLOTLY) ---
    st.header("Step 3: Interactive Visualization")
    st.write("The 3D plot shows the surface. The 2D Contour plot shows the 'map' view. The RED arrow indicates the direction of steepest ascent.")

    # Prepare data for plotting
    x_range = np.linspace(val_a - 2, val_a + 2, 50)
    y_range = np.linspace(val_b - 2, val_b + 2, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Lambdify for fast numerical evaluation
    f_func = sp.lambdify((x, y), f_expr, 'numpy')
    Z = f_func(X, Y)

    # Create 3D Plot
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
    # Add a marker for the point
    fig_3d.add_trace(go.Scatter3d(x=[val_a], y=[val_b], z=[float(f_val)], mode='markers', marker=dict(size=10, color='red'), name='Point P'))
    fig_3d.update_layout(title='3D Surface Visualization', autosize=False, width=600, height=500)

    # Create 2D Contour Plot with Gradient Arrow
    fig_2d = go.Figure(data=go.Contour(z=Z, x=x_range, y=y_range, colorscale='Viridis'))
    # Add the gradient arrow (quiver)
    # We scale the arrow to make it visible
    scale_factor = 0.5
    fig_2d.add_annotation(
        x=val_a + float(fx_val) * scale_factor, 
        y=val_b + float(fy_val) * scale_factor,
        ax=val_a, 
        ay=val_b,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
    )
    fig_2d.update_layout(title=f'2D Contour Map & Gradient Vector (Red Arrow)', autosize=False, width=600, height=500)

    # Display Plots side by side
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_3d, use_container_width=True)
    c2.plotly_chart(fig_2d, use_container_width=True)

    # --- SECTION 3: REAL WORLD APPLICATION ---
    st.divider()
    st.header("Step 4: Significance & AI Application")
    st.info("How does this relate to Generative AI and the Real World?")
    
    st.markdown("""
    **1. Real-World Analogy: Hiking**
    Imagine you are hiking on the terrain shown in the 3D plot above. If you want to reach the peak as fast as possible, you must look at your feet and walk in the direction where the slope is steepest. This is exactly what the **Gradient Vector** calculates.
    
    **2. Application in Artificial Intelligence (Gradient Descent)**
    This concept is the backbone of Machine Learning and AI (including models like ChatGPT).
    * **The Loss Function:** AI models have a "Loss Function" (like an inverted version of our 3D plot) representing errors.
    * **Training:** To train an AI, we want to *minimize* error.
    * **The Method:** The computer calculates the **Gradient** of the error function to find the direction of steepest descent (moving opposite to the gradient). It takes small steps in that direction until it finds the optimal solution.
    """)

except Exception as e:
    st.error(f"Error parsing function. Please check your syntax. Error: {e}")

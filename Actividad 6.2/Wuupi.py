# Importamos librerías
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import plotly.express as px
from scipy import stats
import utils as ut

# Configuración de la página
st.set_page_config(
    page_title="Wuupi Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargamos estilo CSS
ut.local_css("estilo.css")

# Definimos la instancia
@st.cache_resource

def load_data():
    dfc = pd.read_csv('DataAnalyticsCat.csv')
    dfc.columns = dfc.columns.str.title()
    dfn = pd.read_csv('DataAnalyticsNum.csv')
    dfc = dfc.drop(columns=['Unnamed: 0'])
    dfn = dfn.drop(columns=['Unnamed: 0'])
    dfn.columns = dfn.columns.str.title()
    dfc['Usuario'] = dfc['Usuario'].str.title()
    dfc['Administrador'] = dfc['Administrador'].str.title()
    dfc = dfc.applymap(lambda x: x.title() if isinstance(x, str) else x)
    categoricas = ['Presionó Botón Correcto', 'Mini Juego', 'Color Presionado', 'Dificultad', 'Juego', 'Auto Push']
    numericas = ['Tiempo de Interacción', 'Número de Interacción Por Lección', 'Tiempo De Lección', 'Tiempo De Sesión']
    usuarios = dfc['Usuario'].unique()
    usuarios = sorted(usuarios)
    return dfc, dfn, categoricas, numericas, usuarios


dfc, dfn, categoricas, numericas, usuarios = load_data()
Variable_Cat = 'Mini Juego'  

# Sidebar estilizado
with st.sidebar:
    with st.container(border=True):
        st.image("wuupi.png", width=200)
        st.markdown("## 📂 Filtros del Dashboard")
        
        # Selección de tipo de análisis (siempre visible)
        View = st.selectbox("🧠 Tipo de Análisis", [
            "Extracción de Características",
            "Regresión Lineal",
            "Regresión No Lineal",
            "Regresión Logística",
            "ANOVA"
        ])

        # Filtros adicionales solo si se selecciona Extracción de Características
        if View == "Extracción de Características":
            todos_usuarios = st.checkbox("👥 Todos los Usuarios", value=True)
            if not todos_usuarios:
                usuario = st.selectbox("👤 Usuario", options=usuarios)
                dfc = dfc[dfc['Usuario'] == usuario]
                dfn = dfn[dfn['Usuario'] == usuario]

            Variable_Cat = st.selectbox("📊 Variable Categórica", options=categoricas)

        elif View == "Regresión Lineal":
            # Filtros de usuario para la vista de regresión lineal
            #todos_usuarios = st.markdown("👥 Todos los Usuarios")
            #if not todos_usuarios:
                #usuario = st.selectbox("👤 Usuario", options=usuarios)
                #dfc = dfc[dfc['Usuario'] == usuario]
                #dfn = dfn[dfn['Usuario'] == usuario]
            
            # Selección de variables numéricas para regresión
            numeric_df = dfn.select_dtypes(include=["float", "int"])
            Lista_num = sorted(numeric_df.columns.tolist())
            Variable_y = Lista_num
    
            id_a_nombre = {
                3: "Nicolas", 20: "Austin", 5: "Leonardo", 15: "Iker Benjamin", 19: "Kytzia", 12: "Yael David", 10: "Denisse",
                9: "Sergio Angel", 14: "Erick", 18: "Concepcion", 24: "Joshua", 13: "Valentin", 11: "Carlos Enrique",
                22: "Jose Ian", 17: "Erick Osvaldo", 28: "Ingrid", 30: "Carlos Abel", 34: "Jesus Eduardo", 7: "Ramiro Isai", 29: "Rene",
                21: "Jose Ignacio Tadeo", 6: "Jesus Alejandro", 31: "Arlett", 23: "Ashley", 27: "Benjamin", 33: "Irving", 25: "Yeremi Yazmin",
                8: "Adrian", 2: "Aleida", 16: "Nicolas |", 1: "Leonardo", 4: "Jose Javier", 26: "Ma Del Rosario", 32: "Esmeralda"}

            dfn['Usuario Nombre'] = dfn['Usuario'].map(id_a_nombre)

            df_filtrado_multi = dfn
            #df_filtrado_multi = dfn[dfn[Variable_y] > 0].copy()

            usuarios_multi = dfn['Usuario Nombre'].unique()
            usuarios_multi = sorted(usuarios_multi.tolist())

            # Filtros de usuario para la vista de regresión lineal
            todos_usuarios = st.checkbox("👥 Todos los Usuarios", value=True)
            if not todos_usuarios:
                usuario = st.multiselect("👤 Usuarios", options=usuarios_multi, default=usuarios_multi[:1])
                df_filtrado_multi = dfn[dfn['Usuario Nombre'].isin(usuario)]

            # Selección de variables para regresión
            Variable_y = st.selectbox("📌 Variable Objetivo (Y)", options=Lista_num)
            Variable_x = st.selectbox("📎 Variable Independiente (X) - Simple", options=Lista_num)
            Variables_x = st.multiselect("📊 Variables Independientes - Múltiple", options=Lista_num, default=[Variable_x])

# Diccionarios de colores personalizados
color_maps = {
    'Auto Push': {'Si': '#6CE5C3', 'No': '#4A4E69'},
    'Color Presionado': {'Red': '#F26D6D', 'Blue': '#5DADEC', 'Green': '#6CE5C3','Yellow': '#F2C94C', 'Violet': '#B18AE0'},
    'Presionó Botón Correcto': {'Si': '#6CE5C3', 'No': '#4A4E69'},
    'Dificultad': {'Episodio 1': '#D1FAF0', 'Episodio 2': '#A3F1DC', 'Episodio 3': '#6CE5C3', 'Episodio 4': '#2BA78C'},
    'Juego': {'Astro': '#6CE5C3', 'Cadetes': '#6CD2E5'},
    'Mini Juego': {'Asteroides': '#2BA78C', 'Restaurante': '#4BC7AB', 'Estrellas': '#6CE5C3', 'Gusanos': '#89F0D2', 'Sonidos Y Animales': '#A6F5DD',
    'Animales Y Colores': '#C3FAE9', 'Figuras Y Colores': '#B8EAF2', 'Partes Del Cuerpo': '#A0DDE7', 'Despegue': '#7FC8D9', 'Minigame_0': '#5FB3C4',
    'Minigame_1': '#3F9EB0', 'Minigame_2': '#257F91', 'Minigame_3': '#145C6A'},
}

color_map = color_maps.get(Variable_Cat, {})
        
# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":
    
    # Tabla de frecuencias
    Tabla_frecuencias = dfc[Variable_Cat].value_counts().reset_index()
    Tabla_frecuencias.columns = ['Categorias', 'Frecuencia']

    # KPIs
    st.markdown("### KPIs")
    col1, col2, col3, col4 = st.columns(4)

    # Contenedor 1  
    with col1:
        st.metric("📈 Categoría más Frecuente", Tabla_frecuencias.iloc[0]['Categorias'], f"{Tabla_frecuencias.iloc[0]['Frecuencia']} ocurrencias")

    # Contenedor 2
    with col2:
        dfc_total = load_data()[0]
        tiempo_promedio_global = round(dfc_total['Tiempo De Interacción'].mean(), 2)
        tiempo_promedio_usuario = round(dfc['Tiempo De Interacción'].mean(), 2)
        diferencia = round(tiempo_promedio_usuario - tiempo_promedio_global, 2)
        if diferencia > 0:
            delta_texto = f"+{diferencia} seg"
            delta_color = "inverse"
        elif diferencia < 0:
            delta_texto = f"{diferencia} seg"
            delta_color = "inverse"
        else:
            delta_texto = ""
            delta_color = "off"
        st.metric("⏱️ Tiempo Promedio de Interacción", f"{tiempo_promedio_usuario} seg", delta=delta_texto, delta_color=delta_color)
    
    # Contenedor 4
    with col3:
        total_interacciones = len(dfc)
        respuestas_correctas = dfc[dfc['Presionó Botón Correcto'] == 'Si'].shape[0]
        porcentaje_correctas = round((respuestas_correctas / total_interacciones) * 100, 2)
        st.metric("✅ Respuestas Correctas", f"{porcentaje_correctas} %")
    
    # Contenedor 3
    with col4:
        sesiones_completadas = (dfc['Tiempo De Sesión'] != 0).sum()
        st.metric("🧩 Sesiones Completadas", sesiones_completadas)

    
    #Separador
    st.markdown("---")

    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)

    # Contenedor A
    with Contenedor_A:
        figure1 = px.bar(
            data_frame=Tabla_frecuencias,
            x='Frecuencia',
            y='Categorias',
            title=f'Frecuencia por {Variable_Cat.title()}',
            color='Categorias',
            color_discrete_map=color_map,
            orientation='h'
        )
        figure1.update_layout(height=500, yaxis_title=Variable_Cat.title(), xaxis_title='Frecuencia')
        st.plotly_chart(ut.aplicarFormatoChart(figure1), key='chart-figure1', use_container_width=True)

    # Contenedor B
    with Contenedor_B:
        figure2 = px.pie(
            data_frame=Tabla_frecuencias,
            values='Frecuencia',
            names='Categorias',
            title='Frecuencia por Categoría',
            color='Categorias',
            color_discrete_map=color_map
        )
        figure2.update_traces(textposition='inside', textinfo='percent+label')
        figure2.update_layout(height=500)
        st.plotly_chart(ut.aplicarFormatoChart(figure2), key='chart-figure2', use_container_width=True)

    # Fila 2
    Contenedor_C, Contenedor_D = st.columns(2)
    with Contenedor_C:
        dfc['Fecha'] = pd.to_datetime(dfc['Fecha'])
        df_promedio_diario = dfc.groupby(dfc['Fecha'].dt.date)['Tiempo De Interacción'].mean().reset_index()

        figure_points = px.scatter(
            df_promedio_diario,
            x='Fecha',
            y='Tiempo De Interacción',
            title='Tiempo Promedio de Interacción por Día',
            labels={'Fecha': 'Fecha', 'Tiempo De Interacción': 'Tiempo Promedio de Interacción (s)'},
            color_discrete_sequence=['#6ce5c3']
        )
        figure_points.update_traces(marker=dict(opacity=0.5))  # Menor opacidad para los puntos
        
        # Línea de tendencia (usamos scipy para una regresión lineal)
        x = np.array(pd.to_datetime(df_promedio_diario['Fecha']).map(pd.Timestamp.toordinal))
        y = df_promedio_diario['Tiempo De Interacción'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Generar la línea de tendencia
        trendline = slope * x + intercept

        # Añadir la línea de tendencia al gráfico con el color #6ce5c3
        figure_points.add_traces(
            px.line(x=df_promedio_diario['Fecha'], y=trendline, title="Línea de Tendencia", line_shape='linear').data
        )
        figure_points.update_traces(line=dict(color="#6ce5c3", width=2))  # Color de la línea de tendencia y su grosor
 
        st.plotly_chart(ut.aplicarFormatoChart(figure_points), key='chart-figure_points', use_container_width=True)

    # Contenedor D
    with Contenedor_D:
        figure_box = px.box(
            dfc,
            x=Variable_Cat,
            y='Tiempo De Interacción',
            color=Variable_Cat,
            title=f'Distribución del Tiempo de Interacción <br>{Variable_Cat.title()}',
            color_discrete_map=color_map
        )
        st.plotly_chart(ut.aplicarFormatoChart(figure_box), key='chart-figure_box', use_container_width=True)
    
    # Separador
    st.markdown("---")

    # Fila 3
    #with st.container(key='container-texto'):
        #st.markdown("### 📊 Análisis Temporal del Comportamiento")
    st.markdown("### 📊 Análisis Temporal del Comportamiento")

    dfc['Fecha'] = pd.to_datetime(dfc['Fecha'])
    dfc['Fecha Día'] = dfc['Fecha'].dt.date  # Nueva columna con solo la fecha (sin hora)
    dfc = dfc.sort_values('Fecha')

    if todos_usuarios:
        fig_scatter_all = px.scatter(
            dfc,
            x='Tiempo De Interacción',
            y='Número De Interacción Por Lección',
            size='Tiempo De Interacción',
            color='Usuario',
            title="📍 Interacciones por Usuario",
            labels={
                'Tiempo De Interacción': 'Tiempo de Interacción (s)',
                'Usuario': 'Usuario'
            },
            hover_data=['Fecha Día', 'Usuario', 'Tiempo De Interacción']
        )
        st.plotly_chart(ut.aplicarFormatoChart(fig_scatter_all), key='chart-fig_scatter_all', use_container_width=True)

    else:
        df_usuario = dfc[(dfc['Usuario'] == usuario)]
        fig_scatter_user = px.scatter(
            df_usuario,
            x='Tiempo De Interacción',
            y='Número De Interacción Por Lección',
            size='Tiempo De Interacción',
            color='Fecha Día',  # Ahora se colorea por el día (sin hora)
            title=f"📍 Interacciones del Usuario {usuario}",
            labels={
                'Tiempo De Interacción': 'Tiempo de Interacción (s)',
            },
            hover_data=['Fecha Día', 'Tiempo De Interacción']
        )
        st.plotly_chart(ut.aplicarFormatoChart(fig_scatter_user), key='chart-fig_scatter_user', use_container_width=True)

# CONTENIDO DE LA VISTA 2
if View == "Regresión Lineal":
    from sklearn.linear_model import LinearRegression
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    st.title("📈 Regresión Lineal")
    # Filtrar valores y > 0
    #df_filtrado = dfn[dfn[Variable_y] > 0].copy()
    df_filtrado = dfn

    #Modelo Refresión Lineal Simple
    model_simple = LinearRegression()
    X_simple = df_filtrado[[Variable_x]]
    y = df_filtrado[Variable_y]

    model_simple.fit(X_simple, y)
    y_pred_simple = model_simple.predict(X_simple)
    r2_simple = model_simple.score(X_simple, y)
    corr_simple = np.sqrt(r2_simple)

    # Pestañas
    tab1, tab2 = st.tabs(["📈 Análisis de Regresión Lineal", "🌡️ Heatmap"])

    # Pestaña 1
    with tab1:
        #st.subheader("📈 Análisis de Regresión Lineal")
        #Modelo Refresión Lineal Múltiple
        if Variables_x:
            X_multi = df_filtrado_multi[Variables_x]
            y_multi = df_filtrado_multi[Variable_y]

            model_multi = LinearRegression()
            model_multi.fit(X_multi, y_multi)
            y_pred_multi = model_multi.predict(X_multi)
            r2_multi = model_multi.score(X_multi, y_multi)
            corr_multi = np.sqrt(r2_multi)

        # Fila titulos de los modelos
        Contenedor_E, Contenedor_F = st.columns(2)

        # Contenedor E
        with Contenedor_E:
            #with st.container(key='container-texto3'):
            st.subheader("Regresión Lineal Simple")
        
        # Contenedor F
        with Contenedor_F:
            if Variables_x:
                #with st.container(key='container-texto4'):
                st.subheader("Regresión Lineal Múltiple")

        # Filas coeficientes de correlación de los modelos
        Contenedor_G, Contenedor_H = st.columns(2)

        # Contenedor G
        with Contenedor_G:
            st.metric(label="Coeficiente de Correlación (r)", value=f"{corr_simple:.4f}")

        # Contenedor H
        with Contenedor_H:
            if Variables_x:
                st.metric(label="Coeficiente de Correlación (r)", value=f"{corr_multi:.4f}")
        
        # Filas coeficientes de determinación de los modelos
        Contenedor_I, Contenedor_J = st.columns(2)

        # Contenedor I
        with Contenedor_I:
            st.metric(label="Coeficiente de Determinación (R²)", value=f"{r2_simple:.4f}")
        
        # Contenedor J
        with Contenedor_J:
            if Variables_x:
                st.metric(label="Coeficiente de Determinación (R²)", value=f"{r2_multi:.4f}")
        
        #Fila para mostrar los gráficos
        Contenedor_K, Contenedor_L = st.columns(2)

        # Contenedor K
        with Contenedor_K:
            fig_simple = px.scatter(
                x=X_simple[Variable_x],
                y=y,
                labels={'x': Variable_x, 'y': Variable_y},
                title="Modelo de Regresión Lineal Simple",
                color_discrete_sequence=["#6ce5c3"]
            )

        # Línea de regresión simple
            fig_simple.add_traces(go.Scatter(
                x=X_simple[Variable_x],
                y=y_pred_simple,
                mode='lines',
                name='Regresión',
                line=dict(color='#6c9fe5')
            ))

            st.plotly_chart(ut.aplicarFormatoChart(fig_simple), key='chart-fig-simple', use_container_width=True)

        # Contenedor L
        with Contenedor_L:
            if Variables_x:

                fig_multi = px.scatter(
                    x=y_pred_multi,
                    y=y_multi,
                    labels={'x': "Predicción", 'y': Variable_y},
                    title="Modelo de Regresión Lineal Múltiple",
                    color_discrete_sequence=["#6c9fe5"]
                )

                # Línea ideal de regresión (y = x) para ver ajuste predicción vs real
                fig_multi.add_trace(go.Scatter(
                    x=y_pred_multi,
                    y=y_pred_multi,
                    mode='lines',
                    name='Regresión',
                    line=dict(color='#6ce5c3')
                ))

                st.plotly_chart(ut.aplicarFormatoChart(fig_multi), key='chart-fig_multi', use_container_width=True)
            else:
                st.info("Selecciona al menos una variable independiente para el modelo múltiple.")

        # ECUACIONES DE LOS MODELOS
        slope_simple = model_simple.coef_[0]
        intercept_simple = model_simple.intercept_
        eq_simple = f"y = {slope_simple:.3f} * x + {intercept_simple:.3f}"

        if Variables_x:
            coefs_multiple = model_multi.coef_
            intercept_multiple = model_multi.intercept_
            terms = [f"{coef:.3f} * x^{i+1}" for i, coef in enumerate(coefs_multiple)]
            eq_multiple = "y = " + " + ".join(terms) + f" + {intercept_multiple:.3f}"
        else:
            eq_multiple = "Modelo múltiple no disponible"
        
        # Función para truncar ecuaciones largas
        def truncate_latex(eq, max_len=70):
            if len(eq) > max_len:
                return eq[:max_len] + " \\text{...}"
            return eq

        # Fila para mostrar ecuaciones
        Contenedor_K, Contenedor_L = st.columns(2)

        # Contenedor K
        with Contenedor_K:
            with st.container(key='container-texto'):
                st.markdown("**Ecuación Regresión Lineal Simple**")
                st.latex(eq_simple)
    
        # Contenedor L
        with Contenedor_L:
            if Variables_x:
                with st.container(key='container-texto2'):
                    st.markdown("**Ecuación Regresión Lineal Múltiple**")
                    st.latex(truncate_latex(eq_multiple))

    # Pestaña 2
    with tab2:
        #st.title("🌡️ Heatmap")
        # Correlación entre variables numéricas
        dfn_sin_fecha = dfn.drop(columns=['Fecha'])
        corr = dfn_sin_fecha.select_dtypes(include=['number']).corr()
        labels = corr.columns.tolist()
        z = corr.values
        n = len(labels)

        fig_heatmap = go.Figure()

        # Escala de colores personalizada
        colorscale = [[0, "rgb(178,34,34)"],  # rojo fuerte
                    [0.5, "white"],
                    [1, "rgb(0,0,139)"]]   # azul fuerte

        # Para mostrar la barra de colores
        fig_heatmap.add_trace(go.Heatmap(
            z=[[None]],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Correlación",
                #titleside="right",
                ticks="outside",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                thickness=15
            ),
            zmin=-1, zmax=1,
            hoverinfo='none'
        ))

        # Agrega círculos para parte inferior
        for i in range(n):
            for j in range(n):
                coef = z[i][j]
                if i > j:
                    size = np.abs(coef) * 50
                    color = f"rgba({255 if coef < 0 else 0},0,{255 if coef > 0 else 0},0.8)"
                    fig_heatmap.add_trace(go.Scatter(
                        x=[labels[j]], y=[labels[i]],
                        mode="markers",
                        marker=dict(size=size, color=color, line=dict(width=1, color='black')),
                        hoverinfo="text",
                        text=[f"{labels[i]}, {labels[j]}<br>Correlación: {coef:.2f}"],
                        showlegend=False
                    ))

        # Agrega texto para parte superior
        for i in range(n):
            for j in range(n):
                if i <= j:
                    coef = z[i][j]
                    norm_val = (coef + 1) / 2  # Normaliza a [0,1]
                    r = int(255 * (1 - norm_val))
                    b = int(255 * norm_val)
                    color = f"rgb({r},0,{b})"
                    fig_heatmap.add_trace(go.Scatter(
                        x=[labels[j]], y=[labels[i]],
                        mode="text",
                        text=[f"{coef:.2f}"],
                        textfont=dict(size=12, color=color),
                        showlegend=False
                    ))

        fig_heatmap.update_layout(
            title="🌡️ Heatmap de Correlación",
            xaxis=dict(
                tickvals=labels,
                tickangle=90,
                showgrid=True,
                zeroline=False,
                showline=True,
                mirror=True,
                categoryorder='array',
                categoryarray=labels  
            ),
            yaxis=dict(
                tickvals=labels,
                autorange="reversed",
                showgrid=True,
                zeroline=False,
                showline=True,
                mirror=True,
                categoryorder='array',
                categoryarray=labels 
            ),
            plot_bgcolor='white',
            width=750,
            height=1000,
            margin=dict(t=50, b=50, l=50, r=80)
        )
        st.plotly_chart(ut.aplicarFormatoChart(fig_heatmap), key='chart-fig_heatmap', use_container_width=True)
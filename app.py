"""
Taller Final Minería de Datos - Segmentación de clientes con K-Means
Aplicación Streamlit.  Ejecutar con:  streamlit run app.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")
FEATURES = ["Income", "Spending"]
RANDOM_STATE = 42

ESTRATEGIAS = {
    "Clientes de alto valor": "Fidelización VIP, beneficios exclusivos, productos premium y venta cruzada.",
    "Clientes promedio": "Programa de puntos, combos y promociones periódicas para subir la frecuencia de compra.",
    "Clientes con potencial": "Campañas personalizadas, experiencias exclusivas y marcas de mayor calidad para activar su gasto.",
    "Clientes de bajo valor": "Comunicación de bajo costo y ofertas de precio en temporadas de descuento.",
    "Clientes impulsivos (gasto alto, ingreso bajo)": "Promociones de tendencia, redes sociales, pagos a cuotas y descuentos para estudiantes.",
}


@st.cache_data
def cargar_datos():
    df = pd.read_csv("Mall_Customers.csv")
    return df.rename(columns={"Genre": "Gender",
                              "Annual Income (k$)": "Income",
                              "Spending Score (1-100)": "Spending"})


@st.cache_data
def metricas_k(X_scaled):
    filas = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": sil})
    return pd.DataFrame(filas)


def perfil(income, spending, df, tol=0.5):
    im, isd = df["Income"].mean(), df["Income"].std()
    sm, ssd = df["Spending"].mean(), df["Spending"].std()
    ai, bi = income > im + tol * isd, income < im - tol * isd
    ag, bg = spending > sm + tol * ssd, spending < sm - tol * ssd
    if ai and ag: return "Clientes de alto valor"
    if ai and bg: return "Clientes con potencial"
    if bi and bg: return "Clientes de bajo valor"
    if bi and ag: return "Clientes impulsivos (gasto alto, ingreso bajo)"
    return "Clientes promedio"


df = cargar_datos()
scaler = StandardScaler().fit(df[FEATURES])
X_scaled = scaler.transform(df[FEATURES])
met = metricas_k(X_scaled)

# ---------------- Barra lateral
st.sidebar.header("⚙️ Configuración")
k = st.sidebar.slider("Número de clusters (K)", 2, 10, 5)
st.sidebar.caption("K = 5 es el óptimo según el codo y la silueta.")

kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
df["Cluster"] = kmeans.labels_
centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
centroides["Perfil"] = [perfil(r.Income, r.Spending, df) for r in centroides.itertuples()]
df["Perfil"] = df["Cluster"].map(centroides["Perfil"])

st.title("🛍️ Segmentación de clientes con K-Means")
st.write("Mall Customers Dataset · variables: **ingreso anual (k$)** y **puntaje de gasto (1-100)**")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df))
c2.metric("Clusters (K)", k)
c3.metric("Coeficiente de silueta", f"{silhouette_score(X_scaled, kmeans.labels_):.3f}")

tab1, tab2, tab3 = st.tabs(["📉 Método del codo", "🎯 Clusters", "➕ Nuevo cliente"])

# ---------------- Tab 1: codo
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(met["K"], met["Inercia"], "o-", color="steelblue", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set_xlabel("K"); ax.set_ylabel("Inercia (WCSS)"); ax.set_title("Método del codo")
        ax.set_xticks(range(1, 11)); ax.legend()
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        m2 = met.dropna()
        ax.plot(m2["K"], m2["Silueta"], "o-", color="seagreen", lw=2)
        ax.axvline(k, color="red", ls="--", label=f"K seleccionado = {k}")
        ax.set_xlabel("K"); ax.set_ylabel("Silueta"); ax.set_title("Coeficiente de silueta")
        ax.set_xticks(range(2, 11)); ax.legend()
        st.pyplot(fig)
    st.dataframe(met.round(3), hide_index=True, width="stretch")

# ---------------- Tab 2: clusters
with tab2:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(data=df, x="Income", y="Spending", hue="Perfil", style="Cluster",
                    palette="Set1", s=70, ax=ax)
    ax.scatter(centroides["Income"], centroides["Spending"], c="black", s=250, marker="X", label="Centroides")
    ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Puntaje de gasto (1-100)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    st.pyplot(fig)

    resumen = df.groupby(["Cluster", "Perfil"]).agg(
        Clientes=("CustomerID", "count"), Edad_media=("Age", "mean"),
        Ingreso_medio=("Income", "mean"), Gasto_medio=("Spending", "mean")).round(1).reset_index()
    st.subheader("Resumen por cluster")
    st.dataframe(resumen, hide_index=True, width="stretch")

# ---------------- Tab 3: nuevo cliente
with tab3:
    st.subheader("Clasificar un nuevo cliente")
    with st.form("nuevo"):
        a, b = st.columns(2)
        ingreso = a.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        gasto = b.slider("Puntaje de gasto (1-100)", 1, 100, 50)
        ok = st.form_submit_button("Clasificar")

    if ok:
        nuevo = pd.DataFrame([[ingreso, gasto]], columns=FEATURES)
        cl = int(kmeans.predict(scaler.transform(nuevo))[0])
        pf = centroides.loc[cl, "Perfil"]
        st.success(f"El cliente pertenece al **Cluster {cl}** → **{pf}**")
        st.info(f"💡 Estrategia sugerida: {ESTRATEGIAS[pf]}")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.scatterplot(data=df, x="Income", y="Spending", hue="Perfil", palette="Set1",
                        s=50, alpha=0.5, ax=ax)
        ax.scatter(centroides["Income"], centroides["Spending"], c="black", s=200, marker="X")
        ax.scatter(ingreso, gasto, c="gold", edgecolor="black", s=400, marker="*", label="Nuevo cliente", zorder=5)
        ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Puntaje de gasto (1-100)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)

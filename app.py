import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", layout="wide")

FEATURES = ["Income", "Spending"]
RANDOM_STATE = 42

C_FONDO = "#F4F1EA"
C_PANEL = "#FFFFFF"
C_TEXTO = "#1E1E1E"
C_PRINCIPAL = "#1F3A5F"
C_ACENTO = "#C8553D"
C_LINEA = "#D9D4C7"

COLORES_PERFIL = {
    "Clientes de alto valor": "#1F3A5F",
    "Clientes promedio": "#8A8D91",
    "Clientes con potencial": "#C8553D",
    "Clientes de bajo valor": "#588157",
    "Clientes impulsivos (gasto alto, ingreso bajo)": "#E0A458",
}

ESTRATEGIAS = {
    "Clientes de alto valor": "Fidelización VIP, beneficios exclusivos, productos premium y venta cruzada.",
    "Clientes promedio": "Programa de puntos, combos y promociones periódicas para subir la frecuencia de compra.",
    "Clientes con potencial": "Campañas personalizadas, experiencias exclusivas y marcas de mayor calidad para activar su gasto.",
    "Clientes de bajo valor": "Comunicación de bajo costo y ofertas de precio en temporadas de descuento.",
    "Clientes impulsivos (gasto alto, ingreso bajo)": "Promociones de tendencia, redes sociales, pagos a cuotas y descuentos para estudiantes.",
}

st.markdown(f"""
<style>
* {{ border-radius: 0 !important; }}
.stApp {{ background-color: {C_FONDO}; color: {C_TEXTO}; }}
[data-testid="stSidebar"] {{ background-color: {C_PRINCIPAL}; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
[data-testid="stHeader"] {{ background-color: {C_FONDO}; }}
h1, h2, h3 {{ color: {C_PRINCIPAL}; }}
.barra {{ background-color: {C_PRINCIPAL}; color: #FFFFFF; padding: 18px 24px; margin-bottom: 18px;
         border-left: 8px solid {C_ACENTO}; }}
.barra h1 {{ color: #FFFFFF; margin: 0; font-size: 1.9rem; }}
.barra p {{ margin: 4px 0 0 0; color: #DCE3EC; }}
.dato {{ background-color: {C_PANEL}; border: 1px solid {C_LINEA}; border-top: 4px solid {C_PRINCIPAL};
         padding: 12px 16px; }}
.dato .valor {{ font-size: 1.8rem; font-weight: 700; color: {C_PRINCIPAL}; }}
.dato .etiqueta {{ font-size: 0.85rem; color: #5A5A5A; text-transform: uppercase; letter-spacing: 0.05em; }}
.seccion {{ border-bottom: 2px solid {C_PRINCIPAL}; padding-bottom: 4px; margin: 18px 0 10px 0;
            font-size: 1.15rem; font-weight: 700; color: {C_PRINCIPAL}; text-transform: uppercase; }}
.resultado {{ background-color: {C_PANEL}; border: 1px solid {C_LINEA}; border-left: 8px solid {C_ACENTO};
              padding: 14px 18px; margin-top: 10px; }}
.resultado .perfil {{ font-size: 1.3rem; font-weight: 700; color: {C_ACENTO}; }}
.stButton button, .stFormSubmitButton button {{ background-color: {C_ACENTO}; color: #FFFFFF; border: none; }}
.stButton button:hover, .stFormSubmitButton button:hover {{ background-color: {C_PRINCIPAL}; color: #FFFFFF; }}
[data-testid="stForm"] {{ background-color: {C_PANEL}; border: 1px solid {C_LINEA}; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": C_PANEL, "axes.facecolor": C_PANEL, "axes.edgecolor": C_LINEA,
    "axes.labelcolor": C_TEXTO, "xtick.color": C_TEXTO, "ytick.color": C_TEXTO,
    "axes.grid": True, "grid.color": "#ECE8DF", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10,
})


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


def dato(col, valor, etiqueta):
    col.markdown(f'<div class="dato"><div class="valor">{valor}</div>'
                 f'<div class="etiqueta">{etiqueta}</div></div>', unsafe_allow_html=True)


def seccion(texto):
    st.markdown(f'<div class="seccion">{texto}</div>', unsafe_allow_html=True)


def grafica_clusters(df, centroides, nuevo=None):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for p, g in df.groupby("Perfil"):
        ax.scatter(g["Income"], g["Spending"], s=45, color=COLORES_PERFIL[p], label=p,
                   alpha=0.45 if nuevo is not None else 0.85, edgecolor="white", linewidth=0.5)
    ax.scatter(centroides["Income"], centroides["Spending"], s=180, marker="s",
               color=C_TEXTO, label="Centroides")
    if nuevo is not None:
        ax.scatter(*nuevo, s=320, marker="D", color=C_ACENTO, edgecolor=C_TEXTO,
                   linewidth=1.5, label="Nuevo cliente", zorder=5)
    ax.set_xlabel("Ingreso anual (k$)")
    ax.set_ylabel("Puntaje de gasto (1-100)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


df = cargar_datos()
scaler = StandardScaler().fit(df[FEATURES])
X_scaled = scaler.transform(df[FEATURES])
met = metricas_k(X_scaled)

with st.sidebar:
    st.markdown("### Configuración")
    k = st.slider("Número de clusters (K)", 2, 10, 5)
    st.caption("K = 5 es el valor óptimo según el método del codo y el coeficiente de silueta.")
    st.markdown("---")
    st.markdown("### Variables usadas")
    st.markdown("Ingreso anual (k$)  \nPuntaje de gasto (1-100)")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("Mall Customers  \n200 clientes")

kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
df["Cluster"] = kmeans.labels_
centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
centroides["Perfil"] = [perfil(r.Income, r.Spending, df) for r in centroides.itertuples()]
df["Perfil"] = df["Cluster"].map(centroides["Perfil"])
sil_actual = silhouette_score(X_scaled, kmeans.labels_)

st.markdown('<div class="barra"><h1>Segmentación de clientes con K-Means</h1>'
            '<p>Taller final de Minería de Datos - Mall Customers Dataset</p></div>',
            unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns(4)
dato(d1, len(df), "Clientes")
dato(d2, k, "Clusters")
dato(d3, f"{sil_actual:.3f}", "Coeficiente de silueta")
dato(d4, f"{kmeans.inertia_:.1f}", "Inercia")

izq, der = st.columns([1, 2], gap="large")

with izq:
    seccion("Clasificar nuevo cliente")
    with st.form("nuevo"):
        ingreso = st.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        gasto = st.slider("Puntaje de gasto (1-100)", 1, 100, 50)
        ok = st.form_submit_button("Clasificar", width="stretch")

    nuevo = None
    if ok:
        nuevo = (ingreso, gasto)
        cl = int(kmeans.predict(scaler.transform(pd.DataFrame([nuevo], columns=FEATURES)))[0])
        pf = centroides.loc[cl, "Perfil"]
        st.markdown(f'<div class="resultado">Cluster {cl}<div class="perfil">{pf}</div>'
                    f'<p style="margin:8px 0 0 0">{ESTRATEGIAS[pf]}</p></div>', unsafe_allow_html=True)

with der:
    seccion("Clusters generados")
    st.pyplot(grafica_clusters(df, centroides, nuevo))

seccion("Selección del número de clusters")
g1, g2 = st.columns(2, gap="large")
with g1:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(met["K"], met["Inercia"], "s-", color=C_PRINCIPAL, lw=2)
    ax.axvline(k, color=C_ACENTO, ls="--", lw=1.5, label=f"K = {k}")
    ax.set_title("Método del codo", color=C_PRINCIPAL, loc="left", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Inercia (WCSS)"); ax.set_xticks(range(1, 11)); ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
with g2:
    m2 = met.dropna()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(m2["K"], m2["Silueta"], color=[C_ACENTO if x == k else C_PRINCIPAL for x in m2["K"]])
    ax.set_title("Coeficiente de silueta", color=C_PRINCIPAL, loc="left", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Silueta"); ax.set_xticks(range(2, 11))
    ax.set_ylim(0, m2["Silueta"].max() * 1.15)
    fig.tight_layout()
    st.pyplot(fig)

seccion("Resumen por cluster")
resumen = df.groupby(["Cluster", "Perfil"]).agg(
    Clientes=("CustomerID", "count"), Edad_media=("Age", "mean"),
    Ingreso_medio=("Income", "mean"), Gasto_medio=("Spending", "mean")).round(1).reset_index()
st.dataframe(resumen, hide_index=True, width="stretch")

with st.expander("Ver tabla de inercia y silueta por K"):
    st.dataframe(met.round(3), hide_index=True, width="stretch")

# Segmentación de clientes con K-Means

Taller final de Minería de Datos. Segmentación de los clientes del [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) con K-Means, usando el ingreso anual y el puntaje de gasto.

## Contenido

| Archivo | Descripción |
|---|---|
| `Taller_KMeans.ipynb` | Cuaderno con las 7 partes del taller (exploración, preprocesamiento, codo, silueta, modelo, interpretación y despliegue) |
| `app.py` | Aplicación de Streamlit |
| `Mall_Customers.csv` | Dataset |
| `requirements.txt` | Librerías necesarias |

## Resultados

- El método del codo y el coeficiente de silueta coinciden en **K = 5** (silueta ≈ 0.555).
- Segmentos: alto valor, promedio, con potencial, bajo valor e impulsivos.
- El segmento con mayor oportunidad de crecimiento son los **clientes con potencial** (ingreso alto, gasto bajo).

## Aplicación

La app permite ver el método del codo, ver los clusters e ingresar un nuevo cliente para clasificarlo automáticamente.

Para ejecutarla localmente:

```bash
pip install -r requirements.txt
streamlit run app.py
```

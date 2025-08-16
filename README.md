# **Telecom X — Predicción de Cancelación (Churn)**  

---

## **1) Resumen Ejecutivo**
**Objetivo:** anticipar la **cancelación de clientes (churn)** para focalizar acciones de retención con mayor ROI.  
**Enfoque:** pipeline de ML con preparación de datos, **selección de variables dentro del pipeline** (sin fugas), y comparación de **Regresión Logística (LR)** vs **Random Forest (RF)**.

**Resultados clave (holdout 30%):**
| Modelo | ROC-AUC | PR-AUC | Accuracy | Recall (churn) | Precisión (churn) | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **LR_L1_select** | **0.845** | **0.659** | 0.75 | **0.81** | 0.52 | **0.63** |
| **RF_select** | 0.824 | 0.609 | **0.78** | 0.60 | **0.60** | 0.59 |

**Interpretación ejecutiva**
- **LR** captura **más churners** (recall alto) → ideal cuando **perder un cliente es caro**.  
- **RF** genera **menos falsas alarmas** (mayor precisión/accuracy) → útil cuando el **presupuesto de retención es limitado**.  
- **Umbral operativo sugerido (LR):** ~**0.505** (max F1). Ajustar según costo/beneficio.

---

## **2) Estructura del Proyecto**
```
.
├─ notebooks/
│  └─ latam_final_2.ipynb        # cuaderno principal (EDA + modelado)
├─ data/
│  ├─ raw/                       # (opcional) datos crudos
│  └─ processed/
│     └─ df_limpo.csv            # datos tratados
├─ reports/
│  └─ figures/                   # visualizaciones (EDA, coef/imp)
├─ models/                       # (opcional) artefactos entrenados
└─ tabla_modelos_cv_holdout.csv  # (opcional) comparativa de métricas
```

---

## **3) Preparación de Datos (resumen)**
- **Categóricas → dummies** (`drop_first=True`): contrato, método de pago, servicios (internet, seguridad, soporte), etc.  
- **Numéricas:** tenure, cargos totales/mensuales.  
- **Normalización:** `StandardScaler` (para LR) en el **Pipeline**.  
- **Desbalanceo:** **SMOTE** dentro del **Pipeline** (solo en train).  
- **Split:** 70% train / 30% test estratificado + **CV 5-fold**.  
- **Selección de variables (sin fugas):** `SelectFromModel` con **LR L1** y con **RF** (threshold=`'median'`).

**Justificación de decisiones**
- Dummies + escalado mejoran modelos lineales; SMOTE y selección en CV evitan **fuga** y mejoran **recall**.  
- Métricas orientadas a desbalance (**PR-AUC**, **recall**) + ranking global (**ROC-AUC**).  
- Umbral ajustable según **estrategia comercial** (recall vs. precisión).

---

## **4) EDA — Insights Relevantes**
- **Mayor riesgo** en **contrato month-to-month**, **tenure bajo**, **cargos altos**, **Electronic check**, **sin OnlineSecurity/TechSupport**.  
- **Protectores:** **contratos 1/2 años**, **tenure alto**, **OnlineSecurity/TechSupport = Yes**.  
- Visualizaciones típicas: **heatmap de correlaciones**, barras por categoría, distribuciones por churn, **coeficientes (LR)** e **importancias (RF)**.

---

## **5) Factores que Más Influyen en el Churn**
*(confirmados por coeficientes LR e importancias RF)*  
1) **Contrato**: *month-to-month* ↑ churn; **1/2 años** ↓ churn.  
2) **Tenure**: bajo ↑, alto ↓.  
3) **Cargos** (mensuales/totales): altos ↑.  
4) **Servicios**: **sin** OnlineSecurity/TechSupport ↑; con ellos ↓.  
5) **Pago**: **Electronic check** asociado a ↑ churn.  
6) **InternetService (ej. Fiber optic)**: predictor relevante (dirección según coeficientes del modelo).

---

## **6) Recomendaciones de Retención (acción)**
**Segmentación por riesgo (probabilidad churn, p):**
- **Alto (p ≥ 0.60)**: month-to-month, tenure < 6–12m, cargos altos, sin seguridad/soporte, Electronic check.  
  - **Acciones:** oferta de **permanencia 12/24m** (descuento/bill credit), **bundle** con OnlineSecurity/TechSupport 3–6m.
- **Medio (0.40 ≤ p < 0.60)**: prueba de add-ons, migración a contrato anual con incentivo moderado, educación de valor.
- **Bajo (p < 0.40)**: comunicaciones de mantenimiento, programa de referidos.

**Experimentos A/B**  
- **Precio vs. Bundle** en segmentación de alto riesgo; medir **retención 60–90 días** y **ROI** (beneficio − costo de incentivo/contacto).

---

## **7) Ejecución Rápida**
**Requisitos**
```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib statsmodels
```

**Cargar datos**
```python
import pandas as pd
df = pd.read_csv('data/processed/df_limpo.csv')  # en Colab: '/content/df_limpo.csv'
```

**Flujo de notebook**
1. EDA → preparación (dummies/escala/SMOTE).  
2. **Pipelines con selección en CV** (LR_L1_select, RF_select).  
3. **CV 5-fold** → **Holdout** → ajuste de **umbral**.  
4. Interpretabilidad (coeficientes/imp) → conclusiones.

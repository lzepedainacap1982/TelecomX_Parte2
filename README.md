Telecom X — Predicción de Cancelación (Churn) · README
1) Propósito

Este proyecto desarrolla un pipeline de Machine Learning para predecir la cancelación (churn) de clientes de Telecom X a partir de variables demográficas, contractuales y de servicios.
El objetivo principal es priorizar acciones de retención identificando con antelación a los clientes con mayor probabilidad de cancelar.

2) Estructura del proyecto

Sugerida/estándar (lo puedes ajustar a tu carpeta actual):

.
├─ notebooks/
│  └─ latam_final_2.ipynb               # cuaderno principal (EDA + modelado)
├─ data/
│  ├─ raw/                              # (opcional) datos crudos
│  └─ processed/
│     └─ df_limpo.csv                   # datos tratados utilizados en el modelado
├─ models/
│  ├─ modelo_mejor.joblib               # (opcional) pipeline entrenado
│  └─ modelo_mejor_calibrado.joblib     # (opcional) versión calibrada
├─ reports/
│  └─ figures/                          # visualizaciones exportadas (EDA/importancias)
│     ├─ heatmap_correlaciones.png
│     ├─ coeficientes_LR.png
│     └─ importancias_RF.png
└─ tabla_modelos_cv_holdout.csv         # (opcional) tabla comparativa de métricas


Cuaderno principal: notebooks/latam_final_2.ipynb

Datos tratados: data/processed/df_limpo.csv

Visualizaciones: reports/figures/ (heatmaps, coeficientes, importancias)

Artefactos (opcional): pipelines guardados en models/

3) Preparación de datos
3.1 Clasificación de variables

Numéricas: customer.tenure, account.Charges.Total, (y/o account.Charges.Monthly, si existe), etc.

Categóricas: account.Contract, account.PaymentMethod, internet.InternetService, internet.OnlineSecurity, internet.TechSupport, phone.MultipleLines, customer.gender, customer.SeniorCitizen, etc.

3.2 Tratamiento y codificación

Unificación de categorías: "No internet service" → "No" en servicios de internet.

One-hot encoding con drop_first=True para evitar dummy trap.

Conversión segura a numérico y control de nulos (cast con errors='coerce' + SimpleImputer en el Pipeline).

3.3 Normalización / escalado

StandardScaler dentro del Pipeline para modelos lineales (LR).
RF no lo necesita, pero no afecta.

3.4 Desbalanceo

SMOTE en el train (dentro del Pipeline) para equilibrar clases y evitar fuga de información.

3.5 Partición de datos

Holdout estratificado: 70% train / 30% test, random_state=42.

Validación cruzada 5-fold estratificada para estimar desempeño fuera de muestra.

3.6 Selección de variables (sin fugas)

SelectFromModel dentro del Pipeline:

LR con L1 (saga) como selector (threshold='median')

RandomForest como selector alternativo (threshold='median')

Beneficio: la selección ocurre en cada fold, evitando leakage.

3.7 Justificación de decisiones

Codificación: variables categóricas → dummies para modelos supervisados.

Escalado: requerido para LR; neutro para RF.

SMOTE: dataset desbalanceado → mejora recall sin afectar la validación si está dentro del Pipeline.

Selección en CV: evita sobreestimar desempeño (sin usar información del test).

Métricas: se reportan ROC-AUC (ranking/global), PR-AUC (útil en desbalanceo), accuracy, precision, recall, F1.

Umbral: se ajusta con la curva Precisión–Recall según objetivo de negocio (p. ej. maximizar F1 o fijar recall mínimo).

4) EDA (gráficos e insights)

Ejemplos de visualizaciones (ver reports/figures/):

Heatmap de correlaciones: relación de Churn con tenure, cargos mensuales/totales, etc.

Distribuciones segmentadas por Churn:

Tenure: clientes con antigüedad baja presentan mayor churn.

Cargos mensuales/totales: más altos → mayor probabilidad de churn (sensibilidad a precio).

Barras por categoría:

Contrato: month-to-month >> churn que contratos 1/2 años.

OnlineSecurity / TechSupport: ausentes → mayor churn.

PaymentMethod: Electronic check suele asociarse a más churn.

Importancias / Coeficientes:

LR (coeficientes): signo y magnitud indican dirección e impacto relativo.

RF (importancias): ranking de variables predictivas.

Conclusión EDA (resumen): riesgo de churn ↑ en clientes mes a mes, tenure bajo, cargos altos, sin seguridad/soporte, y con electronic check. Contratos largos y servicios de seguridad/soporte ↓ el riesgo.

5) Modelado y resultados

Modelos evaluados (en Pipelines con selección y SMOTE):

Regresión Logística (LR) con L1 como selector

Random Forest (RF) con selector por importancia

Validación cruzada 5-fold (promedio):

LR: ROC-AUC ≈ 0.845, PR-AUC ≈ 0.66, recall alto en churn (~0.79).

RF: ROC-AUC ≈ 0.825, PR-AUC ≈ 0.62, mayor precisión/accuracy, menor recall.

Holdout (30%):

LR: ROC-AUC ≈ 0.845, PR-AUC ≈ 0.659, recall ≈ 0.81, precisión ≈ 0.52.

RF: ROC-AUC ≈ 0.824, PR-AUC ≈ 0.609, precisión ≈ 0.60, recall ≈ 0.60.

Umbral operativo (ejemplo: max F1 con LR):

Threshold ≈ 0.505 → Precisión ≈ 0.523, Recall ≈ 0.809, F1 ≈ 0.635.
(Ajustar según costo/beneficio de retención.)

6) Factores que más influyen en la cancelación

Basado en coeficientes LR e importancias RF:

Tipo de contrato: month-to-month (↑ churn) vs One/Two year (↓ churn).

Antigüedad (tenure): baja (↑), alta (↓).

Cargos: mensuales/totales altos (↑).

Servicios: sin OnlineSecurity/TechSupport (↑); tenerlos (↓).

Método de pago: Electronic check (↑) comparado con alternativas.

Otros: variables de internet/telefonía (p. ej. Fiber optic) aparecen como predictoras; su dirección se confirma con los coeficientes.

7) Estrategias de retención basadas en los hallazgos

Segmentación por riesgo (probabilidad de churn):

Alto (p ≥ 0.60): month-to-month, tenure < 6–12 meses, cargos altos, sin seguridad/soporte, Electronic check.

Acciones: oferta de permanencia (12/24 meses), descuentos escalonados, bundles con OnlineSecurity/TechSupport incluidos por 3–6 meses.

Medio (0.40 ≤ p < 0.60): educación de valor, prueba de add-ons, migración a contrato anual con incentivo moderado.

Bajo (p < 0.40): comunicaciones de mantenimiento, referidos.

Experimentos sugeridos (A/B):

Precio vs. Bundle en segmento month-to-month con tenure bajo.

Métrica: retención a 60–90 días y ROI (beneficio − costo de incentivo/contacto).

8) Cómo ejecutar el cuaderno
8.1 Requisitos

Python ≥ 3.9

pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib statsmodels

8.2 Cargar datos

Coloca df_limpo.csv en data/processed/ y ajusta la ruta en el notebook si es necesario:

import pandas as pd
df = pd.read_csv('data/processed/df_limpo.csv')   # en Colab: '/content/df_limpo.csv'

8.3 Ejecución (local o Google Colab)

Abrir notebooks/latam_final_2.ipynb.

Ejecutar las celdas en orden: EDA → Preparación → Pipeline + CV → Holdout → Umbral → Interpretabilidad → Exportación.

(Opcional) Guardar artefactos:

import joblib
# mejor_pipe: pipeline ganador (p.ej., LR_L1_select)
mejor_pipe.fit(X, y)
joblib.dump(mejor_pipe, 'models/modelo_mejor.joblib')

# Exportar tabla comparativa
tabla_modelos.to_csv('tabla_modelos_cv_holdout.csv', index=True)

9) Notas finales

La combinación de LR (recall alto) y ajuste de umbral permite maximizar retención cuando perder un cliente es costoso.

RF es útil si la prioridad es reducir falsas alarmas (presupuesto limitado).

Mantener monitoreo periódico (drift de datos y métricas) y recalibrar umbral según resultados de campañas.

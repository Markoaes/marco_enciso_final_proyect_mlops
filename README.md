# MLOps Introduction: Final Project
FInal work description in  the [final_project_description.md](final_project_description.md) file.

Student info:
- Full name: Marco Antonio Enciso Siviriche
- e-mail: marcoaes@gmail.com
- Grupo: 2

## Project Name: [Credit Default Prediction] -  Pipeline End-to-End con MLOps

## 1. Descripción del Problema

Este proyecto tiene como objetivo predecir el riesgo de incumplimiento (default) en clientes de tarjetas de crédito utilizando modelos de Machine Learning.


La predicción temprana del riesgo crediticio es fundamental para:

Reducir pérdidas financieras
Optimizar provisiones
Mejorar estrategias de cobranza
Fortalecer políticas de originación

Desde el punto de vista estadístico, el problema corresponde a una clasificación binaria donde:

0 → Cliente sin default
1 → Cliente con default


El propósito del proyecto es:

- Comparar múltiples modelos de clasificación
- Seleccionar un Modelo Campeón
- Serializar el modelo final
- Implementar seguimiento de experimentos con MLflow
- Desplegar el modelo mediante una API REST
- Validar la inferencia en tiempo real

El desarrollo sigue el ciclo completo de vida de Machine Learning bajo principios MLOps.


## 2. Dataset

Se utilizó el dataset público UCI Credit Card Default Dataset, que contiene:

- 30,000 observaciones
- 23 variables predictoras
- 1 variable objetivo

Tipos de variables:

- Demográficas (edad, educación, sexo)
- Financieras (límite de crédito)
- Historial de pagos (atrasos mensuales)
- Montos facturados y pagados

#### Descripción de Variables (Features)

Las variables predictoras pueden agruparse en cuatro grandes categorías:

Variables Demográficas

- SEX
- EDUCATION
- MARRIAGE
- AGE

Estas variables describen características socio-demográficas del cliente.

Variables Financieras

- LIMIT_BAL

Representa el límite de crédito otorgado al cliente.

Historial de Pagos

- PAY_0 a PAY_6

Indican el estado de atraso mensual (en meses anteriores).

Estas variables son altamente relevantes en la predicción de default.

Variables de Facturación y Pago

- BILL_AMT1 a BILL_AMT6
- PAY_AMT1 a PAY_AMT6

Reflejan comportamiento financiero histórico del cliente.

El conjunto de variables incluye tanto información estática como dinámica, permitiendo capturar comportamiento crediticio pasado y perfil financiero general.

El dataset presenta un desbalance moderado (~22% de defaults), lo cual fue tratado mediante partición estratificada en el split train-test.

### 3. Data Preparation

El pipeline incluye:

Limpieza básica
- Renombrado de variable objetivo
- División 80/20 estratificada
- Reproducibilidad mediante random_state=42

Se generaron visualizaciones exploratorias (EDA):

- Distribución del target
- Matriz de correlación
- Boxplots de variables numéricas
- Distribución de variables representativas


### 4. Estructura del Proyecto

data/
├── raw/
├── training/

models/
├── credit_model.pkl

notebooks/
├── 01_model_experiments.ipynb

reports/
├── api_prediction_high_risk.png
├── api_prediction_low_risk.png
├── mlflow.png
├── model_results.md

resources/images/
├── machine_learning_lifecycle.png

src/
├── data_preparation.py
├── train.py
├── serving.py

tests/
├── .gitkeep

requirements.txt
README.md
final_project_description.md
.gitignore

## 5. ML Lifecycle Aplicado

El proyecto sigue las siguientes etapas:

1. Problem Definition
2. Data Preparation
3. Model Training
4. Model Evaluation
5. Model Selection (Best Model)
6. Model Serialization
7. Model Deployment (API REST)
8. Serving & Validation

#### Arquitectura del Pipeline

El proyecto sigue una arquitectura modular y desacoplada:

- data_preparation.py → limpieza, EDA y partición reproducible
- train.py → entrenamiento, evaluación, selección automática del modelo campeón y registro en MLflow
- serving.py → despliegue del modelo vía API REST (FastAPI)
- reports/ → evidencia gráfica y análisis de resultados
- mlruns/ → almacenamiento local de experimentos MLflow

Esta arquitectura facilita:

- Reproducibilidad
- Escalabilidad
- Versionamiento experimental
- Separación clara entre entrenamiento y serving


## 6. Modelos Evaluados

Se evaluaron tres enfoques de modelado:

- Logistic Regression  
- Random Forest  
- XGBoost  

####  Justificación Metodológica

Se seleccionaron tres modelos con distintos niveles de complejidad y supuestos:

- Logistic Regression: modelo lineal interpretable, utilizado como baseline estadístico.
- Random Forest: modelo ensemble basado en árboles, robusto a no linealidades y relaciones complejas.
- XGBoost: algoritmo de gradient boosting con regularización, optimización secuencial y alto desempeño predictivo.

Esta comparación permite evaluar el trade-off entre:

- Interpretabilidad vs. capacidad predictiva
- Modelo lineal vs. modelos no lineales
- Complejidad computacional vs. desempeño

#### Métricas de Evaluación

| Modelo | ROC-AUC | KS |
|--------|---------|----|
| Logistic Regression | 0.7100 | 0.3610 |
| Random Forest | 0.7598 | 0.4065 |
| **XGBoost** | **0.7753** | **0.4266** |

#### Modelo Campeón

XGBoost fue seleccionado como modelo campeón debido a su mayor capacidad discriminatoria.
Un estadístico KS superior a 0.40 indica una adecuada separación entre clientes con y sin default, consistente con estándares aceptables en modelamiento de riesgo crediticio.


#### Interpretación de Resultados

El modelo XGBoost logró:

- ROC-AUC = 0.7753
- KS = 0.4266

Un AUC cercano a 0.78 indica buena capacidad de discriminación entre clientes con y sin default.

El estadístico KS superior a 0.42 muestra una adecuada separación entre las distribuciones acumuladas de buenos y malos clientes.

En términos prácticos, esto implica que el modelo es capaz de ordenar correctamente a los clientes según su nivel de riesgo, lo cual es fundamental en aplicaciones crediticias.


## 7. Entrenamiento del Modelo

Para entrenar el modelo:

python src/data_preparation.py
python src/train.py

Nota: El archivo del modelo no se incluye en el repositorio debido a restricciones de tamaño. Puede generarse ejecutando el script de entrenamiento.

## 8. Deployment – API REST

Para el despligue del modelo se utilizo FastAPI
para inciar el servicio se uso:python -m uvicorn src.serving:app --reload
para acceder a Swagger UI: a Swagger UI


## 9. Validación del Serving

Se realizaron pruebas vía Swagger UI confirmando:

Respuesta HTTP 200 OK
Generación correcta de probabilidades
Clasificación coherente para casos de alto y bajo riesgo
Las evidencias se encuentran en la carpeta reports/.

Ejemplo de predicción

Se realizaron pruebas sobre casos simulados:

- Caso de bajo riesgo → Probabilidad predicha < 0.30
- Caso de alto riesgo → Probabilidad predicha > 0.70

Las predicciones fueron coherentes con la lógica del modelo y consistentes con el perfil del cliente.

## 10. Entorno Reproducible

El proyecto utiliza entorno virtual: 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


## 11. Conclusiones

Se implementó un flujo completo de ML bajo principios MLOps.

Se compararon múltiples modelos.
Se seleccionó automáticamente un modelo campeón.
Se desplegó el modelo en una API REST.
Se validó correctamente la inferencia en tiempo real.
Este proyecto demuestra la integración del ciclo de vida de Machine Learning con prácticas de ingeniería de software.


### 12. Limitaciones

A pesar de los resultados obtenidos, el proyecto presenta algunas limitaciones:

- No se realizó validación temporal (out-of-time validation).
- No se implementó análisis de estabilidad poblacional (PSI).
- No se evaluó calibración formal de probabilidades.
- No se realizó análisis de interpretabilidad avanzada (SHAP).
- No se implementó monitoreo automático de drift en producción.

Estas mejoras serían necesarias para una implementación real en un entorno financiero regulado.

### 13. Mejoras Futuras

Posibles extensiones del proyecto incluyen:

- Implementar validación cruzada y tuning de hiperparámetros.
- Incorporar análisis de interpretabilidad (SHAP values).
- Implementar monitoreo de data drift.
- Registrar modelos en MLflow Model Registry.
- Implementar despliegue en contenedores Docker.
- Incorporar pruebas automatizadas en el pipeline.

### 14. Insights del Modelo

Del análisis de resultados y desempeño de los modelos se desprenden los siguientes hallazgos:

- El historial de pagos es el principal determinante del incumplimiento.
- Las variables de comportamiento financiero tienen mayor impacto predictivo que las variables demográficas.
- Los modelos no lineales (Random Forest y XGBoost) capturan mejor patrones complejos en los datos.
- La mejora significativa respecto a Logistic Regression sugiere relaciones no lineales en el comportamiento crediticio.
- El nivel de desbalance (~22%) es manejable sin necesidad de técnicas avanzadas de re-muestreo.

Estos insights permiten comprender mejor los factores asociados al default y aportan valor más allá del desempeño predictivo.

### 15. Impacto Empresarial

La implementación de un sistema automatizado de predicción de default puede contribuir a:

- Optimizar estrategias de originación crediticia.
- Reducir pérdidas esperadas (Expected Loss).
- Mejorar segmentación de riesgo para campañas de cobranza.
- Asignar provisiones basadas en riesgo real.
- Mejorar eficiencia en asignación de capital.

Desde una perspectiva organizacional, la integración de MLflow y despliegue vía API REST permite que el modelo pase de un entorno experimental a un entorno productivo con trazabilidad y gobernanza.


#### Lecciones aprendidas
Este proyecto permitió comprender que:

- Un modelo con buen AUC no es suficiente sin trazabilidad experimental.
- La reproducibilidad es un pilar fundamental en entornos reales.
- El deployment debe planificarse desde el inicio del diseño.
- La integración entre ingeniería de software y ciencia de datos es clave en MLOps.
- Este proyecto demuestra la integración efectiva entre modelamiento estadístico, ingeniería de software y prácticas de gobernanza experimental bajo un enfoque MLOps.
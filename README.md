# MLOps Introduction: Final Project
FInal work description in  the [final_project_description.md](final_project_description.md) file.

Student info:
- Full name: Marco Antonio Enciso Siviriche
- e-mail: marcoaes@gmail.com
- Grupo: 2

## Project Name: [Credit Default Prediction]

## 1. Descripción del Problema

Este proyecto tiene como objetivo predecir el riesgo de incumplimiento (default) en clientes de tarjetas de crédito utilizando modelos de Machine Learning.

Se emplea el dataset público **UCI Credit Card Default Dataset**, que contiene información demográfica, historial de pagos y comportamiento financiero de 30,000 clientes.

El objetivo es construir un modelo predictivo, evaluarlo y desplegarlo siguiendo buenas prácticas del ciclo de vida de Machine Learning (ML Lifecycle) bajo un enfoque MLOps.

---

## 2. Estructura del Proyecto

data/
├── raw/
├── training/
models/
notebooks/
reports/
src/
├── data_preparation.py
├── train.py
├── serving.py
tests/
requirements.txt
README.md

## 3. ML Lifecycle Aplicado

El proyecto sigue las siguientes etapas:

1. Problem Definition
2. Data Preparation
3. Model Training
4. Model Evaluation
5. Model Selection (Champion Model)
6. Model Serialization
7. Model Deployment (API REST)
8. Serving & Validation



## 4. Modelos Evaluados

Se evaluaron tres enfoques de modelado:

- Logistic Regression  
- Random Forest  
- XGBoost  

### Métricas de Evaluación

| Modelo | ROC-AUC | KS |
|--------|---------|----|
| Logistic Regression | 0.7100 | 0.3610 |
| Random Forest | 0.7598 | 0.4065 |
| **XGBoost** | **0.7753** | **0.4266** |

### Modelo Campeón

XGBoost fue seleccionado como modelo campeón debido a su mayor capacidad discriminatoria.
Un estadístico KS superior a 0.40 indica una adecuada separación entre clientes con y sin default, consistente con estándares aceptables en modelamiento de riesgo crediticio.

## 5. Entrenamiento del Modelo

Para entrenar el modelo:

bash
python src/data_preparation.py
python src/train.py

Nota: El archivo del modelo no se incluye en el repositorio debido a restricciones de tamaño. Puede generarse ejecutando el script de entrenamiento.

## 6. Deployment – API REST

Para el despligue del modelo se utilizo FastAPI
para inciar el servicio se uso:python -m uvicorn src.serving:app
para acceder a Swagger UI: a Swagger UI


## 7. Validación del Serving

Se realizaron pruebas vía Swagger UI confirmando:

Respuesta HTTP 200 OK
Generación correcta de probabilidades
Clasificación coherente para casos de alto y bajo riesgo
Las evidencias se encuentran en la carpeta reports/.

## 8. Entorno Reproducible

El proyecto utiliza entorno virtual: 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


## 9. Conclusiones

Se implementó un flujo completo de ML bajo principios MLOps.

Se compararon múltiples modelos.
Se seleccionó automáticamente un modelo campeón.
Se desplegó el modelo en una API REST.
Se validó correctamente la inferencia en tiempo real.
Este proyecto demuestra la integración del ciclo de vida de Machine Learning con prácticas de ingeniería de software.

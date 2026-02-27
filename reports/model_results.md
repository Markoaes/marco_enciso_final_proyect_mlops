## Desempeño Final del Modelo

Se evaluaron tres enfoques de modelado:

- Logistic Regression  
- Random Forest  
- XGBoost  

Tras la comparación de métricas de desempeño, se seleccionó **XGBoost** como Modelo Campeón.

### Métricas de Evaluación

- **ROC-AUC:** 0.7753  
- **KS (Kolmogorov-Smirnov):** 0.4266  

### Interpretación

XGBoost presentó la mayor capacidad discriminatoria entre clientes que incurren en default y aquellos que no, superando tanto a Random Forest como a Logistic Regression en las métricas ROC-AUC y KS.

El estadístico KS superior a 0.40 indica una buena separación entre las distribuciones de clientes con y sin default, lo cual es consistente con estándares aceptables en modelamiento de riesgo crediticio.

Por lo tanto, XGBoost es seleccionado para la fase de deployment dentro del ciclo de vida MLOps del proyecto.

Los resultados obtenidos validan la capacidad del pipeline implementado para soportar modelos de distinta complejidad dentro de un flujo MLOps reproducible.
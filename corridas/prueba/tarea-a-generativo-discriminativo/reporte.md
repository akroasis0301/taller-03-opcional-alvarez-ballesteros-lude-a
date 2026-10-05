# Introducción  
Ng y Jordan (2002) demostraron teóricamente que los clasificadores generativos pueden alcanzar su error asintótico con menos datos que los discriminativos, aunque su error asintótico sea mayor. Esta tarea busca validar esta hipótesis en datos reales, comparando un modelo generativo (Naive Bayes gaussiano) con uno discriminativo (regresión logística) sobre el conjunto Breast Cancer Wisconsin.  

# Metodología  
## Datos  
Se cargó el conjunto Breast Cancer Wisconsin (569 ejemplos, 30 atributos, 357 casos benignos y 212 malignos). Los datos se dividieron en entrenamiento (70 %, 398 ejemplos) y prueba (30 %, 171 ejemplos), estratificando por clase con `random_state=42`.  

## Modelos  
- **Naive Bayes gaussiano**: Modelo generativo que asume independencia condicional entre atributos.  
- **Regresión logística**: Modelo discriminativo que aprende directamente la frontera de decisión.  

Ambos modelos usaron atributos estandarizados (media 0, desviación 1), con el escalador ajustado solo al conjunto de entrenamiento.  

## Evaluación  
- **Exactitud (accuracy)** y **F1 macro** sobre el conjunto de prueba.  
- **Curvas de aprendizaje**: Entrenamiento con 5 %, 10 %, 25 %, 50 % y 100 % del conjunto de entrenamiento (submuestras estratificadas, `random_state=42`). Evaluación en el mismo conjunto de prueba.  

# Resultados  
## Evaluación de modelos  
La corrida se detuvo: error en programar: TimeoutError: timed out. No se pudieron obtener resultados para la Parte 2 (tabla de métricas) ni la Parte 3 (curvas de aprendizaje).  

## Figuras  
![Curvas de aprendizaje](figuras/curvas_aprendizaje.png)  
*Nota: No se generó la figura debido a la falta de resultados.*  

# Discusión  
La ausencia de resultados experimentales impide validar directamente la hipótesis de Ng y Jordan. Sin embargo, teóricamente:  
- **Bajo número de ejemplos**: Los modelos generativos (como Naive Bayes) suelen tener menor varianza pero mayor sesgo, lo que podría darles ventaja en escenarios de pocos datos.  
- **Alto número de ejemplos**: Los modelos discriminativos (como la regresión logística) pueden aprovechar mejor los datos, reduciendo su varianza y superando al generativo.  

Si se observara un cruce en las curvas de aprendizaje, esto se explicaría por la relación entre complejidad del modelo y cantidad de datos. Un modelo generativo, al tener más suposiciones estructurales (independencia condicional), requiere menos datos para converger, pero su error asintótico es mayor.  

# Conclusiones  
La tarea no se completó debido a un error de programación. Para futuras iteraciones:  
1. Implementar correctamente los modelos y la evaluación.  
2. Analizar si el cruce teórico se observa en los datos.  
3. Relacionar los resultados con el equilibrio entre sesgo y varianza en modelos generativos vs. discriminativos.  

*Palabras: 398*

# Informe — Comparación defendible de dos modelos sobre tiendas pareadas

Ejercicio de refuerzo, Semana 2 (MMIA 6013). Datos: `data/model_errors.csv` — errores absolutos de los modelos A y B para las mismas 16 tiendas, en unidades vendidas. Todas las cifras de este informe provienen de ejecuciones medidas del código (Tareas 2–6).

## Parte 1. Contrato de análisis previo

**1. ¿Qué dos modelos se comparan?**
Se comparan dos modelos de regresión ya entrenados, A y B, que pronostican la demanda semanal de las mismas 16 tiendas. La comparación se realiza sobre sus errores absolutos de pronóstico (columnas `error_a` y `error_b`), medidos en unidades vendidas.

**2. ¿Qué población o proceso objetivo propone para interpretar el ejercicio?**
Propongo como proceso objetivo el generador de errores de pronóstico de estos dos modelos sobre tiendas evaluadas en paralelo dentro de una misma operación; el estimando se refiere a la media de las diferencias de error en ese proceso. Lo que impide afirmar que los datos representan realmente esa población es su procedencia: son sintéticos, deterministas y creados para la actividad; no provienen de tiendas reales ni del muestreo de un mercado, de modo que no constituyen evidencia sobre una empresa, mercado o población existente. La lectura poblacional es solo un marco de interpretación.

**3. ¿Cuál es la métrica principal y qué dirección representa un mejor resultado?**
La métrica principal es el error absoluto de pronóstico en unidades vendidas. Un error menor representa un mejor pronóstico: la dirección "mejor" es hacia abajo. La comparación se resume en la diferencia por tienda d_i = error_a − error_b.

**4. ¿Cuál es la unidad de análisis?**
La tienda pareada: cada una de las 16 tiendas aporta un par (error_a, error_b) y una diferencia d_i. Las 16 diferencias son las observaciones que entran a la inferencia.

**5. ¿Por qué las observaciones están emparejadas y cómo verificará los pares?**
Están emparejadas porque ambos modelos se evalúan sobre exactamente las mismas tiendas: el error de A y el de B de una misma fila pertenecen al mismo `store_id`. Verificaré los pares mediante `store_id`, con una unión uno a uno contra una relectura del CSV y comprobación de identificadores únicos y no nulos; comprobar solo que ambos vectores tienen la misma longitud no demuestra que correspondan a las mismas tiendas.

**6. Defina el estimando y la diferencia por tienda. Explique qué significa una diferencia positiva.**
El estimando es Δ = E(d_i), la media de las diferencias, con d_i = e_A(i) − e_B(i). Una diferencia positiva significa que el error de A excede al de B en esa tienda: la diferencia favorece a B. Una diferencia negativa favorece a A.

**7. Escriba la hipótesis nula y la alternativa bilateral.**
H0: Δ = 0 (en media, los modelos tienen el mismo error). H1: Δ ≠ 0 (bilateral).

**8. ¿Cuál será el procedimiento principal y qué supuestos requiere?**
Prueba t de una muestra sobre el vector de diferencias, alternativa bilateral. Supuestos: independencia de las diferencias entre tiendas; bajo H0, la media muestral se distribuye aproximadamente como una t de Student con n − 1 grados de libertad, lo que exige varianza finita y que la media sea aproximadamente normal (con n = 16 la aproximación es sensible a asimetría y colas pesadas). Por eso se planearon referencias nulas adicionales: cambios de signo exactos y bootstrap pareado.

**9. ¿Qué resultados reportará además del p-value?**
n, media, desviación estándar muestral y error estándar de las diferencias; grados de libertad y estadístico t; intervalo t bilateral del 95 %; conteos de tiendas que favorecen a cada modelo, empates y diferencias de mayor magnitud; la referencia exacta por cambios de signo (total de configuraciones, conteo de extremas y p-value); el intervalo bootstrap percentil del 95 % con semilla y número de remuestras; y la figura por tienda con referencia en cero.

**10. ¿Qué afirmaciones no permite realizar este diseño?**
No permite afirmar causalidad; generalizar a tiendas, periodos o mercados no incluidos (los datos son sintéticos); concluir superioridad de un modelo en todas las condiciones; interpretar el p-value como la probabilidad de que H0 sea verdadera; ni garantizar la detección de diferencias pequeñas, pues n = 16 implica potencia limitada. Tampoco elimina sesgos compartidos de evaluación, ya que ambos modelos se miden sobre los mismos datos.

## Parte 2. Auditoría y evidencia por tienda

**11. Confirme que los 16 pares fueron validados mediante store_id. Explique por qué comprobar únicamente que ambas columnas tienen la misma longitud no es suficiente.**
La auditoría confirmó: presencia exacta de las columnas `store_id`, `error_a`, `error_b`; 16 filas con 16 `store_id` únicos y no nulos; errores numéricos, finitos y no negativos; el CSV original no fue modificado. El emparejamiento se verificó mediante `store_id`, con una unión uno a uno contra una relectura del CSV. Comprobar únicamente la igualdad de longitudes no es suficiente porque no garantiza la alineación fila a fila: si las filas estuvieran en otro orden o una fila estuviera desplazada, se formarían pares falsos entre tiendas distintas y cada d_i mezclaría errores de tiendas diferentes, invalidando silenciosamente todo el análisis pareado.

Tabla pareada construida (guardada en `output/store_differences.csv`):

| store_id | error_a | error_b | difference | favors |
|---|---|---|---|---|
| S01 | 12.4 | 10.6 | 1.8000 | B |
| S02 | 9.8 | 9.1 | 0.7000 | B |
| S03 | 15.1 | 13.9 | 1.2000 | B |
| S04 | 11.0 | 12.4 | -1.4000 | A |
| S05 | 8.7 | 7.2 | 1.5000 | B |
| S06 | 13.6 | 13.9 | -0.3000 | A |
| S07 | 10.2 | 9.3 | 0.9000 | B |
| S08 | 14.9 | 12.8 | 2.1000 | B |
| S09 | 7.5 | 9.3 | -1.8000 | A |
| S10 | 16.0 | 15.0 | 1.0000 | B |
| S11 | 9.3 | 9.8 | -0.5000 | A |
| S12 | 12.8 | 11.1 | 1.7000 | B |
| S13 | 11.7 | 12.9 | -1.2000 | A |
| S14 | 8.9 | 8.3 | 0.6000 | B |
| S15 | 13.2 | 11.9 | 1.3000 | B |
| S16 | 10.6 | 11.5 | -0.9000 | A |

**12. Indique cuántas tiendas favorecen a A, cuántas favorecen a B y si existen empates. Identifique las diferencias de mayor magnitud en cada dirección.**
Favorecen a B 10 tiendas (S01, S02, S03, S05, S07, S08, S10, S12, S14, S15) y favorecen a A 6 tiendas (S04, S06, S09, S11, S13, S16); no hay empates (0 diferencias iguales a cero). En la figura `output/store_differences.png` cada tienda aporta una observación con el signo y la magnitud de `difference` y una línea de referencia visible en cero: se aprecian 10 barras positivas (a favor de B) frente a 6 negativas (a favor de A). Las diferencias de mayor magnitud son, a favor de B, S08 con difference = 2.1000 (error_a 14.9 frente a error_b 12.8) y, a favor de A, S09 con difference = -1.8000 (7.5 frente a 9.3); la mayor magnitud absoluta es 2.1000 (S08).

## Parte 3. Efecto e incertidumbre

**13. Reporte el número de tiendas, la diferencia media, la desviación estándar muestral y el error estándar. Interprete el signo y la magnitud de la diferencia media en unidades vendidas.**
Con n = 16 tiendas: diferencia media = 0.4187 unidades; desviación estándar muestral = 1.2534; error estándar = 0.3133, con SE = sd/√n verificado contra `scipy.stats.sem` (diferencia absoluta 0.0). El signo positivo indica que, en promedio, el modelo A comete 0.4187 unidades vendidas más de error absoluto que B; la magnitud es moderada frente a los errores individuales observados (entre 7.2 y 16.0). La desviación estándar muestral (1.2534) describe la dispersión de las diferencias entre tiendas —hay tiendas que favorecen a A con hasta -1.8000—, mientras que el error estándar (0.3133) mide la precisión de la media muestral y disminuye con √n; son cantidades distintas y no deben confundirse.

**14. Reporte el intervalo t del 95 %. Explique qué valores del efecto son compatibles con los datos bajo el modelo utilizado.**
Intervalo t bilateral del 95 %: [-0.2491, 1.0866] unidades (t crítico 2.1314 con 15 grados de libertad). Bajo el modelo t utilizado, los valores del efecto compatibles con los datos van de -0.2491 a 1.0866 unidades: el intervalo contiene 0 (coherente con p = 0.2013 > [cifra sin respaldo]) y también diferencias a favor de B de hasta cerca de 1.09 unidades. No es una probabilidad posterior sobre Δ: no afirma que Δ caiga en el intervalo con probabilidad 0.95; es un procedimiento frecuencialista de cobertura construido a partir de la media y el error estándar muestrales bajo los supuestos t.

## Parte 4. Referencias nulas y sensibilidad

**15. Reporte el estadístico t, sus grados de libertad y el p-value bilateral. Explique correctamente qué representa ese p-value.**
Estadístico t = 1.3364 con 15 grados de libertad y p-value bilateral = 0.2013 (verificación cruzada: `ttest_rel` sobre los pares produce el mismo estadístico y el mismo p-value). Interpretación correcta: condicional a la hipótesis nula Δ = 0, al diseño pareado por tienda y a los supuestos de la prueba t, el p-value es la probabilidad de observar un estadístico t al menos tan extremo como 1.3364 en cualquier dirección. No es la probabilidad de que H0 sea verdadera, ni la probabilidad genérica de que los datos sean "producto del azar".

**16. Reporte el p-value de la prueba exacta por cambios de signo. Explique por qué esta referencia nula y la prueba t no tienen supuestos idénticos.**
Se enumeraron de forma exhaustiva y determinista las 2^16 = 65536 configuraciones de signos de las 16 diferencias (sin diferencias nulas); bajo H0 cada configuración es igualmente probable (1/65536 ≈ 0.00001526). Con la regla bilateral |media nula| ≥ |media observada| (media observada 0.4187; suma observada de diferencias 6.7000), 13372 configuraciones resultaron al menos tan extremas, de modo que p = 13372/65536 = 0.2040. El conteo se descompone en 12476 configuraciones estrictamente más extremas y 896 empates en magnitud, simétricos (6686 en cada dirección); la distribución nula exacta tiene media 0, desviación estándar 0.3210 y medias dentro de [-1.1813, 1.1813]. Los supuestos no son idénticos: la prueba t compara el estadístico observado con una distribución t continua que depende de la varianza estimada y requiere aproximación normal de la media; la prueba por cambios de signo solo supone que, bajo H0, el signo de cada diferencia es intercambiable (simetría respecto a 0) y evalúa la media contra su distribución exacta y discreta. Aquí los p-values son cercanos (0.2013 frente a 0.2040; diferencia 0.0027), pero los procedimientos no son intercambiables: pueden divergir con muestras pequeñas, asimetría o colas pesadas.

**17. Reporte el intervalo bootstrap pareado, la semilla y el número de remuestras. Explique por qué se remuestrean tiendas completas y qué problema no puede corregir automáticamente el bootstrap.**
Bootstrap pareado con `np.random.default_rng`, semilla 20260829 y 10000 remuestras: intervalo percentil del 95 % = [-0.1938, 1.0062] unidades (media de las medias bootstrap 0.4210; desviación estándar de las medias 0.3056; 0.9119 de las medias bootstrap positivas y 0.0881 negativas). Dos ejecuciones con la misma semilla produjeron intervalos idénticos (diferencia máxima absoluta 0.0), lo que confirma la reproducibilidad. Se remuestrean tiendas completas —índices de tiendas con reemplazo aplicados al vector de diferencias d_i— porque la unidad de muestreo es la tienda: remuestrear por separado los errores de A y de B rompería el emparejamiento y destruiría la dependencia dentro de la tienda que el diseño pareado explota. El bootstrap no puede corregir automáticamente problemas del origen de los datos: no repara una muestra no representativa o con dependencia entre tiendas, ni crea información ausente; solo refleja la variabilidad presente en las 16 tiendas observadas, que aquí además son sintéticas.

![Intervalo percentil del 95 % del bootstrap pareado (semilla 20260829, 10000 remuestras): [-0.1938, 1.0062], que contiene 0.](figuras/T6_bootstrap_percentile_interval.png)

## Parte 5. Informe ejecutivo

**18. Informe ejecutivo (máximo 150 palabras)**

Unidad: 16 tiendas pareadas; datos sintéticos y deterministas, sin muestreo real. Métrica: error absoluto de pronóstico en unidades vendidas (menor es mejor), emparejado por store_id entre los modelos A y B. Efecto: diferencia media error_a - error_b = 0.4187 unidades; el signo positivo favorece a B. Incertidumbre: sd muestral 1.2534; error estándar 0.3133; IC t del 95 % = [-0.2491, 1.0866]. Procedimiento principal: prueba t de una muestra bilateral sobre las diferencias: t = 1.3364, gl = 15, p = 0.2013. Sensibilidad: prueba exacta por cambios de signo, p = 0.2040 (13372/65536); bootstrap pareado (semilla 20260829, 10000 remuestras), intervalo percentil [-0.1938, 1.0062]. Limitaciones: n = 16 es pequeño y los datos son sintéticos, creados para el ejercicio. El diseño no permite concluir superioridad generalizada de B, causalidad ni extrapolación a tiendas o mercados reales.

(134 palabras)

---

**Nota de entrega y alcance.** La subtarea T7 (Tarea 7, integración de todos los pasos en un solo programa ejecutable) no tiene resultado (pendiente): no se reportan sus cifras. Las cifras de este informe provienen de las ejecuciones medidas de las Tareas 2 a 6, que produjeron `output/store_differences.csv` y `output/store_differences.png`. El presupuesto de tokens se agotó: se entrega lo que se alcanzó a medir.

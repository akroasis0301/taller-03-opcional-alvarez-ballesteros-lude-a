## Tarea 1

La clase `ServerMeasurements` almacena `values` (matriz NumPy 2D) y `feature_names` (tupla) y lanza `ValueError` en los tres casos exigidos: matriz no bidimensional, número de columnas distinto de la cantidad de nombres y datos no numéricos; las tres validaciones se ejecutaron y todas lanzan `ValueError`. `build_measurements(df)` selecciona `gpu_utilization`, `cpu_utilization` y `memory_gb` en ese orden y las convierte a matriz flotante. El `repr` obtenido es:

```
ServerMeasurements(shape=(300, 3), features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))
```

La matriz resultante tiene forma (300, 3) y tipo `float64`.

## Tarea 2

Sobre esa matriz se calcularon, por columna, las medias **[57.9931, 50.5416, 29.4553]** y las desviaciones estándar **[14.4301, 12.6985, 6.5368]**; ninguna desviación fue cero, por lo que no actuó la protección contra división por cero. La normalización Z-score, vectorizada, produjo `zscores` de forma **(300, 3)**. Las medias por columna de los z-scores tienen máximo valor absoluto **5.6843e-16** (< 1e-10) y las desviaciones por columna son exactamente **1.0** (máx |std − 1| = 0.0), cumpliendo la tolerancia exigida. `load_score = zscores @ [0.50, 0.30, 0.20]` tiene forma **(300,)** y coincide con el producto matricial con diferencia máxima **0.0**. Resumen del indicador: media **2.6053e-16**, desviación **0.8024**, mínimo **-1.9556**, máximo **3.1362**.

## Tarea 3

Se creó una copia del DataFrame original, que pasa de forma (300, 7) a **(300, 9)** al agregar `load_score` (`float64`) y `requires_review` (`bool`), esta última definida vectorizadamente como `load_score > 1.5` o `temperature_c > 80`; el DataFrame original permanece intacto y la copia coincide con él en las columnas originales. La fila 23 obtenida es:

| server | timestamp | load_score | temperature_c | requires_review |
|---|---|---|---|---|
| AI-SRV-01 | 2026-08-24 09:55:00 | 3.1362 | 66.97 | True |

Coincide con el resultado esperado (load_score 3.136218, temperatura 66.97, revisión True). En total, **49 de 300** filas (proporción 0.1633) requieren revisión: 9 por `load_score > 1.5`, 42 por `temperature_c > 80` y 2 por ambas condiciones.

## Tarea 4

Agrupando por `server` se obtuvo la tabla resumen con una fila por servidor y las columnas en el orden exigido:

| server | observations | mean_power_w | max_temperature_c | mean_load | review_count |
|---|---|---|---|---|---|
| AI-SRV-01 | 50 | 363.4598 | 86.50 | -0.614527 | 2 |
| AI-SRV-02 | 50 | 390.4990 | 76.47 | -0.417730 | 1 |
| AI-SRV-03 | 50 | 416.6088 | 86.50 | -0.085884 | 2 |
| AI-SRV-04 | 50 | 434.8684 | 80.86 | -0.003041 | 5 |
| AI-SRV-05 | 50 | 471.1808 | 86.50 | 0.453583 | 12 |
| AI-SRV-06 | 50 | 495.7276 | 88.01 | 0.667600 | 27 |

Cada servidor aporta 50 observaciones (300 en total) y la suma de `review_count` (49) coincide con las 49 filas marcadas en la Tarea 3. La fila de AI-SRV-01 reproduce el valor esperado (363.4598 W, 86.50 °C, mean_load -0.614527, 2 revisiones). Se observa un gradiente claro: AI-SRV-06, con mayor potencia media (495.7276 W) y temperatura máxima (88.01 °C), concentra la mayor carga media (0.6676) y 27 de las 49 revisiones.

## Tarea 5

El DataFrame analizado de la Tarea 3 se guardó en `output/server_analysis.parquet` con PyArrow (`pa.Table.from_pandas` con `preserve_index=False`, sin índice) y se volvió a leer. El resultado del round trip es:

```
Round trip verificado: output/server_analysis.parquet
Filas: True
Columnas: True
Esquema: True
Valores: True
```

En detalle: las filas son 300 antes y después; las columnas son 9 antes y después, con los mismos nombres (`server`, `timestamp`, `gpu_utilization`, `cpu_utilization`, `memory_gb`, `power_w`, `temperature_c`, `load_score`, `requires_review`); los dtypes son idénticos antes y después (`str`, `datetime64[us]`, cinco `float64` y `bool`), al igual que el esquema Arrow (`large_string`, `timestamp[us]`, `double` y `bool`); y los valores son iguales según `DataFrame.equals`, con tablas PyArrow idénticas y diferencia absoluta máxima de 0.0 en las columnas flotantes. La fila 23 es idéntica antes y después del round trip, y el archivo conserva las 49 filas con `requires_review=True`, consistente con la Tarea 3.

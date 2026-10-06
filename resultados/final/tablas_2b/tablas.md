# Parte 2.b — Solver completo contra la ablación

Generado por `scripts/tablas_2b.py` desde los CSV de `evaluar_solver.py` y las trazas. Con repeticiones: media (mínimo–máximo).

Corridas: **completo** = corridas/final/completo-r1, corridas/final/completo-r2; **sin_grafo** = corridas/final/sin_grafo-r1, corridas/final/sin_grafo-r2

## 1. Por tarea

| tarea | variante | comprobaciones (por repetición) | procedencia | status | subtareas no logradas | intentos de código | tokens entrada | tokens salida | duración (s) |
|---|---|---|---|---|---|---|---|---|---|
| A | completo | 9/9 · 9/9 | 1.00 · 0.96 | completado · completado | 0 | 3 | 28496 (28430–28561) | 53836 (50756–56916) | 286 (265–306) |
| A | sin_grafo | 9/9 · 9/9 | 1.00 · 1.00 | completado · completado | 0 | 3.5 (3–4) | 24096 (20059–28132) | 36168 (30143–42193) | 257 (197–318) |
| B | completo | 9/9 · 9/9 | 1.00 · 1.00 | completado · completado | 0 | 3 | 28990 (26680–31300) | 46598 (44698–48498) | 239 (223–254) |
| B | sin_grafo | 9/9 · 9/9 | 1.00 · 1.00 | completado · completado | 0 | 3 | 22850 (19630–26071) | 46453 (36692–56214) | 322 (245–400) |
| C | completo | 8/8 · 8/8 | 1.00 · 1.00 | completado · completado | 0 | 4 (3–5) | 67856 (66251–69462) | 168958 (130299–207618) | 1005 (873–1137) |
| C | sin_grafo | 7/8 · 5/8 | 1.00 · 1.00 | parcial · parcial | 2.5 (2–3) | 4 (3–5) | 33923 (26501–41345) | 192928 (61596–324259) | 1308 (462–2154) |

## 2. Totales por variante (todas las tareas)

| variante | repeticiones | comprobaciones | subtareas no logradas | intentos de código | tokens entrada | tokens salida | duración (s) |
|---|---|---|---|---|---|---|---|
| completo | 2 | 26/26 · 26/26 | 0 | 10 (9–11) | 125342 (124572–126112) | 269392 (225753–313032) | 1529 (1361–1698) |
| sin_grafo | 2 | 25/26 · 23/26 | 2.5 (2–3) | 10.5 (10–11) | 80869 (80704–81034) | 275548 (160003–391094) | 1888 (1180–2596) |

## 3. Tokens por agente (todas las tareas)

| variante | agente | llamadas | tokens entrada | tokens salida | % del total |
|---|---|---|---|---|---|
| completo | critico | 9 | 46802 (45365–48238) | 16308 (14619–17998) | 16 % |
| completo | indexador | 12 | 4788 | 29964 (26441–33487) | 9 % |
| completo | planificador | 3 | 3377 | 7164 (6689–7640) | 3 % |
| completo | programador | 11 (10–12) | 45741 (40162–51320) | 195568 (148519–242616) | 61 % |
| completo | redactor | 3.5 (3–4) | 24634 (18389–30880) | 20388 (14670–26106) | 11 % |
| sin_grafo | critico | 8.5 (8–9) | 35663 (33827–37499) | 20422 (19522–21321) | 16 % |
| sin_grafo | planificador | 3 | 3377 | 6744 (4214–9273) | 3 % |
| sin_grafo | programador | 12 (10–14) | 25640 (19705–31576) | 213476 (86550–340401) | 67 % |
| sin_grafo | redactor | 4 (3–5) | 16188 (12254–20123) | 34908 (26957–42859) | 14 % |

Los tokens de **salida** incluyen el razonamiento del modelo: el de la H200 razona siempre.

## 4. Comprobaciones que fallan (insumo de la 2.c)

| variante | tarea | check | tipo | en qué repeticiones |
|---|---|---|---|---|
| sin_grafo | C | C04 | cifra | r2: verdad=0.8611 ±0.006; no hallada |
| sin_grafo | C | C05 | cifra | r2: verdad=0.8333 ±0.006; no hallada |
| sin_grafo | C | C06 | cifra | r1: verdad=0.7500 ±0.006; no hallada / r2: verdad=0.7500 ±0.006; no hallada |

## 5. Costo fijo: el índice de las notas del curso

Se construye una vez y queda en caché; no está en las trazas de las tareas.

| clave | modelo | creado | duración (s) | tokens entrada | tokens salida | fragmentos | entidades | comunidades |
|---|---|---|---|---|---|---|---|---|
| 277d8a6398f1 | zai-org/GLM-5.3-Flash | 2026-10-04 21:39:18 | 319.6 | 37359 | 210868 | 31 | 344 | 12 |

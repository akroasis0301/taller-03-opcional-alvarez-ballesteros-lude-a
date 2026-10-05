# Evidencia — `corridas/completo-r2/tarea-C` · subtarea **T3**

- tipo: calculo · secciones: ['Parte 3'] · depende de: ['T1', 'T2']
- objetivo: Identificar por método en qué consultas falla (documento relevante fuera del top 3) y calcular los términos que comparten cada consulta con su documento relevante (y con los documentos recuperados en el top 3) para sustentar el análisis.
- criterio: Lista de consultas falladas por TF-IDF y por LSA (relevante fuera del top 3) y una tabla/listado de términos compartidos entre cada consulta y su documento relevante, obtenidos de los rankings de T1 y T2.
- status final: **pendiente**

## Archivos

- contexto del programador: `cache_solver/contextos/T3.md` (16453 car.)
- RECHAZADO `cache_solver/rechazados/T3/intento-1/`: script.py

## Eventos, en orden

- `#29` **investigador** → `grafo+capa2` · semillas=['análisis por consulta', 'documento relevante', 'top 3', 'consultas', 'términos compartidos', 'recuperación', 'ranking léxico'] · caracteres=7799: citas: ['Preámbulo', 'Parte 3', 's2-rag-y-vector-search § 5. Retrieval: Sparse, Dense, Hybrid … 5.4 Reranking', 'comunidad 1', 'comunidad 10']
- `#30` **programador** llamada: 6864 → 32768 tokens, max_tokens=32768, fin=length, 144.2 s · **error: contenido vacío (fin=length)**
- `#31` **programador** llamada (reintento 2 del cliente): 6864 → 55586 tokens, max_tokens=65536, fin=stop, 235.43 s
- `#32` **ejecutor** intento 1: returncode=None, 0.01 s, archivos=[] · **error: guarda: el código no compila: closing parenthesis '}' does not match opening parenthesis '[' (línea 236)**
- `#33` **critico** → `rechazado` · por=codigo: - guarda: el código no compila: closing parenthesis '}' does not match opening parenthesis '[' (línea 236)
- `#34` **programador** llamada: 13922 → 32768 tokens, max_tokens=32768, fin=length, 117.92 s · **error: contenido vacío (fin=length)**
- `#35` **programador** → `presupuesto_agotado`: consumido=284141 de 300000 (reserva 30000)

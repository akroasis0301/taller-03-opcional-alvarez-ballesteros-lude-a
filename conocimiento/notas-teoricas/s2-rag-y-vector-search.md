# Notas Teóricas — Semana 2: RAG y Vector Search

---

## 1. Por qué RAG existe

Los LLMs tienen conocimiento estático: lo que aprendieron durante pretraining es todo lo que saben. Problemas:
- **Cutoff date:** el modelo no conoce eventos recientes.
- **Dominio privado:** el modelo no conoce documentos internos de tu empresa.
- **Alucinación:** ante preguntas fuera de su conocimiento, el modelo inventa.

**RAG** (Retrieval-Augmented Generation) soluciona esto: en lugar de preguntar al modelo directamente, primero se recuperan documentos relevantes y luego se le pide al modelo que responda *basándose en esos documentos*.

```
Usuario: "¿Cuál es la política de graduación de la USFQ?"
           ↓
1. Recuperar chunks relevantes del corpus de documentos USFQ
2. Construir prompt: "Basándote en los siguientes documentos: [chunks], responde: ¿cuál es la política de graduación?"
3. LLM genera respuesta anclada al contexto recuperado
```

**Conexión con el curso anterior:** KRP (Knowledge Representation and Planning) enseña que los sistemas inteligentes necesitan una base de conocimiento actualizable. RAG es la implementación moderna de ese principio: la base de conocimiento vive fuera del modelo y se recupera dinámicamente.

---

## 2. Embeddings

### 2.1 Qué son

Un embedding es una función `f: texto → ℝ^d` que mapea texto a un vector de alta dimensión (e.g., d=384, d=1536) tal que textos semánticamente similares tienen vectores cercanos.

```
f("El perro corre en el parque") ≈ f("El can galopa en el jardín")  # vectores cercanos
f("El perro corre en el parque") ≠ f("La computadora calcula π")    # vectores lejanos
```

### 2.2 Cómo se entrenan

Los modelos de embeddings se entrenan con pares (texto A, texto B, similar/disimilar) usando contrastive learning:
- Minimiza la distancia entre embeddings de pares similares.
- Maximiza la distancia entre embeddings de pares disimilares.

**Sentence-BERT** (2019, Reimers & Gurevych): adaptación de BERT con Siamese networks para producir embeddings de oraciones. Estándar de la industria hasta 2023.

**Modelos actuales recomendados:**
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, rápido, open source, bueno para búsqueda general)
- `text-embedding-3-small` (OpenAI, 1536 dim, muy bueno, $0.02/1M tokens)
- `text-embedding-3-large` (OpenAI, 3072 dim, el mejor de su clase, más caro)
- `BAAI/bge-m3` (open source, multilingüe, muy bueno para español)

### 2.3 Similitud semántica

Dada una query y un conjunto de documentos embedidos, ¿cómo encontrar el más relevante?

**Similitud coseno** (más usada):
```
sim(A, B) = (A · B) / (||A|| · ||B||)
```
- Rango: [-1, 1]. 1 = idénticos, 0 = ortogonales, -1 = opuestos.
- No depende de la magnitud del vector (útil cuando los textos tienen diferentes longitudes).

**Producto punto:** equivalente a coseno si los vectores están normalizados. Más eficiente.

**Por qué no euclidiana:** en alta dimensión, las distancias euclidianas se concentran (curse of dimensionality). La similitud coseno es más robusta.

---

## 3. Vector Databases

### 3.1 El problema de escala

Con 10 documentos, búsqueda exacta (brute-force kNN) es trivial. Con 10 millones, calcular similitud contra cada vector es O(n) por query — inaceptable.

**Solución:** Approximate Nearest Neighbor (ANN) — sacrifica un poco de recall a cambio de velocidad O(log n) o O(1).

### 3.2 HNSW — Hierarchical Navigable Small World

**Intuición:** construir un grafo de proximidad jerárquico donde los nodos tienen conexiones "largas" (skip connections) en capas superiores y conexiones "cortas" en la capa base.

- **Búsqueda:** empezar en una capa superior, encontrar el vecino más cercano, bajar a la siguiente capa, refinar. Como un índice de árbol pero en espacio vectorial.
- **Construcción:** O(n log n). Búsqueda: O(log n).
- **Parámetros clave:** `M` (conexiones por nodo), `ef_construction` (tamaño del beam de construcción), `ef_search` (tamaño del beam de búsqueda).
- **Trade-off:** mayor M y ef → mayor recall, mayor uso de memoria y tiempo de construcción.

HNSW es el algoritmo por defecto en Qdrant, Weaviate y la mayoría de vector DBs.

### 3.3 IVF — Inverted File Index

**Intuición:** clusterizar los vectores (e.g., con k-means), luego en búsqueda buscar solo en los clusters más cercanos a la query.

- **Construcción:** entrenar k-means (offline). O(n·k·iter).
- **Búsqueda:** encontrar los `nprobe` clusters más cercanos, buscar dentro. O(k + n/k · nprobe).
- **Trade-off:** mayor nprobe → mayor recall, mayor latencia.

Usado frecuentemente con cuantización (IVF-PQ) para reducir memoria.

### 3.4 Stack recomendado

| Sistema | Fortaleza | Cuándo usar |
|---|---|---|
| **Qdrant** | Open source, rendimiento alto, filtros avanzados, deployment simple | Primer choice para proyectos nuevos |
| **Weaviate** | GraphQL API, módulos integrados, good multimodal | Si necesitas integración GraphQL o módulos de ML |
| **pgvector** | Extensión de PostgreSQL, no necesitas infra adicional | Si ya tienes PostgreSQL y el volumen es < 1M vectores |
| **Pinecone** | Managed, fácil de escalar | Si quieres zero ops y el costo no es problema |

---

## 4. Chunking Strategies

### 4.1 Por qué importa el chunking

Los embeddings tienen un límite de tokens de contexto (e.g., 512 tokens para MiniLM, 8192 para text-embedding-3). Si un documento tiene 50 páginas, hay que dividirlo en chunks.

**Problema:** un chunk muy grande incluye información irrelevante que "diluye" el embedding. Un chunk muy pequeño pierde contexto necesario para responder.

### 4.2 Estrategias

**Fixed-size chunking:**
```python
# Divide en chunks de N tokens con overlap de M
chunk_size = 512
overlap = 64
```
- Simple, predecible.
- No respeta límites semánticos (puede cortar una oración a la mitad).

**Sliding window:** variante de fixed-size con overlap mayor. Reduce la probabilidad de que la respuesta quede entre chunks.

**Semantic chunking:**
- Divide en párrafos o secciones naturales del documento.
- Respeta la estructura del texto.
- Más complejo: requiere detectar saltos semánticos (e.g., comparando embedding de oraciones consecutivas).

**Hierarchical chunking:**
- Mantiene chunks a múltiples granularidades: sección → párrafo → oración.
- Permite recuperar en la granularidad correcta según la query.

**Contextual chunking (Anthropic, 2024):**
- Antes de indexar, genera con el LLM un contexto que describe cada chunk en relación al documento completo.
- El chunk indexado = contexto + chunk original.
- Mejora recall significativamente pero cuesta tokens de LLM durante la ingesta (el prompt caching reduce este costo drásticamente porque el documento completo se repite por cada chunk).
- Fuente con benchmarks: anthropic.com/news/contextual-retrieval ("Introducing Contextual Retrieval").

### 4.3 Regla práctica

| Tipo de documento | Estrategia recomendada |
|---|---|
| Artículos académicos | Hierarchical (sección → párrafo) |
| Documentos legales/regulatorios | Semantic (por artículo/sección) |
| FAQ / preguntas frecuentes | Fixed-size pequeño (128-256 tokens) |
| Código fuente | Por función/clase |
| Tablas y datos estructurados | Representación como texto con contexto |

---

## 5. Retrieval: Sparse, Dense, Hybrid

### 5.1 BM25 (Sparse Retrieval)

Evolución de TF-IDF. Calcula relevancia por coincidencia de palabras clave.

```
BM25(q, d) = Σ IDF(qi) · (freq(qi,d) · (k1+1)) / (freq(qi,d) + k1·(1-b+b·|d|/avgdl))
```

**Fortalezas:** exacto en términos técnicos, siglas, nombres propios. No requiere GPU. Funciona bien para queries lexicales.

**Debilidades:** no entiende sinónimos ni paráfrasis. "Perro" y "can" son términos distintos.

### 5.2 Dense Retrieval

Embedding de la query → similitud coseno contra todos los chunks embedidos.

**Fortalezas:** entiende semántica, sinónimos, paráfrasis.

**Debilidades:** puede fallar en términos técnicos exactos, IDs, números específicos.

### 5.3 Hybrid Search

Combina BM25 + dense con **Reciprocal Rank Fusion (RRF)**:

```python
# Para cada documento, tomar el máximo de sus rankings en BM25 y dense
rrf_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
```

k=60 es el valor estándar. El resultado es más robusto que cualquiera de los dos solos.

Qdrant tiene hybrid search nativa.

### 5.4 Reranking

Después de recuperar top-20 con embedding/BM25, usar un **cross-encoder** para reordenar:
- El cross-encoder toma (query, documento) juntos y produce un score de relevancia.
- Más lento (no puede pre-calcular) pero más preciso.
- Recomendado: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```
Retrieval: embedding → top-20 (rápido)
Reranking: cross-encoder → top-5 de los 20 (más lento pero más preciso)
```

---

## 6. Evaluación de RAG

### 6.1 Métricas de retrieval

| Métrica | Qué mide |
|---|---|
| Hit Rate @k | ¿Aparece al menos 1 chunk correcto en los top-k? |
| MRR (Mean Reciprocal Rank) | Posición promedio del primer chunk correcto (1/rank) |
| Recall @k | Fracción de chunks correctos recuperados en top-k |

### 6.2 Métricas de generación (RAGAS)

| Métrica RAGAS | Qué mide |
|---|---|
| Faithfulness | ¿La respuesta está soportada por los chunks recuperados? (detecta alucinación) |
| Answer Relevancy | ¿La respuesta responde la pregunta del usuario? |
| Context Precision | ¿Los chunks recuperados son relevantes para la pregunta? |
| Context Recall | ¿Los chunks recuperados contienen la información necesaria? |

### 6.3 GraphRAG

Cuando la información está distribuida en múltiples documentos y las queries requieren razonamiento sobre relaciones entre entidades, dense retrieval falla. GraphRAG (Microsoft, 2024):

1. Extrae entidades y relaciones del corpus con LLM.
2. Construye un grafo de conocimiento.
3. Detecta comunidades en el grafo (Leiden algorithm).
4. Genera resúmenes de comunidades.
5. Para una query, busca en el grafo + en los embeddings.

**Conexión con el curso anterior:** el grafo de GraphRAG es funcionalmente equivalente a un grafo RDF con extracción automática de tripletas. Las "entidades" son nodos, las "relaciones" son predicados.

---

## 7. Notas del Lab 02

El lab construye un RAG baseline sobre documentos institucionales USFQ, papers o un corpus propio. Puntos clave a enfatizar:

- El golden set es lo más importante — sin evaluación cuantitativa, el sistema es una caja negra.
- Primero debe funcionar el baseline: ingesta, chunking, embeddings, retrieval y generación.
- Chunking es el paso que más impacta el rendimiento. La segunda estrategia de chunking es una extensión, no un bloqueo del baseline.
- Hybrid search, reranking y RAGAS son extensiones comparativas.
- RAGAS requiere un LLM para calcular faithfulness — puede ser costoso si el corpus es grande. Ejecutar una muestra pequeña antes de correr todo el golden set.

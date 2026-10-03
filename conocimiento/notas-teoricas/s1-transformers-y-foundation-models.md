# Notas Teóricas — Semana 1: Transformers y Foundation Models

> Notas de clase para elaborar. Semana 1 prioriza intuición y uso práctico. El bloque bayesiano es puente conceptual, no contenido matemático evaluado en profundidad.

---

## 1. Sesión histórica: IA antes de los LLMs (20 min)

**Mensaje central:** las técnicas clásicas de IA eran poderosas pero frágiles. Escalaron mal porque requerían ingeniería manual de conocimiento o representaciones de dominio específico.

### Línea de tiempo

- **1950s-1980s:** Sistemas expertos (MYCIN, XCON). Reglas escritas a mano. No generalizan.
- **1990s:** Algoritmos evolutivos (GA, PSO, ACO). Optimización sin gradiente. Buenos para espacios de búsqueda discretos pero no para percepción o lenguaje.
- **1990s-2000s:** HMMs, SVM, redes bayesianas. ML estadístico. Requieren features diseñadas a mano.
- **2012:** AlexNet. Deep learning supera a las técnicas anteriores en visión. El feature engineering muere para percepción.
- **2017:** Transformer (Vaswani et al.). El procesamiento secuencial deja de ser secuencial.
- **2020:** GPT-3. In-context learning emerge como paradigma. El modelo no necesita re-entrenarse para nuevas tareas.
- **2022:** ChatGPT / InstructGPT. El alignment con RLHF hace los modelos base en modelos usables.
- **2023-presente:** LLMs en producción, agentes, RAG, multimodal.

### Conexión con el curso anterior

Los estudiantes con background en el curso de IA clásica ya conocen Prolog, algoritmos de búsqueda y GA. El puente:

| Técnica clásica | Análogo moderno |
|---|---|
| Sistema experto (Prolog) | LLM con prompting estructurado |
| Búsqueda A* | Chain-of-Thought / Tree-of-Thought |
| Algoritmo genético | Optimización de prompts (DSPy, OPRO) |
| Base de conocimiento RDF/OWL | Knowledge graph en GraphRAG |
| Agente clásico (PEAS) | Agente LLM con tools |

---

## 2. Modelos Generativos Bayesianos: puente conceptual (90 min total)

### 2.1 La pregunta generativa

Todo modelo de ML responde una de dos preguntas:

- **Modelo discriminativo:** dado X, ¿cuál es Y? Aprende P(Y|X). Ejemplos: regresión logística, SVM, transformer para clasificación.
- **Modelo generativo:** ¿cómo se generan los datos X? Aprende P(X) o P(X,Y). Puede generar nuevas muestras.

Un LLM es un **modelo generativo autorregresivo**: modela P(x₁, x₂, ..., xₙ) como:

```
P(x₁,...,xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁,...,xₙ₋₁)
```

Esta es exactamente la misma pregunta que los modelos de esta sección, solo que en dimensiones y escala radicalmente mayores.

### 2.2 Teorema de Bayes como motor de inferencia

```
P(θ|X) = P(X|θ) · P(θ) / P(X)
         ──────   ──────   ────
        Posterior Verosim. Prior / Evidencia
```

- **Prior P(θ):** qué sabemos de los parámetros antes de ver los datos.
- **Verosimilitud P(X|θ):** qué tan probable es X dado el modelo con parámetros θ.
- **Posterior P(θ|X):** qué sabemos después de ver los datos.

En el contexto del LLM: θ son los pesos del modelo, X es el corpus de entrenamiento. El pretraining maximiza la verosimilitud P(X|θ) — equivalente a buscar el MAP (Maximum A Posteriori) con prior uniforme.

### 2.3 Naive Bayes — el modelo generativo mínimo

```
P(X, Y) = P(Y) · P(x₁|Y) · P(x₂|Y) · ... · P(xₙ|Y)
```

"Generativo" porque define un proceso para generar X: primero muestrea la clase Y, luego muestrea cada feature xᵢ independientemente dado Y.

**Ejemplo:** clasificador de spam.
- P(Y=spam) = frecuencia de spam en el corpus.
- P("oferta"|spam) = frecuencia de "oferta" en emails spam.

**Limitación clave:** la independencia condicional entre features raramente se cumple en texto. Un transformer aprende dependencias arbitrarias entre tokens — no las asume independientes.

### 2.4 Gaussian Mixture Models — variables latentes

```
P(X) = Σₖ πₖ · N(X | μₖ, Σₖ)
```

Introduce el concepto de **variable latente** z (de qué componente viene X, no observable):

```
P(X) = Σz P(X|z) · P(z)
```

**Algoritmo EM:**
- **E-step:** dado los parámetros actuales, calcular la responsabilidad de cada componente para cada punto: qₖ(xᵢ) = P(z=k|xᵢ).
- **M-step:** dados los pesos qₖ, actualizar μₖ, Σₖ, πₖ.
- Converge al MLE local.

**Para generar:** muestrea z ~ Categórico(π), luego x ~ N(μz, Σz).

**Conexión con LLMs:** los embeddings forman "clusters" en espacio latente. La distribución sobre el vocabulario en la capa de salida del LLM es conceptualmente una mezcla sobre todos los tokens posibles.

### 2.5 Hidden Markov Models — generación secuencial

```
P(x₁,...,xₙ) = Σₛ P(s₁) · ∏ₜ P(sₜ|sₜ₋₁) · P(xₜ|sₜ)
```

- Estado oculto sₜ: la "representación interna" en cada timestep.
- Transición P(sₜ|sₜ₋₁): cómo evoluciona el estado.
- Emisión P(xₜ|sₜ): cómo se genera la observación desde el estado.

**Para generar:** muestrea s₁, emite x₁, transiciona a s₂, emite x₂, etc.

**Aplicaciones clásicas:** reconocimiento de voz, etiquetado POS, modelado de secuencias genómicas.

**La limitación que motiva al transformer:** el estado oculto sₜ tiene dimensión fija y solo puede "recordar" información que cabe en ese vector. Para capturar dependencias de largo alcance (x₁ afecta xₙ con n grande), el HMM necesita estados enormes o pierde la dependencia.

**Conexión directa:** el transformer resuelve esto. El mecanismo de atención deja que xₙ atienda *directamente* a x₁ sin pasar por los estados intermedios.

### 2.6 Variational Autoencoder — el puente al espacio latente continuo

El VAE aprende un espacio latente continuo desde el cual puede generar nuevas muestras.

```
Codificador: qφ(z|x) → μ, σ² (distribución sobre z dado x)
Decodificador: pθ(x|z) → x reconstruido
```

**Objetivo (ELBO — Evidence Lower BOund):**
```
L = E[log pθ(x|z)] - KL[qφ(z|x) || p(z)]
     ─────────────   ─────────────────────
    Reconstrucción   Regularización (z ≈ N(0,I))
```

**Reparameterization trick:** para que el sampling sea diferenciable:
```
z = μ + σ · ε,  ε ~ N(0, I)  (ε no depende de φ)
```

**Para generar:** muestrea z ~ N(0,I), pasa por el decodificador pθ(x|z).

**Limitación en texto:** el espacio latente continuo del VAE es difícil de reconciliar con la naturaleza discreta de los tokens. Las muestras tienden a ser "borrosas" o incoherentes en texto.

**Conexión:** el VAE introduce la arquitectura encoder-decoder y el concepto de "comprimir" X en una representación latente y "descomprimir" de vuelta. Esta arquitectura aparece en los transformers encoder-decoder (T5, BART) y en los modelos de difusión.

### 2.7 Del VAE al LLM autorregresivo

| Modelo | Define | Variable latente | Limitación |
|---|---|---|---|
| Naive Bayes | P(X,Y) factorizado | Clase Y (discreta) | Independencia condicional |
| GMM | P(X) como mezcla | Componente z (discreta) | Clusters fijos; no capta estructura secuencial |
| HMM | P(X₁..Xₙ) secuencial | Estado sₜ (discreta) | Cuello de botella de estado; dependencias cortas |
| VAE | P(X) con encoder | z continuo | Muestras borrosas en texto discreto |
| **LLM** | **P(xₜ\|x₁..xₜ₋₁)** | **Ninguna explícita** | **Costo cuadrático en atención; ventana finita** |

El LLM autorregresivo no necesita una variable latente explícita: el contexto completo x₁,...,xₜ₋₁ actúa como "estado" de forma directa a través de la atención. El costo de esto es cuadrático en la longitud de la secuencia — el problema que los modelos de atención eficiente (Flash Attention, Mamba) intentan resolver.

---

## 3. Arquitectura Transformer (90 min)

### 3.1 El problema que resolvió

Las RNNs/LSTMs procesaban tokens secuencialmente. Para una secuencia de 1.000 tokens:
- Token 1000 tiene acceso directo al token 999, pero acceso muy degradado al token 1.
- No paralelizable durante el entrenamiento.
- Gradientes que desaparecen o explotan en secuencias largas.

El transformer reemplaza la recurrencia por **atención**: cualquier token puede "ver" cualquier otro token directamente, independientemente de la distancia.

### 3.2 Mecanismo de atención

**Intuición:** imagina que tienes un sistema de búsqueda documental.
- La **query** es tu consulta de búsqueda.
- Las **keys** son los índices de los documentos.
- Los **values** son el contenido de los documentos.

Para cada token de la secuencia, el modelo calcula:
1. Qué tan relevante es cada otro token (score de similitud query·key).
2. Pondera los valores de los tokens según esa relevancia.
3. El resultado es una representación del token enriquecida con contexto.

**Fórmula:**

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

- `QK^T` mide similitud entre queries y keys.
- `sqrt(d_k)` escala para evitar gradientes pequeños.
- `softmax` normaliza los scores en una distribución de probabilidad.
- El resultado es una suma ponderada de los valores.

### 3.3 Multi-head attention

En lugar de un solo mecanismo de atención, usar múltiples "cabezas" en paralelo:
- Cada cabeza aprende a atender a diferentes tipos de relaciones (sintáctica, semántica, co-referencia, etc.).
- Las salidas de todas las cabezas se concatenan y proyectan.

### 3.4 Estructura completa de un bloque transformer

```
Input
  └─> Multi-Head Attention
        └─> Add & Layer Norm
              └─> Feed-Forward Network
                    └─> Add & Layer Norm
                          └─> Output
```

- **Residual connections** (Add): suman la entrada al output de cada sublayer. Mitigan el problema del gradiente.
- **Layer Normalization:** estabiliza el entrenamiento normalizando activaciones por capa.
- **Feed-Forward Network:** dos capas densas con activación no lineal (ReLU o GeLU). Cada posición se procesa independientemente.

### 3.5 Positional encoding

La atención no tiene noción de posición (a diferencia de RNNs). Para inyectar posición:
- **Seno/coseno** (Vaswani original): funciones deterministas de posición y dimensión.
- **RoPE** (Rotary Position Embedding): usado en Llama, Mistral. Codifica posición relativa vía rotaciones.
- **ALiBi:** sesga los scores de atención según la distancia. Permite extrapolación a secuencias largas.

### 3.6 Encoder vs. Decoder

| Tipo | Atención | Uso típico |
|---|---|---|
| Encoder | Bidireccional (cada token ve todo) | Clasificación, embeddings (BERT) |
| Decoder | Causal (solo ve tokens anteriores) | Generación de texto (GPT, Llama, Claude) |
| Encoder-Decoder | Encoder bidireccional + decoder causal | Traducción, resumen (T5, BART) |

Los LLMs modernos (GPT-4, Claude, Llama) son decoder-only.

---

## 4. Ciclo de vida de un Foundation Model (60 min)

### 4.1 Pretraining

**Objetivo:** predecir el siguiente token dado el contexto anterior.

```
Input:  "El cielo es"
Target: "azul"
```

El modelo aprende a modelar la distribución P(token_t | token_1, ..., token_{t-1}).

**Por qué emerge "inteligencia":** para predecir bien el siguiente token en un corpus de internet+libros+código, el modelo necesita aprender gramática, hechos del mundo, razonamiento, código, etc. La capacidad de completar textos implica comprensión implícita.

**Escala:** GPT-3 (175B parámetros, 300B tokens). Llama 3.1 (405B parámetros, 15T tokens).

### 4.2 Supervised Fine-Tuning (SFT)

El modelo base predice texto pero no sabe seguir instrucciones. SFT entrena sobre pares (instrucción, respuesta_deseada):

```
Instrucción: "Resume este artículo en 3 puntos."
Respuesta:   "1. ... 2. ... 3. ..."
```

Transforma el modelo de "completador de texto" a "asistente que sigue instrucciones".

### 4.3 RLHF — Reinforcement Learning from Human Feedback

**Problema de SFT:** los humanos son buenos anotando pero los pares (instrucción, respuesta) son costosos de producir en cantidad. Además, SFT imita, no razona sobre qué es mejor.

**Solución RLHF:**
1. Genera múltiples respuestas para la misma instrucción.
2. Anotadores humanos las rankean (A > B).
3. Entrena un **reward model** para predecir el ranking humano.
4. Usa PPO (Proximal Policy Optimization) para optimizar el LLM contra el reward model.

**Limitación:** RLHF requiere miles de horas de anotación humana.

### 4.4 DPO — Direct Preference Optimization

Alternativa más eficiente a RLHF: dado un par (respuesta ganadora, respuesta perdedora), optimiza directamente los pesos del LLM sin reward model explícito. Más estable y más simple.

### 4.5 Constitutional AI (Anthropic)

En lugar de feedback humano masivo, define un conjunto de **principios** (constitución). El modelo:
1. Genera una respuesta.
2. Se auto-critica según los principios.
3. Revisa la respuesta hasta que sea harmless y helpful.

Permite escalar alignment con menos intervención humana.

### 4.6 Inference: temperatura y sampling

Dado el vector de logits del modelo, ¿cómo elegir el siguiente token?

- **Temperatura = 0:** siempre el token más probable. Determinista. Útil para código, extracción de datos.
- **Temperatura = 1:** distribución de probabilidad original del modelo.
- **Temperatura > 1:** más aleatorio, más creativo, más propenso a errores.
- **Top-p (nucleus sampling):** considera solo los tokens cuya probabilidad acumulada es ≤ p. Balance creatividad/coherencia.
- **Top-k:** considera solo los k tokens más probables.

**En producción:** temperature=0 para tareas deterministas (extracción, clasificación), temperature=0.7-1.0 para generación creativa.

---

## 5. In-context Learning y Prompting (60 min)

### 5.1 In-context learning

El modelo puede "aprender" de ejemplos en el prompt sin actualizar sus pesos. Esto emerge a cierta escala (>~7B parámetros).

**Por qué funciona:** el mecanismo de atención permite al modelo identificar patrones en los ejemplos del contexto y extrapolar al nuevo input. No es aprendizaje en el sentido de gradientes — es inferencia bayesiana implícita.

### 5.2 Chain-of-Thought

En lugar de pedir la respuesta directamente:

```
Q: Si tengo 5 manzanas y doy 2, ¿cuántas me quedan?
A: Empiezo con 5. Doy 2. 5 - 2 = 3. Me quedan 3 manzanas.
```

CoT mejora significativamente en tareas de razonamiento multi-paso. El modelo "muestra su trabajo" y eso reduce errores intermedios.

**Variantes:**
- **Zero-shot CoT:** "Piensa paso a paso." en el prompt es suficiente en modelos grandes.
- **Few-shot CoT:** proveer ejemplos completos de razonamiento.

### 5.3 Structured outputs

Para integrar LLMs en sistemas, necesitamos outputs predecibles:

```python
# OpenAI JSON mode
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Responde en JSON con campos: nombre, edad, ciudad."},
        {"role": "user", "content": "Mi nombre es Ana, tengo 30 años y vivo en Quito."}
    ]
)
```

**Pydantic + instructor:** librería que fuerza salidas estructuradas con validación de tipos.

---

## 6. Notas del Lab 01

El lab de esta semana compara tres modelos. Puntos clave a enfatizar:

- El costo de los modelos propietarios varía ~10-100x entre modelos pequeños y grandes.
- La calidad no es proporcional al costo para todos los casos de uso.
- Para muchas tareas (extracción estructurada, clasificación, resumen), un modelo de 7B bien promoteado compite con GPT-4.
- La latencia importa: en aplicaciones interactivas, 5 segundos de respuesta puede ser inaceptable.

**API Keys para el lab:**
- Usar `.env`, nunca pegar claves en notebooks o PDFs.
- Verificar modelos y precios en `../fuentes/modelos/modelos-2026-1.*`.
- Open-weight: usar Ollama localmente cuando el hardware lo permita.

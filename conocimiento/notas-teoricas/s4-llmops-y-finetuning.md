# Notas Teóricas — Semana 4: LLMOps y Fine-Tuning Ligero

---

## 1. Por qué LLMOps es diferente a MLOps

En ML clásico, una vez que el modelo está entrenado y desplegado, el pipeline es relativamente estable: los datos de entrada tienen un esquema fijo, la métrica de evaluación es clara (accuracy, RMSE), y el modelo no cambia a menos que se re-entrene.

Los LLMs en producción son diferentes:
- Las **entradas son texto libre** — el espacio de inputs posibles es infinito.
- **No hay una métrica única de calidad** — "buena respuesta" depende del contexto.
- Los **prompts son código** y pueden cambiar independientemente del modelo.
- El **costo es por token** — un prompt mal diseñado puede ser 10x más caro que uno bien diseñado.
- Los modelos base **se actualizan** con frecuencia, cambiando el comportamiento sin aviso.

LLMOps cubre las prácticas para manejar estos sistemas de forma sostenible.

---

## 2. Evaluación Sistemática

### 2.1 Golden Sets

Un golden set es un conjunto de pares (input, output esperado) construido manualmente por expertos del dominio.

**Construcción:**
1. Identificar los casos de uso críticos del sistema.
2. Construir 50-200 pares representativos. Incluir:
   - Casos "felices" (el sistema debería responder bien)
   - Casos límite (queries ambiguas, información parcial)
   - Casos adversariales (queries diseñadas para confundir o hacer alucinar)
3. Documentar el criterio de éxito para cada caso.

**Mantenimiento:**
- Actualizar cuando cambien los requisitos del sistema.
- Agregar casos de fallo reales detectados en producción.
- El golden set es un activo de valor — versionarlo en git.

**Anti-patrones:**
- Golden set construido por el mismo equipo que construye el sistema (sesgo de confirmación).
- Golden set demasiado pequeño (<20 casos) — no es estadísticamente significativo.
- Golden set que no cubre casos adversariales — da falsa confianza.

### 2.2 LLM-as-Judge

En lugar de evaluación humana (cara y lenta), usar otro LLM como juez:

```python
JUDGE_PROMPT = """
Evalúa la calidad de la siguiente respuesta a la pregunta dada.

Pregunta: {question}
Respuesta a evaluar: {answer}
Respuesta de referencia: {reference}

Criterios:
1. Exactitud (1-5): ¿La respuesta es factualmente correcta?
2. Completitud (1-5): ¿La respuesta cubre todos los aspectos de la pregunta?
3. Concisión (1-5): ¿La respuesta es apropiadamente concisa?

Responde en JSON: {{"exactitud": X, "completitud": Y, "concision": Z, "justificacion": "..."}}
"""
```

**Sesgos conocidos de LLM-as-judge:**
- **Verbosity bias:** modelos grandes prefieren respuestas más largas aunque no sean mejores.
- **Self-enhancement bias:** el modelo prefiere sus propias respuestas.
- **Position bias:** en comparaciones, el modelo tiende a preferir la primera respuesta presentada.

**Mitigaciones:**
- Usar un modelo diferente al evaluado como juez.
- Evaluar cada respuesta por separado (no comparar A vs. B directamente).
- Incluir criterios específicos en el prompt del juez.
- Calibrar el juez contra evaluaciones humanas en una muestra pequeña.

### 2.3 RAGAS para RAG

RAGAS proporciona un framework de evaluación específico para sistemas RAG:

| Métrica | Cómo se calcula |
|---|---|
| **Faithfulness** | ¿Cada claim en la respuesta está soportado por el contexto recuperado? (LLM verifica) |
| **Answer Relevancy** | ¿La respuesta responde la pregunta? (embedding de la pregunta vs. embedding de la respuesta) |
| **Context Precision** | ¿Los chunks recuperados son relevantes para la pregunta? (LLM clasifica cada chunk) |
| **Context Recall** | ¿El contexto recuperado contiene la información necesaria para responder? |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=my_dataset,  # HuggingFace Dataset con question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 3. Observabilidad

### 3.1 Por qué es no negociable

En un sistema con 1.000 consultas diarias, el 5% de fallos = 50 fallos por día. Sin trazas, no tienes forma de:
- Identificar qué tipo de consultas fallan más.
- Saber cuánto cuesta cada tipo de consulta.
- Detectar que un cambio de prompt empeoró el rendimiento.
- Auditar una respuesta específica que generó un problema.

### 3.2 Modelo de trazabilidad

Una **traza** es el registro completo de una ejecución. Puede vivir en una plataforma como Langfuse o en archivos locales JSON/JSONL si el sistema todavía está en prototipo:
- Tiene una ID única.
- Contiene múltiples **spans** (llamadas al LLM, llamadas a tools, pasos del pipeline).
- Cada span registra: input, output, duración, costo (tokens), errores.

```
Traza: user_query_12345
  ├── Span: embed_query (15ms, 0 tokens)
  ├── Span: vector_search (45ms, 0 tokens)
  ├── Span: llm_call (1200ms, 850 tokens input, 320 tokens output, $0.0018)
  └── Span: guardrail_check (8ms, 0 tokens)
  Total: 1268ms, $0.0018
```

### 3.3 Langfuse

Langfuse es la herramienta recomendada — open source, self-hosteable, y tiene una capa gratuita. Para el baseline del lab, una traza local bien estructurada es aceptable; para proyecto final se recomienda dashboard.

**Instrumentación básica:**

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse()

@observe()
def my_rag_pipeline(question: str) -> str:
    # Langfuse captura automáticamente inputs, outputs y duración
    chunks = retrieve(question)
    answer = generate(question, chunks)
    return answer
```

**Lo que registra automáticamente:**
- Input y output de cada función decorada.
- Duración.
- Metadatos de modelos (si usa LangChain, registra tokens y costo).

### 3.4 Prompt Versioning

Los prompts son código. Deben versionarse como código:

**Opción A — En el repositorio:** prompts como archivos `.txt` o `.md` versionados en git. Simple, no requiere infra adicional.

**Opción B — Langfuse Prompt Management:** interfaz web para versionar prompts. Permite actualizar prompts sin re-deploy. Registra qué versión de prompt generó qué respuesta.

**Por qué importa:** si cambias el system prompt y el rendimiento baja, necesitas saber exactamente qué cambió y poder hacer rollback.

---

## 4. Guardrails

### 4.1 Input validation

Antes de enviar el input al LLM, verificar:

- **Prompt injection:** detectar intentos de manipular el system prompt.
  ```python
  INJECTION_PATTERNS = ["ignore previous instructions", "forget your instructions", "you are now"]
  if any(p in user_input.lower() for p in INJECTION_PATTERNS):
      return "Lo siento, no puedo procesar esa solicitud."
  ```

- **PII filtering:** detectar y redactar información personal antes de enviarla al LLM (especialmente si usas APIs externas).
  ```python
  import presidio_analyzer  # Microsoft Presidio
  # Detecta emails, teléfonos, números de tarjeta, etc.
  ```

### 4.2 Output validation

Después de recibir la respuesta del LLM, verificar:

- **Formato correcto:** si esperabas JSON, validar que sea JSON válido.
- **Toxicidad:** usar un clasificador de toxicidad (OpenAI Moderation API, o Detoxify).
- **Datos sensibles:** verificar que la respuesta no incluya información que no debería aparecer.

### 4.3 Costo y latencia

**Prompt caching (Anthropic/OpenAI):** si el system prompt es largo y se repite en muchas llamadas, el proveedor puede cachearlo y cobrar menos tokens.

```python
# Anthropic: usar cache_control en el system prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": very_long_system_prompt, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_question}
        ]
    }
]
```

**Semantic caching:** si dos queries son semánticamente equivalentes, retornar la respuesta cacheada.

```python
# GPTCache o implementación propia con vector DB
# Si embed(nueva_query) ≈ embed(query_previa), retornar respuesta_previa
```

---

## 5. Fine-Tuning Ligero

### 5.1 Por qué no full fine-tuning

Un modelo de 7B parámetros en float32 ocupa ~28 GB en memoria solo para los pesos. El fine-tuning requiere gradientes y estados del optimizador: ~4-6x el tamaño del modelo → >100 GB. Inasequible para la mayoría.

### 5.2 LoRA — Low-Rank Adaptation

**Idea clave:** en lugar de actualizar todos los pesos W de una capa, agregar una perturbación de bajo rango:

```
W_nuevo = W_original + ΔW
ΔW = A · B   donde A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r << min(d,k)
```

Solo se entrenan A y B. Si r=8 y d=k=4096, ΔW tiene 4096² = 16.7M parámetros, pero LoRA entrena 2 × 4096 × 8 = 65.536 parámetros. Reducción de ~256x.

**Parámetros de LoRA:**
- `r` (rango): cuántos parámetros aprende. Valores típicos: 4, 8, 16, 32. Mayor r → más capacidad pero más memoria.
- `alpha` (escala): factor de escala para ΔW. Típicamente alpha = r o 2r.
- `target_modules`: qué capas aplicar LoRA. Típicamente las capas de atención (q_proj, v_proj).

### 5.3 QLoRA — Quantization + LoRA

LoRA sobre un modelo cuantizado a 4-bit (NF4):
- El modelo base ocupa ~4 GB en lugar de ~28 GB para 7B parámetros.
- LoRA se entrena en float16/bfloat16 (adaptadores pequeños).
- Hace viable fine-tuning en una GPU de 16 GB (RTX 3080/4080, Colab T4).

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Cargar modelo cuantizado
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", quantization_config=bnb_config)

# Aplicar LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

### 5.4 Árbol de decisión: ¿cuándo usar qué?

```
¿El modelo base puede resolver la tarea con el prompt correcto?
  ├─ SÍ → Mejor prompt (más barato, más flexible)
  └─ NO → ¿El modelo falla por falta de conocimiento actualizado?
            ├─ SÍ → RAG (más flexible, conocimiento actualizable)
            └─ NO → ¿El modelo falla por estilo/tono/formato específico?
                      ├─ SÍ → Fine-tuning ligero (LoRA/QLoRA)
                      └─ NO → ¿El modelo necesita dominio muy especializado con datos privados?
                                ├─ SÍ → Fine-tuning + RAG
                                └─ NO → Revisar el problema
```

**Regla práctica:** intenta prompting → RAG → fine-tuning, en ese orden. En este curso el fine-tuning se evalúa como decisión de arquitectura y demo guiada, no como requisito de entrenamiento completo.

---

## 6. Monitoreo en Producción

### 6.1 Señales a monitorear

| Señal | Qué detecta | Umbral de alerta |
|---|---|---|
| Latencia p95 | Degradación del servicio | >3x latencia normal |
| Costo por query | Budget descontrolado | >2x costo normal |
| RAGAS faithfulness | Alucinación creciente | <0.7 |
| Tasa de rechazo por guardrails | Intentos de abuso | Pico súbito |
| Satisfacción del usuario | Calidad percibida | Si se recolecta (👍/👎) |

### 6.2 Drift de calidad

El comportamiento del LLM puede cambiar sin que tú hagas nada:
- El proveedor actualiza el modelo subyacente.
- El corpus del RAG se desactualiza.
- La distribución de queries cambia (concept drift).

**Mitigación:** ejecutar el golden set automáticamente cada semana y alertar si las métricas bajan más del 5%.

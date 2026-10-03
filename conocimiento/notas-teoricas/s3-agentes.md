# Notas Teóricas — Semana 3: Agentes LLM

---

## 1. Puente con la teoría clásica de agentes

### 1.1 Agentes clásicos (Russel & Norvig)

El curso anterior definió un agente como una entidad que:
- **Percibe** el entorno a través de sensores.
- **Actúa** sobre el entorno a través de actuadores.
- Tiene una **función de agente** que mapea percepciones a acciones.

La taxonomía PEAS: Performance measure, Environment, Actuators, Sensors.

Tipos de agentes:
1. **Reactivo simple:** reglas condición-acción. Sin memoria.
2. **Basado en modelos:** mantiene un estado interno del mundo.
3. **Basado en objetivos:** planifica hacia una meta.
4. **Basado en utilidad:** maximiza una función de utilidad.

### 1.2 El agente LLM en la taxonomía clásica

Un agente LLM moderno es esencialmente un **agente basado en objetivos** con componentes adicionales:

| Componente clásico | Equivalente en agente LLM |
|---|---|
| Sensores | Input del usuario + outputs de tools |
| Actuadores | Llamadas a tools + respuesta al usuario |
| Estado interno | Memoria de conversación (context window) |
| Planificación | CoT + loop ReAct / Plan-and-Execute |
| Función de agente | El LLM mismo |

**Diferencia clave:** el agente clásico tiene una función de agente explícita (código). El agente LLM tiene una función de agente *emergente* del LLM, que puede razonar sobre tareas no vistas previamente.

---

## 2. Tool Use / Function Calling

### 2.1 Qué es

El LLM puede emitir una llamada estructurada a una función externa como parte de su respuesta:

```python
# Definición del tool
tools = [
    {
        "name": "execute_sql",
        "description": "Ejecuta una consulta SQL sobre la base de datos y retorna los resultados.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La consulta SQL a ejecutar"}
            },
            "required": ["query"]
        }
    }
]

# Respuesta del modelo (puede incluir una llamada al tool)
# El modelo emite: {"tool": "execute_sql", "input": {"query": "SELECT * FROM ventas WHERE mes='enero'"}}
# El framework ejecuta la función y devuelve el resultado al modelo
```

### 2.2 Ciclo de tool use

```
1. Usuario envía mensaje
2. LLM decide qué tool usar (o responder directamente)
3. Framework ejecuta el tool
4. Resultado del tool se añade al contexto
5. LLM genera respuesta final (o decide usar otro tool)
```

### 2.3 Model Context Protocol (MCP)

Estándar abierto de Anthropic (2024) para conectar LLMs a herramientas externas de forma estandarizada:
- Define un protocolo cliente-servidor para exponer tools a cualquier LLM compatible.
- Elimina la necesidad de adaptar cada herramienta a cada LLM.
- Adoptado por Claude, próximamente por otros modelos.

---

## 3. Patrones de Agentes

### 3.1 ReAct (Reason + Act)

**Paper:** Yao et al. (2023).

El agente alterna entre pensamiento (Thought) y acción (Action), observando el resultado:

```
Thought: Necesito saber las ventas del último mes. Voy a consultar la base de datos.
Action: execute_sql("SELECT SUM(monto) FROM ventas WHERE mes='2026-05'")
Observation: 45230.50

Thought: El total de ventas de mayo es $45,230.50. Ahora necesito compararlo con abril.
Action: execute_sql("SELECT SUM(monto) FROM ventas WHERE mes='2026-04'")
Observation: 38910.00

Thought: Tengo los datos. Las ventas de mayo superaron a las de abril en $6,320.50 (16.2%).
Respuesta: Las ventas de mayo 2026 fueron $45,230.50, un 16.2% más que abril ($38,910.00).
```

**Ventaja:** el modelo "muestra su trabajo" — el trace es auditable.
**Limitación:** si el modelo comete un error en un Thought, puede propagarse.

### 3.2 Plan-and-Execute

1. **Fase de planificación:** el LLM genera un plan detallado de pasos antes de ejecutar.
2. **Fase de ejecución:** un segundo agente (o el mismo) ejecuta cada paso del plan.

```
Plan:
  1. Obtener ventas de mayo 2026
  2. Obtener ventas de abril 2026
  3. Calcular la diferencia y el porcentaje
  4. Generar un gráfico de barras comparativo
  5. Redactar el análisis

Ejecutar paso 1: [tool call]
Ejecutar paso 2: [tool call]
...
```

**Ventaja:** más robusto para tareas largas con muchos pasos. El plan puede revisarse.
**Limitación:** si el plan inicial es incorrecto, todos los pasos fallan.

### 3.3 Reflexion

El agente evalúa su propia respuesta y la revisa si es insatisfactoria:

```
Respuesta inicial: "Las ventas aumentaron."
Crítica (auto-generada): "La respuesta es muy vaga. No incluye números específicos ni comparación."
Respuesta revisada: "Las ventas de mayo fueron $45,230.50, un 16.2% más que abril."
```

Se puede iterar N veces hasta que la respuesta sea satisfactoria (con un límite para evitar loops).

---

## 4. Multi-Agent Systems (extensión)

### 4.1 Por qué multi-agente

Multi-agente se presenta como extensión avanzada. El baseline del curso es un agente con tools, trazas y límites de ejecución. Un solo agente tiene limitaciones:
- La ventana de contexto es finita.
- Un agente generalista puede ser peor que varios agentes especializados.
- Las tareas largas se benefician de paralelismo.

### 4.2 Supervisor Pattern

```
Usuario → Supervisor Agent
            ├── Data Agent (acceso a datos, SQL, APIs)
            ├── Analysis Agent (estadísticas, visualizaciones)
            └── Report Agent (redacción, formato)
```

El supervisor decide qué subagente activa, pasa el contexto relevante, y consolida las respuestas.

### 4.3 LangGraph

Framework de LangChain para construir sistemas de agentes como grafos de estado. En el lab, LangGraph es recomendado si simplifica la solución; no es obligatorio para aprobar el baseline.

- **Nodos:** agentes, tools, funciones de procesamiento.
- **Edges:** transiciones condicionales entre nodos.
- **State:** dict compartido que todos los nodos pueden leer y escribir.

```python
from langgraph.graph import StateGraph, START, END  # langgraph==1.2.12
# AgentState: un TypedDict con las claves del estado compartido
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("data_agent", data_agent)
workflow.add_node("report_agent", report_agent)
workflow.add_edge(START, "supervisor")  # sin punto de entrada, compile() falla
workflow.add_conditional_edges("supervisor", route_to_agent,
    {"data_agent": "data_agent", "report_agent": "report_agent"})
workflow.add_edge("data_agent", "supervisor")
workflow.add_edge("report_agent", END)
app = workflow.compile()  # y se ejecuta con app.invoke(estado_inicial)
```

### 4.4 Hand-offs

Los agentes se pasan el control entre sí con el contexto relevante. El hand-off incluye:
- El resultado parcial del agente anterior.
- El objetivo del siguiente paso.
- Cualquier restricción o preferencia.

---

## 5. Riesgos y Mitigaciones

### 5.1 Loops infinitos

Un agente puede entrar en un loop si ningún tool retorna el resultado esperado.

**Mitigación:** límite máximo de iteraciones (e.g., max_steps=10). Si se alcanza, el agente responde con lo que tiene y explica que no pudo completar la tarea.

### 5.2 Costos descontrolados

Cada iteración del loop consume tokens. Un agente con 20 iteraciones y GPT-4o puede costar $5-10 por ejecución.

**Mitigación:** monitoreo de tokens en tiempo real. Corte automático si se supera un budget.

### 5.3 Herramientas peligrosas

Un tool de `execute_python_code` o `delete_file` puede causar daño irreversible.

**Mitigación:**
- Human-in-the-loop: el agente pide confirmación antes de acciones destructivas.
- Sandboxing: el código se ejecuta en un ambiente aislado.
- Principio de mínimo privilegio: cada tool tiene los permisos mínimos necesarios.

### 5.4 Prompt injection en herramientas

Un documento recuperado puede contener instrucciones para el LLM que subviertan el comportamiento esperado.

**Mitigación:** separar claramente el contexto del usuario del contenido de herramientas. Usar system prompts robustos que instruyan al modelo a ignorar instrucciones en el contenido recuperado.

---

## 6. Conexión con MSDS 6020 (Ética)

Los temas de esta semana tocan directamente la agenda de ética:
- **Responsabilidad:** cuando un agente autónomo toma una decisión incorrecta, ¿quién es responsable?
- **Transparencia:** los traces del agente son el mecanismo de auditabilidad.
- **Autonomía controlada:** el human-in-the-loop es la respuesta técnica a los riesgos de agencia.
- **Herramientas con impacto real:** un agente que puede enviar emails, ejecutar código o modificar bases de datos tiene impacto en el mundo real.

Coordinar con MSDS 6020 para no repetir sino referenciar.

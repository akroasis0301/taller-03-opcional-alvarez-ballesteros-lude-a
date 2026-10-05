# Parte 3 — Los cuatro frenos, forzados

Generado por `scripts/forzar_frenos.py` con un modelo de guion (sin H200): los frenos, el sandbox y la traza son los del solver real. Cada carpeta tiene su `traza.jsonl` completa.

## presupuesto — disparado

`corridas/frenos/presupuesto/` · status **parcial** · subtareas {'T1': 'aprobada (1 intentos)', 'T2': 'pendiente (0 intentos)'} · entregables ['reporte.md'] · tokens {'tokens_entrada': 800, 'tokens_salida': 400}

- `#13` programador → **presupuesto_agotado** [T2]: consumido=900 de 1500 (reserva 600)
- `#14` orquestador → **presupuesto_agotado** []: programador: presupuesto agotado (900 tokens)
- `#15` redactor redacta (200+100 tokens); recibió la orden de declarar: «- La subtarea T2 (Parte 2) no tiene resultado (pendiente): no se reportan sus cifras.
- El presupuesto de tokens se agotó: se entrega lo que se alcanzó a medir.»

## repeticion — disparado

`corridas/frenos/repeticion/` · status **parcial** · subtareas {'T1': 'aprobada (1 intentos)', 'T2': 'fallida (3 intentos)'} · entregables ['reporte.md'] · tokens {'tokens_entrada': 7000, 'tokens_salida': 1400}

- `#15` critico → **rechazado** [T2]: - plausibilidad: métricas ≥ 0,999 en un problema con ruido: ['accuracy=1.0']
- fuga: se evalúa sobre las mismas variables con que se ajustó: ['X']
- `#17` ejecutor → **script_repetido** [T2]: idéntico (1f1ca7597eb7285e) a uno ya rechazado: no se ejecuta
- `#18` critico → **rechazado** [T2]: - script idéntico a uno ya rechazado: no se ejecutó
- `#20` ejecutor → **script_repetido** [T2]: idéntico (1f1ca7597eb7285e) a uno ya rechazado: no se ejecuta
- `#21` critico → **rechazado** [T2]: - script idéntico a uno ya rechazado: no se ejecutó
- `#22` orquestador → **tope_intentos** [T2]: 3 intentos: la subtarea queda fallida y la cola sigue
- `#23` redactor redacta (1000+200 tokens); recibió la orden de declarar: «- La subtarea T2 (Parte 2) no tiene resultado (fallida): no se reportan sus cifras.»

## timeout — disparado

`corridas/frenos/timeout/` · status **completado** · subtareas {'T1': 'aprobada (1 intentos)', 'T2': 'aprobada (2 intentos)'} · entregables ['reporte.md'] · tokens {'tokens_entrada': 7000, 'tokens_salida': 1400}

- `#14` ejecutor, intento 1 [T2]: returncode=-9, 3.01 s, error: timeout: superó 3 s y se mató el grupo de procesos
- `#15` critico → **rechazado** [T2]: - timeout: superó 3 s y se mató el grupo de procesos
- `#20` redactor redacta (1000+200 tokens)

## red — disparado

`corridas/frenos/red/` · status **parcial** · subtareas {'T1': 'aprobada (1 intentos)', 'T2': 'fallida (1 intentos)'} · entregables ['reporte.md'] · tokens {'tokens_entrada': 5000, 'tokens_salida': 1000}

- `#14` ejecutor → **confirmacion_red** (aprobado=False) [T2]: T2: el script quiere descargar (línea 2: import fetch_openml; línea 3: fetch_openml())
- `#15` ejecutor, intento 1 [T2]: returncode=None, 0.0 s, error: descarga no confirmada por una persona: línea 2: import fetch_openml; línea 3: fetch_openml()
- `#17` orquestador → **tope_intentos** [T2]: 1 intentos: la subtarea queda fallida y la cola sigue
- `#18` redactor redacta (1000+200 tokens); recibió la orden de declarar: «- La subtarea T2 (Parte 2) no tiene resultado (fallida): no se reportan sus cifras.»

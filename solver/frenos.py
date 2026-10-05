"""Parte 3 — dónde vive cada freno (este módulo es un mapa; los frenos están en su sitio).

Un freno tiene que estar en el punto exacto donde se decide, no en un módulo aparte: el
presupuesto se comprueba ANTES de cada llamada (en el cliente), la repetición antes de ejecutar
(en el orquestador), el tiempo máximo en el proceso (en el ejecutor) y la confirmación humana
antes de que exista el proceso (en el ejecutor, con lo que detecta la guarda).

| Freno                              | Código                                                        | Evento en la traza                 |
|------------------------------------|---------------------------------------------------------------|------------------------------------|
| 1. Presupuesto con reserva         | solver/cliente_llm.py · ClienteLLM.chat / disponible()        | decision: presupuesto_agotado      |
|    …y redactar con lo que hay      | solver/orquestador.py · _nodo, _tras_*, _notas                | (el reporte declara qué no se hizo)|
| 2. Tope de intentos                | solver/orquestador.py · _criticar (cfg.max_intentos)          | decision: tope_intentos            |
|    Detector de repetición          | solver/orquestador.py · _ejecutar (huella del script)         | decision: script_repetido          |
| 3. Tiempo máximo (killpg)          | solver/agentes/ejecutor.py · ejecutar (start_new_session)     | ejecucion: error «timeout: …»      |
| 4. Confirmación antes de la red    | solver/sandbox/guarda.py (descargas) + ejecutor.py (confirmar)| decision: confirmacion_red         |

Configuración: solver/config.py (presupuesto_tokens, reserva_redactor, max_intentos, timeout_s,
confirmar). Evidencia forzada con modelo de guion: scripts/forzar_frenos.py → corridas/frenos/.
Además: tope_de_razonamiento (cliente_llm.py), una llamada vacía por longitud en su tope no se
repite (no es un freno del enunciado, pero evita gastar dos veces lo mismo).
"""

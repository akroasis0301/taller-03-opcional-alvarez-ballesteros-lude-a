#!/usr/bin/env bash
# Las corridas finales del Taller 03 v2, en orden. Desde la RAÍZ del repo, con la VPN conectada.
#
#   bash scripts/corridas_2b.sh completo 1     # Solver completo, repetición 1 (≈1 h)
#   bash scripts/corridas_2b.sh completo 2
#   bash scripts/corridas_2b.sh sin_grafo 1    # ablación (≈45 min)
#   bash scripts/corridas_2b.sh sin_grafo 2
#   bash scripts/corridas_2b.sh tablas         # tablas de la 2.b con todas las repeticiones que existan
#   bash scripts/corridas_2b.sh frenos         # Parte 3 (sin H200, 1 min)
#   bash scripts/corridas_2b.sh juez           # Parte 4 (gemma3 en el Ollama de la H200)
#
# Cada corrida tiene su carpeta y su CSV propios (corridas/final/<variante>-r<n>/,
# resultados/final/resultados_<variante>_r<n>.csv): nada se mezcla ni se sobrescribe.
set -euo pipefail
cd "$(dirname "$0")/.."
GOLDEN=golden/golden_tareas.json
TAREAS="A B C S1 S2"
mkdir -p resultados/final corridas/final

corrida() {  # variante clase repeticion
  caffeinate -i uv run python solver-v2/evaluar_solver.py --golden "$GOLDEN" \
    --solver "solver.orquestador:$2" --ruta . --corridas "corridas/final/$1-r$3" \
    --solo $TAREAS --salida "resultados/final/resultados_$1_r$3.csv"
}

variantes() {  # arma los argumentos --variante/--corrida con lo que exista
  local bandera=$1 v r args=()
  for v in completo sin_grafo; do
    local lista=()
    for r in 1 2 3; do
      [[ -f "resultados/final/resultados_${v}_r$r.csv" ]] && lista+=("corridas/final/$v-r$r:resultados/final/resultados_${v}_r$r.csv")
    done
    ((${#lista[@]})) && args+=("$bandera" "$v" "${lista[@]}")
  done
  echo "${args[@]}"
}

case "${1:-}" in
  completo)  corrida completo Solver "${2:-1}" ;;
  sin_grafo) corrida sin_grafo SolverSinGrafo "${2:-1}" ;;
  tablas)    uv run python scripts/tablas_2b.py $(variantes --variante) --salida resultados/final/tablas_2b ;;
  frenos)    uv run python scripts/forzar_frenos.py ;;
  juez)      uv run python scripts/juez.py $(variantes --corrida) --salida resultados/final/juez ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac

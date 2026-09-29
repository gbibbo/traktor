---
name: auditar
description: Protocolo de investigación para auditar un problema, revisar un trabajo, diagnosticar un bug o comparar representaciones, parámetros u organizaciones. Cargar ANTES de investigar cuando el pedido sea "auditá", "revisá", "investigá", "por qué falla", "qué está pasando con", "diagnosticá", "cuál es mejor", o cuando una decisión (modelo, umbral, hiperparámetro, organización) dependa de una conclusión técnica. Gobierna cómo se llega a una conclusión.
---

# Auditar: protocolo de investigación

Adaptado del protocolo de Paradiddle (2026-09-29). Gobierna cómo se llega a una conclusión en
traktor. Otros agentes (Codex) lo leen desde esta ruta; AGENTS.md apunta acá.

## Regla cero

README, `docs/STATUS.md`, `docs/DECISIONS.md`, los informes de `docs/reports/`, los planes y los
comentarios del código son contexto: dicen dónde mirar. No son evidencia. La evidencia es lo que se
leyó o corrió en esta sesión. Una conclusión escrita se vuelve a verificar antes de apoyarse en ella.

## Protocolo

1. **El problema en una frase falsable**: qué se observa, qué se esperaba y qué distinguiría una
   explicación de otra. Si no sale la frase, falta el síntoma exacto: pedirlo.

2. **Reproducir u observar antes de opinar.** Correr el comando, leer el código, abrir el artefacto
   (`.parquet`, `.npy`, `config_<hash>.json`). "No pude reproducirlo con X, Y, Z" es un resultado.

3. **Hipótesis plausibles, cada una con su prueba discriminante**, sin cuota de relleno. Si queda
   una sola, escribir por qué se descartaron las otras.

4. **Evidencia dirigida.** Cada comando tiene que poder cambiar el veredicto. Los subagentes juntan
   evidencia y ahorran contexto, pero no son revisión independiente (mismo modelo, mismos sesgos).

5. **Las tres coordenadas de todo dato.** Sin las tres, el dato no entra a una conclusión:
   - **Qué lo produjo**: representación y capa (`clap_full`, MAEST-HF capa 7…), script y fase.
   - **Sobre qué datos**: dataset (`test_20` o `musica`), N real (`track_uids.json`, no el conteo
     del catálogo) y qué subconjunto (tripletas uniformes n = 57, carpetas propias, etc.).
   - **Con qué versión**: hash de config de Phase 2, commit, modelo y umbrales.

   Regenerar el catálogo o los embeddings invalida lo medido sobre los anteriores. La alineación de
   filas sale de `track_uids.json`, nunca del orden de archivos. Un dato copiado de un informe está
   transcripto, no verificado: verificar es ir al artefacto o al código.

6. **Reglas del dominio**:
   - **Un mejor puntaje no es una mejor organización.** Con n = 57 tripletas casi ninguna diferencia
     es concluyente; entre representaciones decide Gabriel escuchando playlists enteras (DECISIONS
     2026-09-29). Un puntaje sirve para descartar lo claramente peor, no para elegir.
   - **No ajustar contra un solo ejemplo.** Si un umbral o parámetro se mueve para que entre o salga
     un tema o una playlist, informar además cuántos otros temas cambian de etiqueta o de grupo.
   - **Una sola variable por comparación.** A/B con los mismos datos, la misma config y la misma
     semilla salvo lo que se compara. Una fila elegida mirando los mismos datos de evaluación es
     exploratoria, no una mejora validada.

7. **Revisión hostil antes de concluir.** Cambiar de rol y tratar de tumbar el propio informe. Para
   cada conclusión: ¿qué tendría que ser cierto para que sea falsa, y lo descarté con evidencia o solo
   no lo miré? Ataques mínimos:
   - ¿Razoné un efecto que se podía medir?
   - ¿El dato es correcto pero la frase habla de otra cosa?
   - ¿La mejora pudo comprarse con ceguera (menos temas evaluados, ruido reasignado, N distinto)?
   - ¿La evidencia sigue valiendo, o su fuente cambió después de medirla?
   - ¿Leí el comentario o el código?
   - ¿La afirmación es más ancha que lo verificado?
   - ¿La aritmética cierra (conteos, porcentajes, N)?
   - ¿Estoy cerrando un caso abierto porque cerrar se siente mejor que "no concluyente"?

   Lo que no sobrevive baja un nivel. Para decisiones grandes, un subagente con contexto fresco en
   rol de revisor hostil (recibe el informe y la evidencia, no el razonamiento) ataca sin inercia.

8. **Veredicto etiquetado.** Cada afirmación lleva una etiqueta:
   - **EVIDENCIA OBSERVADA**: lo que se vio, con `archivo:línea` o la salida de esta sesión.
   - **INFERENCIA**: lo que se deduce, con el paso explícito. No asciende a causa probada.
   - **HIPÓTESIS**: lo plausible no contrastado, con el experimento que la decidiría.
   - **NO CONCLUYENTE**: qué falta y cómo conseguirlo. Es una salida válida.

   Causalidad solo con un contraste que cambie una sola variable.

9. **Formato**: el veredicto primero, en prosa corta; después una tabla
   `afirmación | etiqueta | evidencia`, cada fila apuntando a algo de esta sesión. Si el resultado
   va a un informe de `docs/reports/`, el informe conserva las etiquetas.

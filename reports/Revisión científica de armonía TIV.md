# Revisión científica de armonía TIV

Revisión del 2026-11-06. Informe auditado: `harmonic-tiv`, commit `b228858722805cd2e66e81261d6006a0e5f68da5`. Recomendaciones, sin modificar el protocolo ni la implementación.

**Decisión (a). P4 modificada ahora → P5(b) después.** Conservaría FEM#10 y «sin blanqueo» para un piloto cuando puedas escuchar. No elegiría por S1 ni por teoría, pero tampoco descartaría variantes por correlacionar peor con quintas: eso convertiría indirectamente la tonalidad en el objetivo. P3 serviría como diagnóstico. Tras el piloto, congelaría una sola configuración para confirmarla con temas nuevos.

**Decisión (b). Compararía contra ambos controles.** Primero, la política real del recomendador: tonalidad de Traktor con tags como respaldo, filtro Camelot y orden CLAP. Después, CLAP+BPM sin armonía, retirando también el filtro tonal. No basta poner su peso en cero.

Mi criterio nuevo —una propuesta de producto, no un estándar científico— sería:

- 60 contextos nuevos por comparación, representativos del uso, separados del piloto por temas/artistas; no solamente desacuerdos entre sistemas.
- En ambos controles, exigir `U = (victorias + 0,5 × empates) / N ≥ 0,60` y prueba pareada unilateral `p < 0,05`, con orden confirmatorio fijo: el segundo contraste solo se habilita si pasa el primero.
- Conservar empates en U. Usar ítems independientes o análisis por bloques si comparten pistas. Congelar renderizado, BPM, ganancia y regiones; permitir ±2 a ambos métodos que optimizan transposición y fijar previamente el pitch del control sin armonía.
- Añadir 20 repeticiones separadas, excluidas de N; exigir 16 acuerdos como control operativo propio, sin presentarlo como garantía científica de fiabilidad.

Con 60 comparaciones independientes sin empates hacen falta 37 victorias. Si la preferencia verdadera fuera 60 %, la potencia sería apenas 45,1 %. No aprobar significaría evidencia insuficiente, no equivalencia. La conclusión sería personalizada para Gabriel.

**Justificación.**

| Afirmación | Etiqueta | Evidencia |
|---|---|---|
| Los proxies no seleccionan inequívocamente una variante | EVIDENCIA OBSERVADA | §3.3 y JSON congelado: MRR-hard favorece «sin blanqueo», pero AUC-hard baja de 0,9464 a 0,9411. |
| La evidencia favorable externa no prueba este uso | EVIDENCIA OBSERVADA | [Gebhardt 2015](https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15_submission_36.pdf): 8/10 frente a Traktor, con clips de ocho segundos y hasta ±6; no tus transiciones de dos minutos. |
| Conviene juzgar el solapamiento real | INFERENCIA | [Mixed In Key](https://mixedinkey.com/workflows/how-to-use-the-camelot-wheel/) recomienda escucharlo y reconoce que Camelot no garantiza una buena mezcla. Su práctica comercial no demuestra eficacia de TIV. |

**Qué me haría cambiar de decisión.** Integraría la configuración que cumpla los umbrales anteriores en datos nuevos. Volvería a elegir mediante un proxy solo si predice preferencias humanas fuera de los datos usados para ajustarlo. Hoy eso es NO CONCLUYENTE.

**Fallas del informe.** El bootstrap de teoría elimina multiplicidades con `dict.fromkeys`: sus intervalos deben recalcularse. El control de escucha maximiza Camelot sobre tags y no reproduce el producto. S1 usa 200 fragmentos, no necesariamente 200 temas; los proxies principales omiten transposición. Menor correlación con quintas no demuestra peores mezclas. Los 18 tests pasan, pero no resuelven estas limitaciones.

Evidencia y límites de verificación: [auditoría local](../research_notes/Revisión%20científica%20de%20armonía%20TIV/auditoria_local.md), [literatura](../research_notes/Revisión%20científica%20de%20armonía%20TIV/ciencia.md), [productos y evaluación](../research_notes/Revisión%20científica%20de%20armonía%20TIV/productos_evaluacion.md).

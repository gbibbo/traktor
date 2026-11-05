# Mapeos de Traktor de Gabriel

Mapeos para importar en otro ordenador, en Preferences > Controller Manager. Salen de la configuración
de Traktor 4.4.1 de Gabriel del 2026-11-05, y los exportó `tools/traktor_live/mapeo_tsi.py`.

| Archivo | Qué es |
| --- | --- |
| `Xone K2 - TRAKTOR ML (D0 libre).tsi` | Ver abajo. |
| `Teclado - TRAKTOR ML.tsi` | Su mapeo de teclado (Generic Keyboard, 56 asignaciones). |

**`Xone K2 - TRAKTOR ML (D0 libre).tsi`:**

- Es el mapeo propio de Gabriel para el Allen & Heath Xone K2: un Generic MIDI con todo en el canal 15,
  hecho en Traktor 3.2.0.
- Tiene 126 asignaciones. Se quitaron las dos de `Ch15.Note.D0`, que eran la preescucha: D0 es el botón
  de las sugerencias en vivo.
- `Ch15.Note.C#0` sigue desplegando el panel de temas.
- Puertos: `XONE:K2 MIDI`, que es como lo ve Windows MIDI Services.

El mapeo que vino antes es `XONE K2 + TECLADO (TEMPO).tsi`. Es una exportación completa de la
configuración de Traktor, con audio y ruteo, así que no se sube al repo.

## Importar

1. Abrir Preferences > Controller Manager.
2. Debajo de *Device*, elegir Add… > Import TSI > Import Other… y abrir el `.tsi`.
3. Elegir *In-Port* y *Out-Port*:
   - el K2 en Windows con Windows MIDI Services: `XONE:K2 MIDI`;
   - en otros equipos puede ser `XONE:K2`.
4. Para el teclado, importar `Teclado - TRAKTOR ML.tsi` de la misma forma.

Traktor agrega cada mapeo como un dispositivo nuevo. Si ya había uno del K2, borrar el viejo para que
no queden dos.

## Volver a exportar o liberar otro botón

Con la configuración de Traktor de este ordenador:

```
python tools/traktor_live/mapeo_tsi.py --listar --liberar "Ch15.Note.D0"
python tools/traktor_live/mapeo_tsi.py --dispositivo 0 --liberar "Ch15.Note.D0" --salida "tools/traktor_live/mapeos/Xone K2 - TRAKTOR ML (D0 libre).tsi"
python tools/traktor_live/mapeo_tsi.py --tipo teclado --dispositivo 0 --salida "tools/traktor_live/mapeos/Teclado - TRAKTOR ML.tsi"
```

- `--desde <archivo.tsi>` lee otro archivo en vez de la configuración de Traktor.
- `--puertos "Entrada|Salida"` cambia los puertos.
- La herramienta no toca la configuración de Traktor: solo escribe el archivo nuevo.

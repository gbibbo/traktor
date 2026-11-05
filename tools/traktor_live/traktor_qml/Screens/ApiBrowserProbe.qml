import QtQuick 2.0
import Traktor.Gui 1.0 as Traktor

// TRAKTOR ML - sonda del navegador de la pantalla del D2 (prueba de factibilidad, 2026-11-04).
// Se agrega al final de Screens/S8/Views/Screen.qml (instalar_api_client.py --sonda). Envia a
// http://localhost:8080/browserProbe/<lado> cuando se crea la pantalla, cada vez que cambia el nodo de su
// navegador (revisado cada 1 s) y un latido cada 10 s. Sirve para saber (1) si Traktor crea la pantalla
// de un D2 virtual sin el aparato conectado y (2) si su navegador sigue al arbol que se ve en Traktor.
// Solo lee: no navega ni carga nada.

Item {
  id: probe
  property string screenSide: ""
  property string lastPath: "-"
  property int beats: 0
  visible: false

  Traktor.Browser { id: probeBrowser; isActive: true }

  ListView {
    id: probeList
    model: probeBrowser.dataSet
    visible: false
    width: 1
    height: 1
    delegate: Item { width: 1; height: 1 }
  }

  function send(reason) {
    var request = new XMLHttpRequest()
    request.open("POST", "http://localhost:8080/browserProbe/" + screenSide, true)
    request.setRequestHeader("Content-Type", "application/json")
    request.send(JSON.stringify({
      reason: reason,
      path: String(probeBrowser.currentPath),
      isContentList: probeBrowser.isContentList,
      currentIndex: probeBrowser.currentIndex,
      count: probeList.count
    }))
  }

  Component.onCompleted: send("screenLoaded")

  Timer {
    interval: 1000
    repeat: true
    running: true
    onTriggered: {
      var p = String(probeBrowser.currentPath)
      probe.beats += 1
      if (p != probe.lastPath) {
        probe.lastPath = p
        probe.send("pathChanged")
      } else if (probe.beats % 10 == 0) {
        probe.send("heartbeat")
      }
    }
  }
}

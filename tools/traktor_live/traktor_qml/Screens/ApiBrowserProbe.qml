import CSI 1.0
import QtQuick 2.0
import Traktor.Gui 1.0 as Traktor

// TRAKTOR ML - sonda del navegador de la pantalla del D2 (prueba de factibilidad, 2026-11-04).
// Se agrega al final de Screens/S8/Views/Screen.qml (instalar_api_client.py --sonda).
// Envia a http://localhost:8080/browserProbe/<lado>: cuando se crea la pantalla (con la lista de
// propiedades y funciones de Traktor.Browser), cada vez que cambia el nodo de su navegador (revisado cada
// 250 ms, con los nombres de las carpetas del nivel actual) y un latido cada 10 s.
// Pide ordenes a GET http://localhost:8080/command/<lado> cada 500 ms (cada 5 s si el receptor no
// responde): "exit" (subir un nivel), "select" con un indice, "enter" (entrar a la carpeta elegida) y
// "report". Nunca entra a un tema: en la pantalla del D2, entrar a un tema lo carga en el deck.

Item {
  id: probe
  property string screenSide: ""
  property string lastPath: "-"
  property int beats: 0
  property int pollInterval: 500
  visible: false

  Traktor.Browser { id: probeBrowser; isActive: true }

  // Una instancia por elemento del nivel actual (solo listas de carpetas, para no crear miles con los temas)
  Repeater {
    id: rows
    model: probeBrowser.isContentList ? 0 : probeBrowser.dataSet
    delegate: Item {
      visible: false
      property string nm: model.nodeName
      property bool isFolder: model.dataType == BrowserDataType.Folder
    }
  }

  function names() {
    var out = []
    for (var i = 0; i < rows.count && i < 300; i++) {
      var it = rows.itemAt(i)
      out.push(it ? (it.isFolder ? "" : "~") + it.nm : "?")
    }
    return out
  }

  function apiKeys() {
    var keys = []
    for (var k in probeBrowser) keys.push(k + ":" + typeof probeBrowser[k])
    return keys
  }

  function post(reason, extra) {
    var body = {
      reason: reason,
      path: String(probeBrowser.currentPath),
      isContentList: probeBrowser.isContentList,
      currentIndex: probeBrowser.currentIndex,
      count: rows.count,
      names: probeBrowser.isContentList ? [] : names()
    }
    if (extra) for (var k in extra) body[k] = extra[k]
    var request = new XMLHttpRequest()
    request.open("POST", "http://localhost:8080/browserProbe/" + screenSide, true)
    request.setRequestHeader("Content-Type", "application/json")
    request.send(JSON.stringify(body))
  }

  function run(cmd) {
    if (cmd.cmd == "exit") {
      probeBrowser.exitNode()
    } else if (cmd.cmd == "select") {
      probeBrowser.changeCurrentIndex(cmd.arg)
    } else if (cmd.cmd == "enter") {
      var it = rows.itemAt(probeBrowser.currentIndex)
      if (probeBrowser.isContentList || !it || !it.isFolder) {
        post("refused", {cmd: cmd.cmd})
        return
      }
      if (probeBrowser.enterNode(0, probeBrowser.currentIndex)) probeBrowser.relocateCurrentIndex()
    }
    post("done", {cmd: cmd.cmd, id: cmd.id})
  }

  function poll() {
    var request = new XMLHttpRequest()
    request.onreadystatechange = function() {
      if (request.readyState != XMLHttpRequest.DONE) return
      probe.pollInterval = (request.status == 200) ? 500 : 5000
      if (request.status == 200 && request.responseText.length > 2) {
        try { run(JSON.parse(request.responseText)) } catch (e) { post("error", {error: String(e)}) }
      }
    }
    request.open("GET", "http://localhost:8080/command/" + screenSide, true)
    request.send()
  }

  Component.onCompleted: post("screenLoaded", {api: apiKeys()})

  Timer {
    interval: 250
    repeat: true
    running: true
    onTriggered: {
      var p = String(probeBrowser.currentPath)
      probe.beats += 1
      if (p != probe.lastPath) {
        probe.lastPath = p
        probe.post("pathChanged")
      } else if (probe.beats % 40 == 0) {
        probe.post("heartbeat")
      }
    }
  }

  Timer {
    interval: probe.pollInterval
    repeat: true
    running: true
    onTriggered: probe.poll()
  }
}

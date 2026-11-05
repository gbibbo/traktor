import CSI 1.0
import QtQuick 2.0
import Traktor.Gui 1.0 as Traktor

// TRAKTOR ML - navegador de la pantalla del D2 al servicio de las sugerencias en vivo (2026-11-04).
// Se agrega al final de Screens/S8/Views/Screen.qml (instalar_api_client.py --sonda). Este navegador es el
// mismo arbol que ve el DJ en Traktor: entrar y salir aqui lo mueve alli.
// Envia a http://localhost:8080/browserProbe/<lado>: cuando se crea la pantalla (con las propiedades y
// funciones de Traktor.Browser), cuando cambia el nodo (revisado cada 250 ms, con los nombres del nivel
// actual: carpetas tal cual, temas con "~") y un latido cada 10 s.
// Pide ordenes a GET http://localhost:8080/command/<lado> cada 150 ms (cada 5 s si el receptor no
// responde): "exit", "select" (indice), "enter" (la carpeta elegida), "report" y "goto" ({path: [...],
// index: n}): sale hasta la parte comun de la ruta y entra nivel por nivel buscando cada nombre; al
// final elige el indice. Nunca entra a un tema: en esta pantalla, entrar a un tema lo carga en el deck.
// 2026-11-05: el modelo de nombres se refresca a mano y el "goto" espera a que carguen las carpetas del
// Explorer (la primera prueba se rindio a los 2 s en C: > Users y vio vacio C: > Users > bibbo).

Item {
  id: probe
  property string screenSide: ""
  property string lastPath: "-"
  property int beats: 0
  property int pollInterval: 150
  property var job: null
  visible: false

  Traktor.Browser { id: probeBrowser; isActive: true }

  // Cuenta los elementos sin crear uno por elemento (ListView solo crea los visibles)
  ListView {
    id: counter
    model: probeBrowser.dataSet
    visible: false
    width: 1
    height: 1
    delegate: Item { width: 1; height: 1 }
  }

  // Un elemento por fila del nivel actual, para leer nombres y tipos (no en listas enormes). El modelo se
  // asigna a mano (refreshRows): con un binding, al entrar a C: > Users el ListView contaba 3 y el
  // Repeater seguia vacio mas de 2 s (2026-11-05).
  Repeater {
    id: rows
    model: 0
    delegate: Item {
      visible: false
      property bool isFolder: model.dataType == BrowserDataType.Folder
      property string nm: isFolder ? model.nodeName : model.trackName
    }
  }

  function wantedRows() { return (counter.count > 0 && counter.count <= 1500) ? counter.count : 0 }
  function rowsStale() { return rows.count != wantedRows() }
  function refreshRows() {
    rows.model = 0
    if (wantedRows() > 0) rows.model = probeBrowser.dataSet
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
    if (rowsStale()) refreshRows()
    var body = {
      reason: reason,
      path: String(probeBrowser.currentPath),
      isContentList: probeBrowser.isContentList,
      currentIndex: probeBrowser.currentIndex,
      count: counter.count,
      names: names()
    }
    if (extra) for (var k in extra) body[k] = extra[k]
    var request = new XMLHttpRequest()
    request.open("POST", "http://localhost:8080/browserProbe/" + screenSide, true)
    request.setRequestHeader("Content-Type", "application/json")
    request.send(JSON.stringify(body))
  }

  // Mismo nombre sin mayusculas, o con una "s" final de diferencia ("Playlist" en la ruta, "Playlists"
  // en el arbol). Nada de prefijos: "A1" no es "A10".
  function sameName(a, b) {
    var x = String(a).toLowerCase(), y = String(b).toLowerCase()
    return x == y || x + "s" == y || y + "s" == x
  }

  // Indice de la carpeta llamada name en el nivel actual: primero exacto, despues con sameName
  function findFolder(name) {
    var i, it
    for (i = 0; i < rows.count; i++) { it = rows.itemAt(i); if (it && it.isFolder && it.nm == name) return i }
    for (i = 0; i < rows.count; i++) { it = rows.itemAt(i); if (it && it.isFolder && sameName(it.nm, name)) return i }
    return -1
  }

  function finish(reason, message) {
    var j = probe.job
    probe.job = null
    post(reason, {id: j ? j.id : 0, error: message || ""})
  }

  // Un paso del "goto" cada 40 ms: salir mientras la ruta no coincida, entrar mientras falte bajar.
  // Un nivel vacio se espera (Traktor carga las carpetas del Explorer de a poco); si sigue vacio 1,6 s,
  // sale y vuelve a entrar una vez, y a los 4 s se rinde.
  function gotoStep() {
    var j = probe.job
    if (!j) return
    var now = String(probeBrowser.currentPath)
    if (Date.now() > j.deadline) { finish("gotoFailed", "tiempo agotado en " + now); return }
    if (j.waitPath !== "" && now == j.waitPath) return            // la ruta todavia no cambio
    if (j.waitPath !== "") { j.waitPath = ""; j.emptyTicks = 0; j.missTicks = 0; refreshRows() }
    var cur = now.split(" | ")
    var target = j.target
    var k = 0
    while (k < cur.length && k < target.length && sameName(cur[k], target[k])) k++
    if (k < cur.length) {
      j.waitPath = now
      probeBrowser.exitNode()
      return
    }
    if (cur.length == target.length) {
      if (j.index >= 0 && j.index < counter.count) probeBrowser.changeCurrentIndex(j.index)
      finish("gotoDone", "")
      return
    }
    if (rowsStale()) refreshRows()
    if (rows.count == 0) {                                          // el nivel todavia se esta cargando
      j.emptyTicks += 1
      if (j.emptyTicks == 40 && !j.reentered && cur.length > 2) {   // salir y volver a entrar, una vez
        j.reentered = true
        j.waitPath = now
        probeBrowser.exitNode()
        return
      }
      if (j.emptyTicks > 100) finish("gotoFailed", "lista vacia en " + now)
      return
    }
    var i = findFolder(target[k])
    if (i < 0) {                                                    // puede faltar cargar el resto
      j.missTicks += 1
      if (j.missTicks > 25) finish("gotoFailed", "no encontre la carpeta '" + target[k] + "' en " + now)
      return
    }
    j.emptyTicks = 0
    j.missTicks = 0
    probeBrowser.changeCurrentIndex(i)
    j.waitPath = now
    if (probeBrowser.enterNode(0, i)) probeBrowser.relocateCurrentIndex()
  }

  function run(cmd) {
    if (cmd.cmd == "goto") {
      var path = cmd.arg && cmd.arg.path ? cmd.arg.path : []
      if (path.length == 0 || path[0] != "Browser") path = ["Browser"].concat(path)
      probe.job = {id: cmd.id, target: path, index: (cmd.arg && cmd.arg.index !== undefined) ? cmd.arg.index : -1,
                   deadline: Date.now() + 9000, waitPath: "", emptyTicks: 0, missTicks: 0, reentered: false}
      return
    }
    if (cmd.cmd == "exit") {
      probeBrowser.exitNode()
    } else if (cmd.cmd == "select") {
      probeBrowser.changeCurrentIndex(cmd.arg)
    } else if (cmd.cmd == "enter") {
      var it = rows.itemAt(probeBrowser.currentIndex)
      if (!it || !it.isFolder) {
        post("refused", {cmd: cmd.cmd, id: cmd.id})
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
      probe.pollInterval = (request.status == 200) ? 150 : 5000
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
      if (p != probe.lastPath || probe.rowsStale()) probe.refreshRows()
      if (p != probe.lastPath) {
        probe.lastPath = p
        if (!probe.job) probe.post("pathChanged")
      } else if (probe.beats % 40 == 0) {
        probe.post("heartbeat")
      }
    }
  }

  Timer {
    interval: 40
    repeat: true
    running: probe.job !== null
    onTriggered: probe.gotoStep()
  }

  Timer {
    interval: probe.pollInterval
    repeat: true
    running: probe.job === null
    onTriggered: probe.poll()
  }
}

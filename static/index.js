var socket = io();
let isRunning = false, setupDone = false;
let layerShowing = 0;
let currentPlayID = undefined;
const fps = 15;

socket.on('connect', function() {
    socket.emit('update');
    socket.emit('step');
});

socket.on('update', function(data) {
    redraw(JSON.parse(data));
});

socket.on('layers_update', function(data) {
    if (setupDone || !data || typeof data !== 'object') return;
    setupDone = true;
    const container = document.getElementById('container');
    let frag = document.createDocumentFragment(),
    select = document.createElement('select');
    select.id = 'layerSelect';

    data.forEach((layer, index) => {
        select.options.add(new Option(layer, index))
    });

    frag.append(select);
    container.appendChild(frag);

    const changedText = document.getElementById('layerSelect');
    function layerList(){
        layerShowing = Number(this.value);
    }
    changedText.onchange = layerList;
});

function start() {
    if (isRunning) return;
    isRunning = true;
    currentPlayID = setInterval(function() {
        step();
    }, 1000 / fps);
}

function stop() {
    if (currentPlayID === undefined) return;
    clearInterval(currentPlayID);
    currentPlayID = undefined;
    isRunning = false;
}

function step() {
    socket.emit('step');
}

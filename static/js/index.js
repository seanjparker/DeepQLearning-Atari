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
    const data_as_json = JSON.parse(data);
    redraw(data_as_json);
    updateQValueGraph(data_as_json['layers'][3]);
});

socket.on('layers_update', function(data) {
    if (setupDone || !data || typeof data !== 'object') return;
    setupDone = true;
    const select = document.getElementById('layerSelect');
    data.forEach((layer, index) => {
        select.options.add(new Option(layer, index))
    });

    function layerList(){
        layerShowing = Number(this.value);
    }
    select.onchange = layerList;
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

var socket = io();
socket.on('connect', function() {
    console.log('connected');
    socket.emit('update', {data: 'I\'m connected!'});
});

socket.on('update', function(data) {
    console.log('update event recieved');
    redraw(JSON.parse(data));
});

function step() {
    console.log('stepping');
    socket.emit('step');
}

let canvas = document.getElementById("c");
let ctx = canvas.getContext("2d");
ctx.mozImageSmoothingEnabled = false;
ctx.webkitImageSmoothingEnabled = false;
ctx.msImageSmoothingEnabled = false;
ctx.imageSmoothingEnabled = false;

canvas.width = 1200;
canvas.height = 500;

let canvas_buffer = document.createElement("canvas");
let ctx_buffer = canvas_buffer.getContext("2d");

function redraw(data_as_json) {
    let data = data_as_json['src'].data;
    canvas_buffer.width = 84;
    canvas_buffer.height = 84;
    let imgData = ctx_buffer.createImageData(84, 84);
    // First, draw the processed observation
    for (let i = 0, j = 0; i < imgData.data.length; i+=4, j++) {
        // let x = (i / 4) % 84;
        // let y = Math.floor(i / (84 * 4));
        imgData.data[i] = data[j];
        imgData.data[i + 1] = data[j];
        imgData.data[i + 2] = data[j];
        imgData.data[i + 3] = 255;
    }
    ctx_buffer.putImageData(imgData, 0, 0);
    ctx.drawImage(canvas_buffer, 0, 0, 252, 252);

    // Third, draw the filter activations for a specific layer
    let num_filters = data_as_json['layers'].length;
    // num_filters = 5;
    canvas_buffer.width = 20;
    canvas_buffer.height = 20;
    for (let filter_i = 0; filter_i < num_filters; filter_i++) {
        let x = filter_i % 8;
        let y = Math.floor(filter_i / 8);
        data = data_as_json['layers'][filter_i].data;
        imgData = ctx_buffer.createImageData(20, 20); // one more to give some border between filters
        // First, draw the processed observation
        for (let i = 0, j = 0; i < imgData.data.length; i+=4, j++) {
            imgData.data[i] = data[j];
            imgData.data[i + 1] = data[j];
            imgData.data[i + 2] = data[j];
            imgData.data[i + 3] = 255;
        }
        ctx_buffer.putImageData(imgData, 0, 0);
        ctx.drawImage(canvas_buffer, 253 + (105 * x), 105 * y, 100, 100);
    }

    // let imgData = ctx.createImageData(84, 84); // width x height
    // let imgdata = imgData.data;
    //
    // for (let i = 0, j = 0; i < 84 * 84 * 4; i+=4, j++) {
    //     imgdata[i] = data[j];
    //     imgdata[i + 1] = data[j];
    //     imgdata[i + 2] = data[j];
    //     imgdata[i + 3] = 255;
    // }
    //
    // ctx.putImageData(imgData, 0, 0);
}
// a = {
//     src: {
//         width: 84,
//         height: 84,
//         data: []
//     },
//     layers: [
//         {
//             name: 'conv2d',
//             col: 8,
//             row: 4,
//             output_shape: 20,
//             data: []
//         }
//     ]
// };

//
// let data = imgData.data;
//
// for (let i = 0, j = 0; i < 20 * 20 * 4; i+=4, j++) {
//     data[i] = imgd[j];
//     data[i + 1] = imgd[j];
//     data[i + 2] = imgd[j];
//     data[i + 3] = 255;
// }
//
// ctx.putImageData(imgData, 0, 0);
// document.body.appendChild(canvas);

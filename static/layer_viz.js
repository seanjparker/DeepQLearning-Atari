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
    let top_offset = 30;

    let data = data_as_json['src'].data;
    canvas_buffer.width = 84;
    canvas_buffer.height = 84;
    let imgData = ctx_buffer.createImageData(84, 84);
    // First, draw the processed observation
    for (let i = 0, j = 0; i < imgData.data.length; i += 4, j++) {
        imgData.data[i] = data[j];
        imgData.data[i + 1] = data[j];
        imgData.data[i + 2] = data[j];
        imgData.data[i + 3] = 255;
    }
    ctx_buffer.putImageData(imgData, 0, 0);
    ctx.drawImage(canvas_buffer, 0, top_offset, 252, 252);

    // Second, draw the layer names
    ctx.fillStyle = 'white';
    ctx.fillRect(300, 0, 1500, 25); // Clear the old names

    ctx.fillStyle = 'black';
    let currentXOffset = 450;
    for (const [i, layer] of data_as_json['layers'].entries()) {
        if (i === layerShowing) ctx.font = 'bold 22px Arial';
        else ctx.font = '20px Arial';
        ctx.fillText(layer.name, currentXOffset, 20);
        currentXOffset += 15 * layer.name.length * 2;
    }

    // Third, draw the filter activations for a specific layer
    let num_filters = data_as_json['layers'][layerShowing]['layer_data'].length;
    canvas_buffer.width = 20;
    canvas_buffer.height = 20;
    for (let filter_i = 0; filter_i < num_filters; filter_i++) {
        let x = filter_i % 8;
        let y = Math.floor(filter_i / 8);
        data = data_as_json['layers'][layerShowing]['layer_data'][filter_i];
        imgData = ctx_buffer.createImageData(20, 20); // one more to give some border between filters
        // First, draw the processed observation
        for (let i = 0, j = 0; i < imgData.data.length; i += 4, j++) {
            imgData.data[i] = data[j];
            imgData.data[i + 1] = data[j];
            imgData.data[i + 2] = data[j];
            imgData.data[i + 3] = 255;
        }
        ctx_buffer.putImageData(imgData, 0, 0);
        ctx.drawImage(canvas_buffer, 253 + (105 * x), 105 * y + top_offset, 100, 100);
    }
}
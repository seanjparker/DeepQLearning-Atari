let canvas = document.getElementById("layer_viz");
let ctx = canvas.getContext("2d");
ctx.mozImageSmoothingEnabled = false;
ctx.webkitImageSmoothingEnabled = false;
ctx.msImageSmoothingEnabled = false;
ctx.imageSmoothingEnabled = false;

canvas.width = 1100;
canvas.height = 800;

let canvas_buffer = document.createElement("canvas");
let ctx_buffer = canvas_buffer.getContext("2d");

function redraw(data_as_json) {
    console.log(data_as_json);
    let top_offset = 30;


    let data = data_as_json['src'].data;
    const src_width = data_as_json['src']['width'];
    const src_height = data_as_json['src']['height'];
    canvas_buffer.width = src_width;
    canvas_buffer.height = src_height;
    let imgData = ctx_buffer.createImageData(src_width, src_height);
    // Draw the processed observation
    for (let i = 0, j = 0; i < imgData.data.length; i += 4, j++) {
        imgData.data[i] = data[j];
        imgData.data[i + 1] = data[j];
        imgData.data[i + 2] = data[j];
        imgData.data[i + 3] = 255;
    }
    ctx_buffer.putImageData(imgData, 0, 0);
    ctx.drawImage(canvas_buffer, 0, top_offset, 252, 252);

    // Draw the layer names
    // Clear the old names
    ctx.fillStyle = 'white';
    ctx.fillRect(300, 0, 1500, 25);

    ctx.fillStyle = 'black';
    let currentXOffset = 450;
    for (const [i, layer] of data_as_json['layers'].entries()) {
        if (layer.name == null) continue;
        if (i === layerShowing) ctx.font = 'bold 22px Arial';
        else ctx.font = '20px Arial';
        ctx.fillText(layer.name, currentXOffset, 20);
        currentXOffset += 15 * layer.name.length * 2;
    }

    // Draw the filter activations for a specific layer
    // Clear the old filters
    ctx.fillStyle = 'white';
    ctx.fillRect(250, 30, 1500, 1500);

    // Filter canvas width & height
    const filter_canvas = 800;
    const fc_padding = 5;

    let num_filters = data_as_json['layers'][layerShowing]['layer_data'].length;
    const filter_size = data_as_json['layers'][layerShowing]['output_shape'];

    // Calculate the width per filter
    const space_per_filter = Math.floor(filter_canvas / Math.sqrt(num_filters));

    // The number of cols and rows to place on the canvas
    const cols = Math.ceil(filter_canvas / space_per_filter) + 1;
    const rows = Math.ceil(num_filters / cols);

    // The resized filter size to make them more viewable
    const filter_resized = Math.floor(filter_canvas / Math.max(cols, rows));

    canvas_buffer.width = canvas_buffer.height = filter_size;
    for (let filter_i = 0; filter_i < num_filters; filter_i++) {
        let x = filter_i % cols;
        let y = Math.floor(filter_i / cols);
        data = data_as_json['layers'][layerShowing]['layer_data'][filter_i];
        imgData = ctx_buffer.createImageData(canvas_buffer.width, canvas_buffer.height); // one more to give some border between filters
        // First, draw the processed observation
        for (let i = 0, j = 0; i < imgData.data.length; i += 4, j++) {
            imgData.data[i] = data[j];
            imgData.data[i + 1] = data[j];
            imgData.data[i + 2] = data[j];
            imgData.data[i + 3] = 255;
        }
        ctx_buffer.putImageData(imgData, 0, 0);
        ctx.drawImage(canvas_buffer,
            253 + ((filter_resized + fc_padding) * x),
            (filter_resized + fc_padding) * y + top_offset,
            filter_resized,
            filter_resized);
    }
}
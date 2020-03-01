/* global Chart */

let q_val_canvas = $('#q_value_viz');
let q_val_line_canvas = $('#q_value_plot');
let chart_bar;
let chart_line;

let timeStep = 0;

function resetVizGraphs() {
    if (!chart_bar || !chart_line) return;

    chart_bar.data['datasets'][0].data = [];
    chart_bar.data.labels = [];

    chart_line.data['datasets'][0].data = [];
    chart_line.data.labels = [];
}

function findMaxIndex(array) {
    return array.reduce((i_max, x, i, arr) => x > arr[i_max] ? i : i_max, 0);
}

function updateGraphBarData(data) {
    const bestMoveIndex = findMaxIndex(data['layer_data']);
    chart_bar.data.labels = data['labels'];
    chart_bar.data['datasets'][0].data = data['layer_data'];
    chart_bar.data['datasets'][0].backgroundColor = function(context) {
        let index = context.dataIndex;
        return bestMoveIndex === index ? 'blue' : 'gray';
    };
}

function updateGraphLineData(data) {
    const best_q_val = Math.max(...data['layer_data']);
    const line_chart_data = chart_line.data['datasets'][0];

    if (line_chart_data.data.length > 50) {
        chart_line.data.labels.shift();
        line_chart_data.data.shift();
    }
    chart_line.data.labels.push(timeStep++);
    line_chart_data.data.push(best_q_val);
}

function createQValueLineGraph() {
    chart_line = new Chart(q_val_line_canvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            hover: {
                animationDuration: 10 // duration of animations when hovering an item
            },
            responsiveAnimationDuration: 0, // animation duration after a resize,
            legend: {
                display: false
            }
        }
    });
}

function createQValueBarGraph() {
    chart_bar = new Chart(q_val_canvas, {
        type: 'bar',
        data: {
            labels: ['1', '2', '3', '4'],
            datasets: [{
                data: []
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            hover: {
                animationDuration: 10 // duration of animations when hovering an item
            },
            responsiveAnimationDuration: 0, // animation duration after a resize,
            legend: {
                display: false
            }
        }
    });
}

function updateQValueGraph(eventData) {
    if (!chart_bar) createQValueBarGraph();
    if (!chart_line) createQValueLineGraph();

    updateGraphBarData(eventData);
    updateGraphLineData(eventData);
    chart_bar.update({
        preservation: false
    });
    chart_line.update();
}
/* global Chart */

let q_val_canvas = $('#q_value_viz');
let chart;

function findMaxIndex(array) {
    return array.reduce((i_max, x, i, arr) => x > arr[i_max] ? i : i_max, 0);
}

function updateGraphData(data) {
    const bestMoveIndex = findMaxIndex(data['layer_data']);
    chart.data.labels = data['labels'];
    chart.data['datasets'][0].data = data['layer_data'];
    chart.data['datasets'][0].backgroundColor = function(context) {
        let index = context.dataIndex;
        return bestMoveIndex === index ? 'blue' : 'gray';
    };
}

function createQValueGraph(data) {
    const bestMoveIndex = findMaxIndex(data['layer_data']);
    chart = new Chart(q_val_canvas, {
        type: 'bar',
        data: {
            labels: ['1', '2', '3', '4'],
            datasets: [{
                data: [0, 0, 0, 0]
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
    if (!chart) createQValueGraph(eventData);

    updateGraphData(eventData);
    chart.update({
        preservation: false
    });
}
function updatePrediction() {
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            document.getElementById("prediction").innerText = data.prediction;
        });
}

// Update prediction every second
setInterval(updatePrediction, 1000);

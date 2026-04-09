document.getElementById("analyzeBtn").addEventListener("click", function () {

    const fileInput = document.getElementById("audioFile");

    if (fileInput.files.length === 0) {
        alert("Please select an audio file");
        return;
    }

    const file = fileInput.files[0];

    // Show audio preview
    const audioPlayer = document.getElementById("audioPlayer");
    audioPlayer.src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        console.log("Server Response:", data);

        if (data.error) {
            document.getElementById("resultText").innerText = "Error";
            document.getElementById("confidenceText").innerText = data.error;
        } else {
            document.getElementById("resultText").innerText = data.result;
            document.getElementById("confidenceText").innerText =
                "Confidence: " + data.confidence + "%";
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
});
document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault(); // This prevents the default form submission behavior (page reload)

    const jobDescription = document.getElementById('job-description').value;
    const resultDiv = document.getElementById('result');

    // Show a loading message to the user
    resultDiv.textContent = 'Predicting...';
    resultDiv.className = ''; // Clear previous styling

    // Send the data to our Flask backend using the fetch API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `job_description=${encodeURIComponent(jobDescription)}` // Encode the text for URL transmission
    })
    .then(response => response.json()) // Parse the JSON response from the backend
    .then(data => {
        const prediction = data.prediction;
        resultDiv.textContent = `This job is likely: ${prediction}`;
        // Add a class for styling based on the prediction
        if (prediction === 'REAL') {
            resultDiv.className = 'real';
        } else {
            resultDiv.className = 'fake';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.textContent = 'An error occurred while getting the prediction.';
        resultDiv.className = 'fake';
    });
});
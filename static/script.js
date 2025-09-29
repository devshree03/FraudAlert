document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault(); // This prevents the default form submission behavior (page reload)
    
    const form = event.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById('result');
    
    // Show a loading message to the user
    resultDiv.textContent = 'Predicting...';
    resultDiv.className = ''; // Clear previous styling
    
    // Send the form data to the Flask backend using the fetch API
    fetch('/predict', {
        method: 'POST',
        body: formData // Send the entire form data object
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
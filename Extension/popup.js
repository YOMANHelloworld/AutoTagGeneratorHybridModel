document.getElementById('submitButton').addEventListener('click', function() {
  const title = document.getElementById('title').value.trim();
  const description = document.getElementById('description').value.trim();

  if (title || description) {
    const data = { titles: [title] };  

    fetch('http://localhost:5000/predict', {  
      method: 'POST',
      headers: { 'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      const predictedTags = data.predicted_tags;
      
      const outputContainer = document.getElementById('output-container');
      outputContainer.textContent = predictedTags.join(", ");  
    })
    .catch(error => {
      console.error('Error fetching predictions:', error);
        
      const outputContainer = document.getElementById('output-container');
      outputContainer.textContent = "Error: Unable to retrieve predictions.";
    });
  } else {
    const outputContainer = document.getElementById('output-container');

    outputContainer.textContent = "Please enter a title or description (optional)";
    console.error("Please enter a title or description.");
  }
});

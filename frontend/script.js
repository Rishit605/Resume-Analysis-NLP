document.addEventListener('DOMContentLoaded', function() {
    const resumeText = document.getElementById('resumeText');
    const resumeFile = document.getElementById('resumeFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const predictBtn = document.getElementById('predictBtn');
    const filePreview = document.getElementById('filePreview');
    const categoryResult = document.getElementById('categoryResult');
    const confidenceResult = document.getElementById('confidenceResult');

    let currentFile = null;

    // Handle file upload
    uploadBtn.addEventListener('click', () => {
        resumeFile.click();
    });

    resumeFile.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        currentFile = file;
        filePreview.innerHTML = `<p>Selected file: ${file.name}</p>`;
        
        // Clear text input when file is selected
        resumeText.value = '';
    });

    // Handle prediction
    predictBtn.addEventListener('click', async () => {
        try {
            let response;
            
            if (currentFile) {
                // If file is selected, use file endpoint
                const formData = new FormData();
                formData.append('file', currentFile);
                
                response = await fetch('/classify_resume/file/', {
                    method: 'POST',
                    body: formData
                });
            } else if (resumeText.value.trim()) {
                // If text is entered, use text endpoint
                response = await fetch('/classify_resume/text/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        resume_text: resumeText.value
                    })
                });
            } else {
                alert('Please either upload a file or enter resume text');
                return;
            }

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Update results
            categoryResult.textContent = result.category;
            confidenceResult.textContent = `${(result.confidence * 100).toFixed(2)}%`;

            // If file was used, show extracted text
            if (currentFile && result.extracted_text) {
                filePreview.innerHTML = `
                    <p>Selected file: ${currentFile.name}</p>
                    <div class="extracted-text">
                        <h3>Extracted Text:</h3>
                        <pre>${result.extracted_text}</pre>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request');
        }
    });

    // Clear file when text is entered
    resumeText.addEventListener('input', () => {
        if (resumeText.value.trim()) {
            currentFile = null;
            resumeFile.value = '';
            filePreview.innerHTML = '<p class="placeholder">No file selected</p>';
        }
    });
});

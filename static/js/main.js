document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const predictionResult = document.getElementById('prediction-result');

    // Initialize Bootstrap tabs
    const tabElements = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabElements.forEach(tab => {
        tab.addEventListener('click', function (event) {
            event.preventDefault();

            // Remove active class from all tabs and tab panes
            document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('show');
                pane.classList.remove('active');
            });

            // Add active class to the clicked tab
            this.classList.add('active');

            // Show the corresponding tab pane
            const target = document.querySelector(this.dataset.bsTarget);
            if (target) {
                target.classList.add('show');
                target.classList.add('active');
            }
        });
    });

    uploadForm.addEventListener('submit', function (event) {
        event.preventDefault();

        // Validate file input
        if (!fileInput.files.length) {
            alert('Please select a file to upload');
            return;
        }

        const file = fileInput.files[0];

        // Validate file type
        const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav'];
        if (!validTypes.includes(file.type) &&
            !file.name.endsWith('.mp3') &&
            !file.name.endsWith('.wav')) {
            alert('Please select a valid MP3 or WAV file');
            return;
        }

        // Show loading indicator and disable upload button
        loadingIndicator.classList.remove('d-none');
        uploadBtn.disabled = true;
        resultsContainer.classList.add('d-none');

        // Create FormData object and append file
        const formData = new FormData();
        formData.append('file', file);

        // Send request to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Update prediction results
                predictionResult.textContent = data.prediction;

                // Update visualization images
                document.getElementById('waveform-img').src = 'data:image/png;base64,' + data.waveform;
                document.getElementById('spectrogram-img').src = 'data:image/png;base64,' + data.spectrogram;
                document.getElementById('mel-spectrogram-img').src = 'data:image/png;base64,' + data.mel_spectrogram;
                document.getElementById('mfcc-img').src = 'data:image/png;base64,' + data.mfcc;

                // Show results container
                resultsContainer.classList.remove('d-none');

                // Scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while processing your file. Please try again.');
            })
            .finally(() => {
                // Hide loading indicator and enable upload button
                loadingIndicator.classList.add('d-none');
                uploadBtn.disabled = false;
            });
    });
});
// ============================================================================
// Process Images - Simple & Clean
// ============================================================================

async function processImages() {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const paginationControls = document.getElementById('paginationControls');

    // Reset state
    resultsGrid.innerHTML = '';
    paginationControls.style.display = 'none';
    downloadBtn.style.display = 'none';
    processingControls.style.display = 'flex';
    AppState.processedResults = [];
    AppState.allResults = [];
    AppState.currentPage = 1;
    isProcessing = true;

    const totalImages = AppState.uploadQueue.length;
    let processedCount = 0;

    // Get parameters once
    const parameters = getModelParameters();

    // Process each image one by one
    for (const queueItem of AppState.uploadQueue) {
        // Check if should stop
        if (shouldStop) {
            showToast('Processing stopped');
            break;
        }

        // Wait while paused
        while (isPaused && !shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Build request (NEW: using image_id)
        const formData = new FormData();

        // NEW: Just send image_id (backend will handle file access)
        formData.append('image_id', queueItem.image_id);
        formData.append('model', AppState.selectedModel);
        formData.append('parameters', JSON.stringify(parameters));
        if (AppState.customPrompt) {
            formData.append('prompt', AppState.customPrompt);
        }

        try {
            // Send to backend
            const response = await fetch(`${AppState.apiBaseUrl}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // Store result
            AppState.allResults.push({ queueItem, data });

            // Store for download
            AppState.processedResults.push({
                filename: queueItem.filename,
                caption: data.caption,
                path: queueItem.path || queueItem.filename
            });

            // Only add new item if it's on the current page (don't re-render everything)
            addResultItemToCurrentPage(queueItem, data);

        } catch (error) {
            console.error(`Error processing ${queueItem.filename}:`, error);
        }

        // Update progress
        processedCount++;
        const progress = processedCount / totalImages;
        showToast(`Processed ${processedCount}/${totalImages}`, true, progress);
    }

    // Reset stop/pause flags
    shouldStop = false;
    isPaused = false;

    // Finished
    isProcessing = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
        showToast('All images processed!');
    }
}

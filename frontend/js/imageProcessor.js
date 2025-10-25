async function processImages() {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const paginationControls = document.getElementById('paginationControls');

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

    const parameters = getModelParameters();

    for (const queueItem of AppState.uploadQueue) {
        if (shouldStop) {
            showToast('Processing stopped');
            break;
        }

        while (isPaused && !shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        const formData = new FormData();

        formData.append('image_id', queueItem.image_id);
        formData.append('model', AppState.selectedModel);
        formData.append('parameters', JSON.stringify(parameters));
        if (AppState.customPrompt) {
            formData.append('prompt', AppState.customPrompt);
        }

        try {
            const response = await fetch(`${AppState.apiBaseUrl}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            AppState.allResults.push({ queueItem, data });

            AppState.processedResults.push({
                filename: queueItem.filename,
                caption: data.caption,
                path: queueItem.path || queueItem.filename
            });

            addResultItemToCurrentPage(queueItem, data);

        } catch (error) {
            console.error(`Error processing ${queueItem.filename}:`, error);
        }

        processedCount++;
        const progress = processedCount / totalImages;
        showToast(`Processed ${processedCount}/${totalImages}`, true, progress);
    }

    shouldStop = false;
    isPaused = false;

    isProcessing = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
        showToast('All images processed!');
    }
}

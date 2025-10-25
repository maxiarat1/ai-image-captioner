function removeFromQueue(image_id) {
    AppState.uploadQueue = AppState.uploadQueue.filter(item => item.image_id !== image_id);
    updateUploadGrid();
    updateInputNodes();
    showToast('Image removed from queue');
}

async function clearQueue() {
    const gridContainer = document.getElementById('uploadGridContainer');
    const folderBrowser = document.getElementById('folderBrowser');

    gridContainer.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    gridContainer.style.opacity = '0';
    gridContainer.style.transform = 'scale(0.95)';

    setTimeout(async () => {
        try {
            await fetch(`${AppState.apiBaseUrl}/session/clear`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Error clearing session:', error);
        }

        AppState.uploadQueue = [];
        updateUploadGrid();
        updateInputNodes();
        showToast('Queue cleared');

        folderBrowser.value = '';

        gridContainer.style.opacity = '';
        gridContainer.style.transform = '';
    }, 300);
}

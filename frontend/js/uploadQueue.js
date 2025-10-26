async function removeFromQueue(image_id) {
    try {
        // Remove from backend database
        const response = await fetch(`${AppState.apiBaseUrl}/session/remove/${image_id}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            console.error('Error removing image:', error);
            showToast('Failed to remove image', 'error');
            return;
        }

        // Remove from frontend state
        AppState.uploadQueue = AppState.uploadQueue.filter(item => item.image_id !== image_id);
        updateUploadGrid();
        updateInputNodes();
        showToast('Image removed from queue');
    } catch (error) {
        console.error('Error removing image:', error);
        showToast('Failed to remove image', 'error');
    }
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

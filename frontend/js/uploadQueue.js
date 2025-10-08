// ============================================================================
// Upload Queue Management
// ============================================================================

function removeFromQueue(id) {
    AppState.uploadQueue = AppState.uploadQueue.filter(item => item.id !== id);
    updateUploadGrid();
    showToast('Image removed from queue');
}

function clearQueue() {
    const gridContainer = document.getElementById('uploadGridContainer');
    const folderBrowser = document.getElementById('folderBrowser');

    // Add fade-out animation
    gridContainer.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    gridContainer.style.opacity = '0';
    gridContainer.style.transform = 'scale(0.95)';

    // Wait for animation to complete before clearing
    setTimeout(() => {
        AppState.uploadQueue = [];
        updateUploadGrid();
        showToast('Queue cleared');

        // Reset file input so same folder can be selected again
        folderBrowser.value = '';

        // Reset styles for next time
        gridContainer.style.opacity = '';
        gridContainer.style.transform = '';
    }, 300);
}

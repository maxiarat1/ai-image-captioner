// ============================================================================
// Upload Queue Management
// ============================================================================

function removeFromQueue(image_id) {
    // NEW: Filter by image_id instead of id
    AppState.uploadQueue = AppState.uploadQueue.filter(item => item.image_id !== image_id);
    updateUploadGrid();
    updateInputNodes();
    showToast('Image removed from queue');
}

async function clearQueue() {
    const gridContainer = document.getElementById('uploadGridContainer');
    const folderBrowser = document.getElementById('folderBrowser');

    // Add fade-out animation
    gridContainer.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    gridContainer.style.opacity = '0';
    gridContainer.style.transform = 'scale(0.95)';

    // Wait for animation to complete before clearing
    setTimeout(async () => {
        // NEW: Clear session on backend
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

        // Reset file input so same folder can be selected again
        folderBrowser.value = '';

        // Reset styles for next time
        gridContainer.style.opacity = '';
        gridContainer.style.transform = '';
    }, 300);
}

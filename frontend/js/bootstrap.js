// ============================================================================
// Application Bootstrap
// ============================================================================

/**
 * Auto-resume session: Load images from previous session if available
 */
async function autoResumeSession() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/images?page=1&per_page=1000`);
        if (!response.ok) {
            console.warn('Failed to fetch session images');
            return;
        }

        const data = await response.json();

        if (data.total > 0) {
            // Load images into queue
            AppState.uploadQueue = data.images;

            // Calculate total size for display
            const totalSize = data.images.reduce((sum, img) => sum + (img.size || 0), 0);
            const sizeText = formatFileSize(totalSize);

            // Update UI
            if (typeof updateUploadGrid === 'function') {
                updateUploadGrid();
            }
            if (typeof updateInputNodes === 'function') {
                updateInputNodes();
            }

            // Show notification
            showToast(`Session resumed: ${data.total} images (${sizeText})`, false);

            console.log(`Auto-resumed session with ${data.total} images`);
        }
    } catch (error) {
        console.error('Error auto-resuming session:', error);
        // Silently fail - don't block app initialization
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    // Fetch available models from backend
    if (typeof fetchAvailableModels === 'function') {
        await fetchAvailableModels();
    }

    // Auto-resume session: Load any existing images from previous session
    await autoResumeSession();

    // UI
    initTabNavigation();
    initThemeToggle();

    // Upload
    initUploadHandlers();

    // Node Editor
    initNodeEditor();
    // Node Editor context menu
    if (typeof NEContextMenu !== 'undefined') NEContextMenu.init();

    // Results & Export
    initCopyFunctionality();
    initExportModal();

    // Controls & Pagination & Search
    initProcessingControls();
    initPaginationControls();
    initSearchHandlers();

    // Misc
    initModalHandlers();
    initDownloadButton();
    initToastHoverBehavior();

    if (window.Logger) {
        Logger.info('AI Image Tagger with Node Editor initialized');
    }
});

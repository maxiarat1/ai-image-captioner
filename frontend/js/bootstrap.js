async function autoResumeSession() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/images?page=1&per_page=1000`);
        if (!response.ok) {
            console.warn('Failed to fetch session images');
            return;
        }

        const data = await response.json();

        if (data.total > 0) {
            AppState.uploadQueue = data.images;

            const totalSize = data.images.reduce((sum, img) => sum + (img.size || 0), 0);
            const sizeText = formatFileSize(totalSize);

            if (typeof updateUploadGrid === 'function') {
                updateUploadGrid();
            }
            if (typeof updateInputNodes === 'function') {
                updateInputNodes();
            }

            showToast(`Session resumed: ${data.total} images (${sizeText})`, false);

            console.log(`Auto-resumed session with ${data.total} images`);
        }
    } catch (error) {
        console.error('Error auto-resuming session:', error);
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    if (typeof fetchAvailableModels === 'function') {
        await fetchAvailableModels();
    }

    await autoResumeSession();

    initTabNavigation();
    initThemeToggle();

    initUploadHandlers();

    initNodeEditor();
    if (typeof NEContextMenu !== 'undefined') NEContextMenu.init();

    initCopyFunctionality();
    initExportModal();

    initProcessingControls();
    initPaginationControls();
    initSearchHandlers();

    initModalHandlers();
    initDownloadButton();
    initToastHoverBehavior();

    if (window.Logger) {
        Logger.info('AI Image Tagger with Node Editor initialized');
    }
});

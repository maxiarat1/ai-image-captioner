// ============================================================================
// Application Bootstrap
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // Fetch available models from backend
    if (typeof fetchAvailableModels === 'function') {
        await fetchAvailableModels();
    }

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

// ============================================================================
// Application Bootstrap
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // UI
    initTabNavigation();
    initThemeToggle();

    // Upload
    initUploadHandlers();

    // Node Editor
    initNodeEditor();

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

    console.log('AI Image Tagger with Node Editor initialized');
});

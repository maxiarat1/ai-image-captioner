// ============================================================================
// Application Bootstrap
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // UI
    initTabNavigation();
    initThemeToggle();

    // Upload
    initUploadHandlers();

    // Options & Config
    initOptionsHandlers();
    initConfigModals();

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

    // Fetch available models from backend
    await fetchAvailableModels();

    // Load user configuration
    await loadUserConfig();

    console.log('AI Image Tagger initialized');
});

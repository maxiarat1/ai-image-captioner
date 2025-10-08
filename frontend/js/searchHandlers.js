// ============================================================================
// Search Handlers
// ============================================================================

function initSearchHandlers() {
    // Upload search
    const uploadSearchInput = document.getElementById('uploadSearchInput');
    if (uploadSearchInput) {
        uploadSearchInput.addEventListener('input', (e) => {
            AppState.uploadSearchQuery = e.target.value.trim();
            AppState.uploadCurrentPage = 1; // Reset to first page when searching
            updateUploadGrid();
        });
    }

    // Results search
    const resultsSearchInput = document.getElementById('resultsSearchInput');
    if (resultsSearchInput) {
        resultsSearchInput.addEventListener('input', (e) => {
            AppState.resultsSearchQuery = e.target.value.trim();
            AppState.currentPage = 1; // Reset to first page when searching
            renderCurrentPage();
        });
    }
}

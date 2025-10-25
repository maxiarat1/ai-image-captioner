// ============================================================================
// Search Handlers
// ============================================================================

function initSearchHandlers() {
    // Upload search
    const uploadSearchInput = document.getElementById('uploadSearchInput');
    const uploadSearchClear = document.getElementById('uploadSearchClear');

    if (uploadSearchInput) {
        uploadSearchInput.addEventListener('input', (e) => {
            AppState.uploadSearchQuery = e.target.value.trim();
            AppState.uploadCurrentPage = 1;
            updateUploadGrid();

            if (uploadSearchClear) {
                uploadSearchClear.style.display = e.target.value ? 'flex' : 'none';
            }
        });
    }

    if (uploadSearchClear) {
        uploadSearchClear.addEventListener('click', () => {
            uploadSearchInput.value = '';
            AppState.uploadSearchQuery = '';
            AppState.uploadCurrentPage = 1;
            updateUploadGrid();
            uploadSearchClear.style.display = 'none';
            uploadSearchInput.focus();
        });
    }

    // Results search
    const resultsSearchInput = document.getElementById('resultsSearchInput');
    const resultsSearchClear = document.getElementById('resultsSearchClear');

    if (resultsSearchInput) {
        resultsSearchInput.addEventListener('input', (e) => {
            AppState.resultsSearchQuery = e.target.value.trim();
            AppState.currentPage = 1;
            renderCurrentPage();

            if (resultsSearchClear) {
                resultsSearchClear.style.display = e.target.value ? 'flex' : 'none';
            }
        });
    }

    if (resultsSearchClear) {
        resultsSearchClear.addEventListener('click', () => {
            resultsSearchInput.value = '';
            AppState.resultsSearchQuery = '';
            AppState.currentPage = 1;
            renderCurrentPage();
            resultsSearchClear.style.display = 'none';
            resultsSearchInput.focus();
        });
    }
}

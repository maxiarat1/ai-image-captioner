// ============================================================================
// Results Grid Management
// ============================================================================

function addResultItemToCurrentPage(queueItem, data) {
    const resultsGrid = document.getElementById('resultsGrid');
    const paginationControls = document.getElementById('paginationControls');

    // Calculate which page this item belongs to
    const itemIndex = AppState.allResults.length - 1;
    const itemPage = Math.ceil((itemIndex + 1) / AppState.itemsPerPage);

    // Only add to DOM if it's on the current page
    if (itemPage === AppState.currentPage) {
        const currentPageItemCount = resultsGrid.children.length;

        // Only add if we haven't exceeded the page limit
        if (currentPageItemCount < AppState.itemsPerPage) {
            const resultDiv = createResultElement(queueItem, data);

            // Add staggered animation delay
            const delayMs = currentPageItemCount * 80;
            resultDiv.style.animationDelay = `${delayMs}ms`;

            resultsGrid.appendChild(resultDiv);

            // Setup lazy loading for the new item
            setupLazyLoadingForGrid('resultsGrid');
        }
    }

    // Update pagination controls
    updatePaginationControls();
}

function createResultElement(queueItem, data) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result-item';
    resultDiv.dataset.imageId = queueItem.image_id;

    // Create placeholder for lazy-loaded image (same approach as Upload tab)
    resultDiv.innerHTML = `
        <div class="result-image">
            <div class="result-thumbnail-container" data-image-id="${queueItem.image_id}">
                <div class="thumbnail-placeholder">📷</div>
            </div>
        </div>
        <div class="result-text">
            <p>${data.caption}</p>
        </div>
    `;

    // Add Ctrl-click handler on result text for caption copying
    const resultText = resultDiv.querySelector('.result-text');
    resultText.addEventListener('click', (e) => {
        if (e.ctrlKey || e.metaKey) {
            navigator.clipboard.writeText(data.caption)
                .then(() => showToast('Caption copied!'))
                .catch(() => showToast('Copy failed'));
        }
    });

    return resultDiv;
}

function getFilteredResults() {
    if (!AppState.resultsSearchQuery) {
        return AppState.allResults;
    }

    const query = AppState.resultsSearchQuery.toLowerCase();
    return AppState.allResults.filter(({ data }) =>
        data.caption.toLowerCase().includes(query)
    );
}

function renderCurrentPage() {
    const resultsGrid = document.getElementById('resultsGrid');

    // Get filtered results
    const filteredResults = getFilteredResults();

    // Calculate pagination
    const start = (AppState.currentPage - 1) * AppState.itemsPerPage;
    const end = start + AppState.itemsPerPage;
    const pageItems = filteredResults.slice(start, end);

    // Clear and render items for current page
    resultsGrid.innerHTML = '';
    for (const { queueItem, data } of pageItems) {
        const resultDiv = createResultElement(queueItem, data);
        resultsGrid.appendChild(resultDiv);
    }

    // Setup lazy loading for this page
    setupLazyLoadingForGrid('resultsGrid');

    // Update pagination controls
    updatePaginationControls();
}

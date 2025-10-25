function addResultItemToCurrentPage(queueItem, data) {
    const resultsGrid = document.getElementById('resultsGrid');
    const paginationControls = document.getElementById('paginationControls');

    const itemIndex = AppState.allResults.length - 1;
    const itemPage = Math.ceil((itemIndex + 1) / AppState.itemsPerPage);

    if (itemPage === AppState.currentPage) {
        const currentPageItemCount = resultsGrid.children.length;

        if (currentPageItemCount < AppState.itemsPerPage) {
            const resultDiv = createResultElement(queueItem, data);

            const delayMs = currentPageItemCount * 80;
            resultDiv.style.animationDelay = `${delayMs}ms`;

            resultsGrid.appendChild(resultDiv);

            setupLazyLoadingForGrid('resultsGrid');
        }
    }

    updatePaginationControls();
}

function createResultElement(queueItem, data) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result-item';
    resultDiv.dataset.imageId = queueItem.image_id;

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

    const filteredResults = getFilteredResults();

    const start = (AppState.currentPage - 1) * AppState.itemsPerPage;
    const end = start + AppState.itemsPerPage;
    const pageItems = filteredResults.slice(start, end);

    resultsGrid.innerHTML = '';
    for (const { queueItem, data } of pageItems) {
        const resultDiv = createResultElement(queueItem, data);
        resultsGrid.appendChild(resultDiv);
    }

    setupLazyLoadingForGrid('resultsGrid');

    updatePaginationControls();
}

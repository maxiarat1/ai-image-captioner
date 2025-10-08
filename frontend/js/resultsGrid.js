// ============================================================================
// Results Grid Management
// ============================================================================

async function addResultItemToCurrentPage(queueItem, data) {
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
            const resultDiv = await createResultElement(queueItem, data);

            // Add staggered animation delay
            const delayMs = currentPageItemCount * 80;
            resultDiv.style.animationDelay = `${delayMs}ms`;

            resultsGrid.appendChild(resultDiv);
        }
    }

    // Update pagination controls
    updatePaginationControls();
}

async function createResultElement(queueItem, data) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result-item';

    // Get thumbnail
    let thumbnail;
    if (queueItem.file) {
        // Load from File object
        thumbnail = await loadThumbnailFromFile(queueItem.file);
    } else if (queueItem.path) {
        // Load from filesystem path
        thumbnail = await loadThumbnail(queueItem.path);
    } else {
        // Fallback: use data.image_preview from backend response
        thumbnail = data.image_preview || '';
    }

    resultDiv.innerHTML = `
        <div class="result-image">
            <img src="${thumbnail}" alt="${queueItem.filename}">
        </div>
        <div class="result-text">
            <p>${data.caption}</p>
        </div>
    `;

    const img = resultDiv.querySelector('.result-image img');

    // Check image aspect ratio and add class for stretched images
    if (thumbnail) {
        img.addEventListener('load', () => {
            const aspectRatio = img.naturalWidth / img.naturalHeight;
            if (aspectRatio > 2.5) {
                resultDiv.classList.add('stretched-image');
            }
        });
    }

    // Add click handler with Ctrl modifier support
    const resultImage = resultDiv.querySelector('.result-image');
    resultImage.addEventListener('click', (e) => {
        if (e.ctrlKey || e.metaKey) {
            navigator.clipboard.writeText(data.caption)
                .then(() => showToast('Caption copied!'))
                .catch(() => showToast('Copy failed'));
        } else {
            // Use the current thumbnail for preview
            const currentThumbnail = thumbnail || img.src;
            openImagePreview(currentThumbnail, data.caption, queueItem.filename);
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

async function renderCurrentPage() {
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
        const resultDiv = await createResultElement(queueItem, data);
        resultsGrid.appendChild(resultDiv);
    }

    // Update pagination controls
    updatePaginationControls();
}

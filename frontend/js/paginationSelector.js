// ============================================================================
// Page Selector Dropdown
// ============================================================================

function togglePageSelector(gridType) {
    const dropdownId = gridType === 'upload' ? 'uploadPageSelector' : 'resultsPageSelector';
    const dropdown = document.getElementById(dropdownId);

    if (!dropdown) return;

    const isActive = dropdown.classList.contains('active');

    // Close all dropdowns first
    document.querySelectorAll('.page-selector-dropdown').forEach(d => {
        d.classList.remove('active');
    });

    // Toggle this dropdown
    if (!isActive) {
        dropdown.classList.add('active');
        renderPageSelector(gridType);
    }
}

function renderPageSelector(gridType) {
    const dropdownId = gridType === 'upload' ? 'uploadPageSelector' : 'resultsPageSelector';
    const dropdown = document.getElementById(dropdownId);

    if (!dropdown) return;

    // Get current page and total pages based on grid type
    let currentPage, totalPages;

    if (gridType === 'upload') {
        const filteredQueue = getFilteredUploadQueue();
        currentPage = AppState.uploadCurrentPage;
        totalPages = Math.ceil(filteredQueue.length / AppState.itemsPerPage);
    } else {
        const filteredResults = getFilteredResults();
        currentPage = AppState.currentPage;
        totalPages = Math.ceil(filteredResults.length / AppState.itemsPerPage);
    }

    if (totalPages <= 1) {
        dropdown.classList.remove('active');
        return;
    }

    // Show all pages (scrollable dropdown)
    const startPage = 1;
    const endPage = totalPages;

    // Build page items
    let html = '';
    for (let page = startPage; page <= endPage; page++) {
        const isCurrentPage = page === currentPage;
        const className = isCurrentPage ? 'page-number-item current' : 'page-number-item';
        html += `<div class="${className}" data-page="${page}" data-grid-type="${gridType}">Page ${page}</div>`;
    }

    dropdown.innerHTML = html;

    // Add click handlers
    dropdown.querySelectorAll('.page-number-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            const page = parseInt(item.dataset.page);
            const type = item.dataset.gridType;
            goToPageNumber(page, type);
            dropdown.classList.remove('active');
        });
    });

    // Scroll to current page item
    setTimeout(() => {
        const currentItem = dropdown.querySelector('.page-number-item.current');
        if (currentItem) {
            currentItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, 0);
}

function goToPageNumber(page, gridType) {
    if (gridType === 'upload') {
        const filteredQueue = getFilteredUploadQueue();
        const totalPages = Math.ceil(filteredQueue.length / AppState.itemsPerPage);
        if (page >= 1 && page <= totalPages) {
            AppState.uploadCurrentPage = page;
            renderUploadGridPage();
        }
    } else {
        const filteredResults = getFilteredResults();
        const totalPages = Math.ceil(filteredResults.length / AppState.itemsPerPage);
        if (page >= 1 && page <= totalPages) {
            AppState.currentPage = page;
            renderCurrentPage();
        }
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.pagination-info-wrapper')) {
        document.querySelectorAll('.page-selector-dropdown').forEach(dropdown => {
            dropdown.classList.remove('active');
        });
    }
});

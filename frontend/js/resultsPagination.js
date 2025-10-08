// ============================================================================
// Results Pagination Functions
// ============================================================================

function updatePaginationControls() {
    const paginationControls = document.getElementById('paginationControls');
    const paginationInfo = document.getElementById('paginationInfo');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');

    const filteredResults = getFilteredResults();
    const totalItems = filteredResults.length;
    const totalPages = Math.ceil(totalItems / AppState.itemsPerPage);

    if (totalPages > 1) {
        paginationControls.style.display = 'flex';
        paginationInfo.textContent = `Page ${AppState.currentPage} of ${totalPages}`;
        prevBtn.disabled = AppState.currentPage === 1;
        nextBtn.disabled = AppState.currentPage === totalPages;
    } else {
        paginationControls.style.display = 'none';
    }
}

function nextPage() {
    const filteredResults = getFilteredResults();
    const totalPages = Math.ceil(filteredResults.length / AppState.itemsPerPage);
    if (AppState.currentPage < totalPages) {
        AppState.currentPage++;
        renderCurrentPage();
    }
}

function prevPage() {
    if (AppState.currentPage > 1) {
        AppState.currentPage--;
        renderCurrentPage();
    }
}

function goToPage(pageNumber) {
    const filteredResults = getFilteredResults();
    const totalPages = Math.ceil(filteredResults.length / AppState.itemsPerPage);
    if (pageNumber >= 1 && pageNumber <= totalPages) {
        AppState.currentPage = pageNumber;
        renderCurrentPage();
    }
}

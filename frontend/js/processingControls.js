function initProcessingControls() {
    const pauseBtn = document.getElementById('pauseBtn');
    const stopBtn = document.getElementById('stopBtn');

    // Hide pause button (backend execution doesn't support pause yet)
    if (pauseBtn) {
        pauseBtn.style.display = 'none';
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            const jobId = sessionStorage.getItem('currentJobId');
            if (!jobId) return;

            try {
                showToast('Cancelling execution...', true);

                const response = await fetch(`${AppState.apiBaseUrl}/graph/cancel/${jobId}`, {
                    method: 'POST'
                });

                if (response.ok) {
                    showToast('Execution cancelled');
                } else {
                    showToast('Failed to cancel execution');
                }
            } catch (error) {
                console.error('Error cancelling execution:', error);
                showToast('Failed to cancel execution');
            }
        });
    }
}

function initPaginationControls() {
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const paginationInfo = document.getElementById('paginationInfo');

    if (prevBtn) {
        prevBtn.addEventListener('click', prevPage);
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', nextPage);
    }

    if (paginationInfo) {
        paginationInfo.addEventListener('click', (e) => {
            e.stopPropagation();
            togglePageSelector('results');
        });
    }

    const uploadPrevBtn = document.getElementById('uploadPrevPageBtn');
    const uploadNextBtn = document.getElementById('uploadNextPageBtn');
    const uploadPaginationInfo = document.getElementById('uploadPaginationInfo');

    if (uploadPrevBtn) {
        uploadPrevBtn.addEventListener('click', () => {
            if (AppState.uploadCurrentPage > 1) {
                AppState.uploadCurrentPage--;
                renderUploadGridPage();
            }
        });
    }

    if (uploadNextBtn) {
        uploadNextBtn.addEventListener('click', () => {
            const filteredQueue = getFilteredUploadQueue();
            const totalPages = Math.ceil(filteredQueue.length / AppState.itemsPerPage);
            if (AppState.uploadCurrentPage < totalPages) {
                AppState.uploadCurrentPage++;
                renderUploadGridPage();
            }
        });
    }

    if (uploadPaginationInfo) {
        uploadPaginationInfo.addEventListener('click', (e) => {
            e.stopPropagation();
            togglePageSelector('upload');
        });
    }
}

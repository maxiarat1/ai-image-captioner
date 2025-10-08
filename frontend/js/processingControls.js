// ============================================================================
// Processing Controls
// ============================================================================

function initProcessingControls() {
    const pauseBtn = document.getElementById('pauseBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            if (!isProcessing) return;

            isPaused = !isPaused;
            pauseBtn.classList.toggle('paused', isPaused);

            if (isPaused) {
                pauseBtn.title = 'Resume processing';
                showToast('Processing paused', true);
            } else {
                pauseBtn.title = 'Pause processing';
                showToast('Processing resumed', true);
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (!isProcessing) return;

            shouldStop = true;
            isPaused = false; // Unpause if paused so the stop can execute

            const pauseBtnElement = document.getElementById('pauseBtn');
            if (pauseBtnElement) {
                pauseBtnElement.classList.remove('paused');
            }

            showToast('Stopping processing...', true);
        });
    }
}

function initPaginationControls() {
    // Results pagination
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');

    if (prevBtn) {
        prevBtn.addEventListener('click', prevPage);
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', nextPage);
    }

    // Upload pagination
    const uploadPrevBtn = document.getElementById('uploadPrevPageBtn');
    const uploadNextBtn = document.getElementById('uploadNextPageBtn');

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
}

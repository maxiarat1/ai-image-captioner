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
            isPaused = false;

            const pauseBtnElement = document.getElementById('pauseBtn');
            if (pauseBtnElement) {
                pauseBtnElement.classList.remove('paused');
            }

            showToast('Stopping processing...', true);
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

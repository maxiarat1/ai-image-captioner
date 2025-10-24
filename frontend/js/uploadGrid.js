// ============================================================================
// Upload Grid Management (unified with Results grid)
// ============================================================================

function updateUploadGrid() {
    const uploadCard = document.getElementById('uploadCard');
    const gridContainer = document.getElementById('uploadGridContainer');
    const uploadGrid = document.getElementById('uploadGrid');
    const uploadStats = document.getElementById('uploadStats');
    const queueSize = AppState.uploadQueue.length;

    if (queueSize === 0) {
        // Show upload card, hide grid
        uploadCard.style.display = 'block';
        gridContainer.style.display = 'none';
        uploadGrid.innerHTML = '';
        return;
    }

    // Show grid, hide upload card
    uploadCard.style.display = 'none';
    gridContainer.style.display = 'block';

    // Update stats
    const filteredQueue = getFilteredUploadQueue();
    const totalSize = filteredQueue.reduce((sum, item) => sum + item.size, 0);
    uploadStats.textContent = `${filteredQueue.length} image${filteredQueue.length !== 1 ? 's' : ''} (${formatFileSize(totalSize)})`;

    // Render current page
    renderUploadGridPage();
}

function getFilteredUploadQueue() {
    if (!AppState.uploadSearchQuery) {
        return AppState.uploadQueue;
    }

    const query = AppState.uploadSearchQuery.toLowerCase();
    return AppState.uploadQueue.filter(item =>
        item.filename.toLowerCase().includes(query)
    );
}

function renderUploadGridPage() {
    const uploadGrid = document.getElementById('uploadGrid');

    // Get filtered queue
    const filteredQueue = getFilteredUploadQueue();

    // Calculate pagination
    const start = (AppState.uploadCurrentPage - 1) * AppState.itemsPerPage;
    const end = start + AppState.itemsPerPage;
    const pageItems = filteredQueue.slice(start, end);

    // Clear and render grid items
    uploadGrid.innerHTML = '';
    pageItems.forEach(item => {
        const gridItem = createUploadGridItem(item);
        uploadGrid.appendChild(gridItem);
    });

    // Update pagination controls
    updateUploadPaginationControls();

    // Setup lazy loading for this page
    setupLazyLoadingForGrid('uploadGrid');
}

function createUploadGridItem(item) {
    const gridItem = document.createElement('div');
    gridItem.className = 'result-item upload-grid-item';
    gridItem.dataset.imageId = item.image_id;  // NEW: use image_id

    gridItem.innerHTML = `
        <div class="result-image">
            <div class="upload-thumbnail-container" data-image-id="${item.image_id}">
                <div class="thumbnail-placeholder">📷</div>
            </div>
            <button class="upload-remove-btn" data-image-id="${item.image_id}" title="Remove">×</button>
        </div>
        <div class="result-text upload-item-info">
            <div class="upload-item-name">${item.filename}</div>
            <div class="upload-item-size">${formatFileSize(item.size)}</div>
        </div>
    `;

    // Add remove button handler
    const removeBtn = gridItem.querySelector('.upload-remove-btn');
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeFromQueue(item.image_id);  // NEW: use image_id
    });

    return gridItem;
}

function updateUploadPaginationControls() {
    const paginationControls = document.getElementById('uploadPaginationControls');
    const paginationInfo = document.getElementById('uploadPaginationInfo');
    const prevBtn = document.getElementById('uploadPrevPageBtn');
    const nextBtn = document.getElementById('uploadNextPageBtn');

    const filteredQueue = getFilteredUploadQueue();
    const totalItems = filteredQueue.length;
    const totalPages = Math.ceil(totalItems / AppState.itemsPerPage);

    if (totalPages > 1) {
        paginationControls.style.display = 'flex';
        paginationInfo.textContent = `Page ${AppState.uploadCurrentPage} of ${totalPages}`;
        prevBtn.disabled = AppState.uploadCurrentPage === 1;
        nextBtn.disabled = AppState.uploadCurrentPage === totalPages;
    } else {
        paginationControls.style.display = 'none';
    }
}

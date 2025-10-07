// ============================================================================
// Application State
// ============================================================================
const AppState = {
    uploadQueue: [], // Array of {id, filename, size, path, file, thumbnail: null}
    apiBaseUrl: 'http://localhost:5000',
    selectedModel: 'blip',
    customPrompt: '',
    availableModels: [],
    processedResults: [], // Store processed results: {filename, caption, path}
    userConfig: null,
    allResults: [], // All result items for pagination: {queueItem, data}
    currentPage: 1,
    itemsPerPage: 12,
    thumbnailCache: new Map(), // Cache loaded thumbnails with LRU
    thumbnailCacheMaxSize: 300, // Max 300 thumbnails total (~45MB) - shared between tabs
    uploadCurrentPage: 1, // Separate pagination for upload grid
    uploadSearchQuery: '', // Search query for upload tab
    resultsSearchQuery: '' // Search query for results tab
};

// ============================================================================
// Utility Functions
// ============================================================================

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// ============================================================================
// Folder Scanning
// ============================================================================

async function scanFolder() {
    const folderPath = document.getElementById('folderPathInput').value.trim();

    if (!folderPath) {
        showToast('Please enter a folder path');
        return;
    }

    showToast('Scanning folder...', true);

    try {
        const response = await fetch(`${AppState.apiBaseUrl}/scan-folder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to scan folder');
        }

        const data = await response.json();

        // Clear existing queue
        AppState.uploadQueue = [];
        AppState.thumbnailCache.clear();

        // Add images to queue
        data.images.forEach(img => {
            AppState.uploadQueue.push({
                id: generateId(),
                filename: img.filename,
                size: img.size,
                path: img.path,
                thumbnail: null
            });
        });

        updateUploadGrid();
        showToast(`Found ${data.count} images`);

    } catch (error) {
        console.error('Error scanning folder:', error);
        showToast(error.message || 'Failed to scan folder');
    }
}

async function loadThumbnail(path) {
    // Check cache first (Map maintains insertion order for LRU)
    if (AppState.thumbnailCache.has(path)) {
        // Move to end (most recently used)
        const thumbnail = AppState.thumbnailCache.get(path);
        AppState.thumbnailCache.delete(path);
        AppState.thumbnailCache.set(path, thumbnail);
        return thumbnail;
    }

    try {
        const response = await fetch(`${AppState.apiBaseUrl}/image/thumbnail?path=${encodeURIComponent(path)}`);
        if (!response.ok) throw new Error('Failed to load thumbnail');

        const data = await response.json();

        // Add to cache with LRU eviction
        if (AppState.thumbnailCache.size >= AppState.thumbnailCacheMaxSize) {
            // Remove oldest (first) entry
            const firstKey = AppState.thumbnailCache.keys().next().value;
            AppState.thumbnailCache.delete(firstKey);
        }

        AppState.thumbnailCache.set(path, data.thumbnail);
        return data.thumbnail;
    } catch (error) {
        console.error('Error loading thumbnail:', error);
        return null;
    }
}

function removeFromQueue(id) {
    AppState.uploadQueue = AppState.uploadQueue.filter(item => item.id !== id);
    updateUploadGrid();
    showToast('Image removed from queue');
}

function clearQueue() {
    const gridContainer = document.getElementById('uploadGridContainer');
    const folderBrowser = document.getElementById('folderBrowser');

    // Add fade-out animation
    gridContainer.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    gridContainer.style.opacity = '0';
    gridContainer.style.transform = 'scale(0.95)';

    // Wait for animation to complete before clearing
    setTimeout(() => {
        AppState.uploadQueue = [];
        updateUploadGrid();
        showToast('Queue cleared');

        // Reset file input so same folder can be selected again
        folderBrowser.value = '';

        // Reset styles for next time
        gridContainer.style.opacity = '';
        gridContainer.style.transform = '';
    }, 300);
}

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
    gridItem.dataset.itemId = item.id;

    gridItem.innerHTML = `
        <div class="result-image">
            <div class="upload-thumbnail-container" data-item-id="${item.id}">
                <div class="thumbnail-placeholder">📷</div>
            </div>
            <button class="upload-remove-btn" data-id="${item.id}" title="Remove">×</button>
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
        removeFromQueue(item.id);
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

async function loadThumbnailFromFile(file) {
    // Check cache first by file name
    const cacheKey = `file:${file.name}:${file.size}`;
    if (AppState.thumbnailCache.has(cacheKey)) {
        const thumbnail = AppState.thumbnailCache.get(cacheKey);
        AppState.thumbnailCache.delete(cacheKey);
        AppState.thumbnailCache.set(cacheKey, thumbnail);
        return thumbnail;
    }

    try {
        // Create thumbnail from File object
        const reader = new FileReader();
        const thumbnail = await new Promise((resolve, reject) => {
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });

        // Add to cache with LRU eviction
        if (AppState.thumbnailCache.size >= AppState.thumbnailCacheMaxSize) {
            const firstKey = AppState.thumbnailCache.keys().next().value;
            AppState.thumbnailCache.delete(firstKey);
        }

        AppState.thumbnailCache.set(cacheKey, thumbnail);
        return thumbnail;
    } catch (error) {
        console.error('Error loading thumbnail from file:', error);
        return null;
    }
}

// Unified lazy loading for both upload and results grids
function setupLazyLoadingForGrid(gridId) {
    const grid = document.getElementById(gridId);
    const containers = grid.querySelectorAll('.upload-thumbnail-container[data-item-id]');

    if (containers.length === 0) {
        return;
    }

    const observer = new IntersectionObserver(async (entries) => {
        for (const entry of entries) {
            const container = entry.target;
            const itemId = container.dataset.itemId;

            if (!itemId) continue;

            const item = AppState.uploadQueue.find(i => i.id === itemId);
            if (!item) continue;

            if (entry.isIntersecting) {
                // LOAD: Element is visible, load thumbnail
                let thumbnail;

                if (item.file) {
                    // Load from File object
                    thumbnail = await loadThumbnailFromFile(item.file);
                } else if (item.path) {
                    // Load from filesystem path
                    thumbnail = await loadThumbnail(item.path);
                }

                if (thumbnail) {
                    container.innerHTML = `<img src="${thumbnail}" alt="Thumbnail" style="width: 100%; height: 100%; object-fit: cover; border-radius: var(--radius-md);">`;

                    // Add click handler for preview
                    const img = container.querySelector('img');
                    if (img) {
                        img.addEventListener('click', () => {
                            openImagePreview(thumbnail, item.filename, formatFileSize(item.size));
                        });
                    }
                }
            }
            // Note: No auto-unload - LRU cache handles memory management
        }
    }, {
        rootMargin: '500px', // Load well before visible for smoother scrolling
        threshold: 0
    });

    containers.forEach(container => observer.observe(container));
}

// ============================================================================
// Image Preview Modal
// ============================================================================

function openImagePreview(imageSrc, captionText, fileName) {
    const modal = document.getElementById('imagePreviewModal');
    const modalImage = document.getElementById('modalImage');
    const modalInfo = document.getElementById('modalInfo');
    const backdrop = document.getElementById('modalBackdrop');
    const currentTab = document.querySelector('.tab-content.active').id;

    modalImage.src = imageSrc;
    modalInfo.textContent = captionText || fileName;
    backdrop.className = `modal-backdrop ${currentTab}`;
    modal.classList.add('active');

    // Prevent body scrolling when modal is open
    document.body.style.overflow = 'hidden';
}

function closeImagePreview() {
    const modal = document.getElementById('imagePreviewModal');
    modal.classList.remove('active');

    // Re-enable body scrolling
    document.body.style.overflow = '';
}

// Initialize modal close handlers
function initModalHandlers() {
    const modal = document.getElementById('imagePreviewModal');
    const modalBackdrop = document.getElementById('modalBackdrop');
    const modalContent = document.querySelector('.modal-content');

    // Close on backdrop click (clicking outside modal content)
    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            closeImagePreview();
        }
    });

    // Prevent closing when clicking inside modal content
    if (modalContent) {
        modalContent.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    // Close on Esc key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeImagePreview();
        }
    });
}

// ============================================================================
// Toast Notifications
// ============================================================================

let toastTimeout;
let isProcessing = false;
let isPaused = false;
let shouldStop = false;

function showToast(message, keepVisible = false, progress = null) {
    const statusToast = document.getElementById('statusToast');

    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }

    // If progress is provided, show progress bar
    if (progress !== null && progress !== undefined) {
        const percentage = Math.round(progress * 100);
        statusToast.innerHTML = `
            <div class="toast-message">${message}</div>
            <div class="toast-progress-bar">
                <div class="toast-progress-fill" style="width: ${percentage}%"></div>
            </div>
        `;
    } else {
        statusToast.innerHTML = `<div class="toast-message">${message}</div>`;
    }

    statusToast.classList.add('show');

    if (!keepVisible && !isProcessing) {
        toastTimeout = setTimeout(() => {
            statusToast.classList.remove('show');
        }, 2000);
    }
}

// Initialize toast hover behavior
function initToastHoverBehavior() {
    const statusToast = document.getElementById('statusToast');

    document.addEventListener('mousemove', (e) => {
        if (!statusToast.classList.contains('show')) return;

        // Get toast position and dimensions
        const rect = statusToast.getBoundingClientRect();

        // Calculate distance from cursor to toast (with 100px buffer zone)
        const bufferZone = 100;
        const distanceX = Math.max(0, Math.max(rect.left - e.clientX, e.clientX - rect.right));
        const distanceY = Math.max(0, Math.max(rect.top - e.clientY, e.clientY - rect.bottom));
        const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

        // Hide toast if cursor is within buffer zone
        if (distance < bufferZone) {
            statusToast.classList.add('toast-hidden');
        } else {
            statusToast.classList.remove('toast-hidden');
        }
    });
}

// ============================================================================
// Page Load Transition
// ============================================================================

window.addEventListener('load', () => {
    const pageTransition = document.getElementById('pageTransition');

    setTimeout(() => {
        pageTransition.classList.add('fade-out');
        setTimeout(() => {
            pageTransition.style.display = 'none';
        }, 400);
    }, 800);
});

// ============================================================================
// Options Tab Functionality
// ============================================================================

async function fetchAvailableModels() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        AppState.availableModels = data.models;
        console.log('Available models:', AppState.availableModels);
    } catch (error) {
        console.error('Error fetching models:', error);
        showToast('Failed to load models from server');
    }
}

async function initOptionsHandlers() {
    const modelSelect = document.getElementById('modelSelect');
    const promptInput = document.getElementById('promptInput');
    const processImagesBtn = document.getElementById('processImagesBtn');
    const modelDescription = document.getElementById('modelDescription');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedContent = document.getElementById('advancedContent');

    // Model descriptions
    const modelDescriptions = {
        'blip': 'Fast, basic image captioning',
        'r4b': 'Advanced reasoning model with configurable parameters'
    };

    // Handle advanced settings toggle
    advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('active');
        advancedContent.classList.toggle('open');
    });

    // Handle model selection change
    modelSelect.addEventListener('change', async (e) => {
        AppState.selectedModel = e.target.value;
        modelDescription.textContent = modelDescriptions[e.target.value] || '';

        // Load parameters for the selected model
        await loadModelParameters(e.target.value);

        console.log('Model selected:', AppState.selectedModel);
    });

    // Handle prompt input change
    promptInput.addEventListener('input', (e) => {
        AppState.customPrompt = e.target.value.trim();
    });

    // Handle process images button
    processImagesBtn.addEventListener('click', async () => {
        if (AppState.uploadQueue.length === 0) {
            showToast('Please upload images first');
            return;
        }

        // Switch to Results tab
        const resultsTab = document.querySelector('.tab-btn[data-tab="results"]');
        resultsTab.click();

        // Start processing images
        await processImages();
    });

    // Load initial parameters for default model (BLIP)
    await loadModelParameters(AppState.selectedModel);
}

async function loadModelParameters(modelName) {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/model/info?model=${modelName}`);
        if (!response.ok) {
            throw new Error('Failed to fetch model parameters');
        }
        const data = await response.json();

        // Render parameters dynamically
        renderParameters(data.parameters);
    } catch (error) {
        console.error('Error loading model parameters:', error);
        showToast('Failed to load model parameters');
    }
}

function renderParameters(parameters) {
    const container = document.getElementById('dynamicParams');
    container.innerHTML = '';

    if (!parameters || parameters.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No parameters available for this model</p>';
        return;
    }

    parameters.forEach(param => {
        const paramGroup = document.createElement('div');
        paramGroup.className = 'param-group';

        // Make select and checkbox inputs full-width for better readability
        if (param.type === 'select' || param.type === 'checkbox') {
            paramGroup.classList.add('full-width');
        }

        // Create label
        const label = document.createElement('label');
        label.setAttribute('for', param.param_key);
        label.textContent = param.name;
        paramGroup.appendChild(label);

        // Create input based on type
        if (param.type === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.id = param.param_key;
            input.className = 'param-input';
            input.setAttribute('min', param.min);
            input.setAttribute('max', param.max);
            input.setAttribute('step', param.step);
            input.placeholder = 'Default';
            paramGroup.appendChild(input);
        } else if (param.type === 'select') {
            const select = document.createElement('select');
            select.id = param.param_key;
            select.className = 'param-select';

            // Add default option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Use default';
            select.appendChild(defaultOption);

            // Add other options
            param.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                select.appendChild(option);
            });

            paramGroup.appendChild(select);
        } else if (param.type === 'checkbox') {
            const checkboxLabel = document.createElement('label');
            checkboxLabel.className = 'checkbox-label';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = param.param_key;

            const span = document.createElement('span');
            span.textContent = param.name;

            checkboxLabel.appendChild(checkbox);
            checkboxLabel.appendChild(span);
            paramGroup.appendChild(checkboxLabel);
        }

        // Create description
        const description = document.createElement('p');
        description.className = 'param-description';
        description.textContent = param.description;
        paramGroup.appendChild(description);

        container.appendChild(paramGroup);
    });
}

function getModelParameters() {
    const parameters = {};
    const container = document.getElementById('dynamicParams');

    // Get all parameter inputs
    const numberInputs = container.querySelectorAll('input[type="number"]');
    const selects = container.querySelectorAll('select');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');

    // Collect number inputs
    numberInputs.forEach(input => {
        if (input.value) {
            const value = parseFloat(input.value);
            if (!isNaN(value)) {
                parameters[input.id] = value;
            }
        }
    });

    // Collect select values
    selects.forEach(select => {
        if (select.value) {
            parameters[select.id] = select.value;
        }
    });

    // Collect checkboxes
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            parameters[checkbox.id] = true;
        }
    });

    return parameters;
}

// ============================================================================
// Process Images - Simple & Clean
// ============================================================================
async function processImages() {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const paginationControls = document.getElementById('paginationControls');

    // Reset state
    resultsGrid.innerHTML = '';
    paginationControls.style.display = 'none';
    downloadBtn.style.display = 'none';
    processingControls.style.display = 'flex';
    AppState.processedResults = [];
    AppState.allResults = [];
    AppState.currentPage = 1;
    isProcessing = true;

    const totalImages = AppState.uploadQueue.length;
    let processedCount = 0;

    // Get parameters once
    const parameters = getModelParameters();

    // Process each image one by one
    for (const queueItem of AppState.uploadQueue) {
        // Check if should stop
        if (shouldStop) {
            showToast('Processing stopped');
            break;
        }

        // Wait while paused
        while (isPaused && !shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Build request
        const formData = new FormData();

        // Check if we have a file object (from file browser) or path (from backend scan)
        if (queueItem.file) {
            // Use uploaded file
            formData.append('image', queueItem.file);
        } else if (queueItem.path) {
            // Use file path
            formData.append('image_path', queueItem.path);
        }

        formData.append('model', AppState.selectedModel);
        formData.append('parameters', JSON.stringify(parameters));
        if (AppState.customPrompt) {
            formData.append('prompt', AppState.customPrompt);
        }

        try {
            // Send to backend
            const response = await fetch(`${AppState.apiBaseUrl}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // Store result
            AppState.allResults.push({ queueItem, data });

            // Store for download
            AppState.processedResults.push({
                filename: queueItem.filename,
                caption: data.caption,
                path: queueItem.path || queueItem.filename
            });

            // Only add new item if it's on the current page (don't re-render everything)
            await addResultItemToCurrentPage(queueItem, data);

        } catch (error) {
            console.error(`Error processing ${queueItem.filename}:`, error);
        }

        // Update progress
        processedCount++;
        const progress = processedCount / totalImages;
        showToast(`Processed ${processedCount}/${totalImages}`, true, progress);
    }

    // Reset stop/pause flags
    shouldStop = false;
    isPaused = false;

    // Finished
    isProcessing = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
        showToast('All images processed!');
    }
}

// ============================================================================
// Pagination Functions
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


// ============================================================================
// Export Functionality
// ============================================================================

async function exportAsTextZip() {
    const zip = new JSZip();
    AppState.processedResults.forEach((result) => {
        const txtFilename = result.filename.replace(/\.[^/.]+$/, '.txt');
        zip.file(txtFilename, result.caption);
    });

    const blob = await zip.generateAsync({
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 9 }
    });

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.zip`);
    showToast(`Exported ${AppState.processedResults.length} text files`);
}

async function exportAsJson() {
    const data = {};
    AppState.processedResults.forEach((result) => {
        data[result.filename] = result.caption;
    });

    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.json`);
    showToast('Exported as JSON');
}

async function exportAsCsv() {
    let csv = 'filename,caption\n';
    AppState.processedResults.forEach((result) => {
        const escapedCaption = `"${result.caption.replace(/"/g, '""')}"`;
        csv += `"${result.filename}",${escapedCaption}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.csv`);
    showToast('Exported as CSV');
}

async function exportWithExif() {
    showToast('Preparing images for EXIF embedding...', true);

    try {
        // Check if we have file objects or paths
        const hasFiles = AppState.uploadQueue.some(item => item.file);

        if (hasFiles) {
            // Use old method with file uploads
            const formData = new FormData();

            for (const result of AppState.processedResults) {
                const queueItem = AppState.uploadQueue.find(item => item.filename === result.filename);
                if (queueItem && queueItem.file) {
                    formData.append('images', queueItem.file);
                    formData.append('captions', result.caption);
                }
            }

            const response = await fetch(`${AppState.apiBaseUrl}/export/metadata`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to embed metadata');
            }

            const blob = await response.blob();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            downloadBlob(blob, `images_with_metadata_${timestamp}.zip`);
            showToast('Exported images with EXIF metadata');
        } else {
            // Use new method with paths
            const image_paths = AppState.processedResults.map(r => r.path);
            const captions = AppState.processedResults.map(r => r.caption);

            const response = await fetch(`${AppState.apiBaseUrl}/export/metadata`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_paths, captions })
            });

            if (!response.ok) {
                throw new Error('Failed to embed metadata');
            }

            const blob = await response.blob();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            downloadBlob(blob, `images_with_metadata_${timestamp}.zip`);
            showToast('Exported images with EXIF metadata');
        }
    } catch (error) {
        console.error('Error exporting with EXIF:', error);
        showToast('Failed to embed EXIF metadata');
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function initDownloadButton() {
    const downloadBtn = document.getElementById('downloadAllBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', openExportModal);
    }
}

function openExportModal() {
    if (AppState.processedResults.length === 0) {
        showToast('No results to export');
        return;
    }
    const modal = document.getElementById('exportModal');
    openModal(modal);
}

function initExportModal() {
    const modal = document.getElementById('exportModal');
    const closeBtn = document.getElementById('closeExportModal');
    const cancelBtn = document.getElementById('cancelExport');
    const confirmBtn = document.getElementById('confirmExport');

    closeBtn?.addEventListener('click', () => closeModal(modal));
    cancelBtn?.addEventListener('click', () => closeModal(modal));

    confirmBtn?.addEventListener('click', async () => {
        const format = document.querySelector('input[name="exportFormat"]:checked')?.value;
        closeModal(modal);

        const downloadBtn = document.getElementById('downloadAllBtn');
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<span class="download-icon">⏳</span> Exporting...';
        downloadBtn.disabled = true;

        try {
            if (format === 'text') await exportAsTextZip();
            else if (format === 'json') await exportAsJson();
            else if (format === 'csv') await exportAsCsv();
            else if (format === 'exif') await exportWithExif();
        } catch (error) {
            console.error('Export error:', error);
            showToast('Export failed');
        } finally {
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
        }
    });

    // Close on backdrop click
    const backdrop = modal.querySelector('.config-modal-backdrop');
    backdrop?.addEventListener('click', (e) => {
        if (e.target === backdrop) closeModal(modal);
    });
}

// ============================================================================
// Configuration Management
// ============================================================================

async function loadUserConfig() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/config`);
        if (!response.ok) {
            throw new Error('Failed to load user configuration');
        }
        const data = await response.json();
        AppState.userConfig = data.config;
        console.log('User configuration loaded:', AppState.userConfig);
    } catch (error) {
        console.error('Error loading user config:', error);
        showToast('Failed to load user configuration');
    }
}

async function saveUserConfig() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(AppState.userConfig)
        });

        if (!response.ok) {
            throw new Error('Failed to save user configuration');
        }

        const data = await response.json();
        console.log('Configuration saved successfully:', data);
        return true;
    } catch (error) {
        console.error('Error saving user config:', error);
        showToast('Failed to save configuration');
        return false;
    }
}

function initConfigModals() {
    const loadConfigModal = document.getElementById('loadConfigModal');
    const saveConfigModal = document.getElementById('saveConfigModal');
    const loadPromptModal = document.getElementById('loadPromptModal');
    const savePromptModal = document.getElementById('savePromptModal');

    const closeLoadConfig = document.getElementById('closeLoadConfig');
    const closeSaveConfig = document.getElementById('closeSaveConfig');
    const closeLoadPrompt = document.getElementById('closeLoadPrompt');
    const closeSavePrompt = document.getElementById('closeSavePrompt');

    const loadConfigBtn = document.getElementById('loadConfigBtn');
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    const loadPromptBtn = document.getElementById('loadPromptBtn');
    const savePromptBtn = document.getElementById('savePromptBtn');

    const cancelSaveConfig = document.getElementById('cancelSaveConfig');
    const confirmSaveConfig = document.getElementById('confirmSaveConfig');
    const cancelSavePrompt = document.getElementById('cancelSavePrompt');
    const confirmSavePrompt = document.getElementById('confirmSavePrompt');

    // Open modals
    loadConfigBtn?.addEventListener('click', () => openLoadConfigModal());
    saveConfigBtn?.addEventListener('click', () => openSaveConfigModal());
    loadPromptBtn?.addEventListener('click', () => openLoadPromptModal());
    savePromptBtn?.addEventListener('click', () => openSavePromptModal());

    // Close modals
    closeLoadConfig?.addEventListener('click', () => closeModal(loadConfigModal));
    closeSaveConfig?.addEventListener('click', () => closeModal(saveConfigModal));
    closeLoadPrompt?.addEventListener('click', () => closeModal(loadPromptModal));
    closeSavePrompt?.addEventListener('click', () => closeModal(savePromptModal));
    cancelSaveConfig?.addEventListener('click', () => closeModal(saveConfigModal));
    cancelSavePrompt?.addEventListener('click', () => closeModal(savePromptModal));

    // Close on backdrop click
    [loadConfigModal, saveConfigModal, loadPromptModal, savePromptModal].forEach(modal => {
        if (modal) {
            const backdrop = modal.querySelector('.config-modal-backdrop');
            backdrop?.addEventListener('click', (e) => {
                if (e.target === backdrop) {
                    closeModal(modal);
                }
            });
        }
    });

    // Close on Esc key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (loadConfigModal.classList.contains('active')) closeModal(loadConfigModal);
            if (saveConfigModal.classList.contains('active')) closeModal(saveConfigModal);
            if (loadPromptModal.classList.contains('active')) closeModal(loadPromptModal);
            if (savePromptModal.classList.contains('active')) closeModal(savePromptModal);
        }
    });

    // Save configuration and prompt
    confirmSaveConfig?.addEventListener('click', () => saveConfiguration());
    confirmSavePrompt?.addEventListener('click', () => savePrompt());
}

function openModal(modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal(modal) {
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

function openLoadConfigModal() {
    const modal = document.getElementById('loadConfigModal');
    const configList = document.getElementById('configList');

    if (!AppState.userConfig) {
        configList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No configurations available</p>';
        openModal(modal);
        return;
    }

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel] || {};
    const configs = Object.values(modelConfigs);

    if (configs.length === 0) {
        configList.innerHTML = `<p style="text-align: center; color: var(--text-secondary);">No saved configurations for ${AppState.selectedModel.toUpperCase()}</p>`;
    } else {
        configList.innerHTML = configs.map(config => `
            <div class="config-item" data-config-id="${config.id}">
                <div class="config-item-info">
                    <div class="config-item-name">${config.name}</div>
                    <div class="config-item-date">${new Date(config.createdAt).toLocaleDateString()}</div>
                </div>
                <button class="config-item-delete" data-config-id="${config.id}" title="Delete configuration">×</button>
            </div>
        `).join('');

        // Add click handlers for loading
        configList.querySelectorAll('.config-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking delete button
                if (e.target.closest('.config-item-delete')) return;

                const configId = item.dataset.configId;
                loadConfiguration(configId);
                closeModal(modal);
            });
        });

        // Add click handlers for delete buttons
        configList.querySelectorAll('.config-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const configId = btn.dataset.configId;
                await deleteConfiguration(configId);
                openLoadConfigModal(); // Refresh the list
            });
        });
    }

    openModal(modal);
}

function openSaveConfigModal() {
    const modal = document.getElementById('saveConfigModal');
    const configName = document.getElementById('configName');
    configName.value = '';
    openModal(modal);
}

function openLoadPromptModal() {
    const modal = document.getElementById('loadPromptModal');
    const promptList = document.getElementById('promptList');

    if (!AppState.userConfig || !AppState.userConfig.customPrompts || AppState.userConfig.customPrompts.length === 0) {
        promptList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No saved prompts available</p>';
        openModal(modal);
        return;
    }

    const prompts = AppState.userConfig.customPrompts;

    promptList.innerHTML = prompts.map(prompt => `
        <div class="config-item" data-prompt-id="${prompt.id}">
            <div class="config-item-info">
                <div class="config-item-name">${prompt.name}</div>
                <div class="config-item-date">${new Date(prompt.createdAt).toLocaleDateString()}</div>
            </div>
            <button class="config-item-delete" data-prompt-id="${prompt.id}" title="Delete prompt">×</button>
        </div>
    `).join('');

    // Add click handlers for loading
    promptList.querySelectorAll('.config-item').forEach(item => {
        item.addEventListener('click', (e) => {
            // Don't trigger if clicking delete button
            if (e.target.closest('.config-item-delete')) return;

            const promptId = item.dataset.promptId;
            loadPrompt(promptId);
            closeModal(modal);
        });
    });

    // Add click handlers for delete buttons
    promptList.querySelectorAll('.config-item-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const promptId = btn.dataset.promptId;
            await deletePrompt(promptId);
            openLoadPromptModal(); // Refresh the list
        });
    });

    openModal(modal);
}

function loadConfiguration(configId) {
    if (!AppState.userConfig) return;

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel] || {};
    const config = Object.values(modelConfigs).find(c => c.id === configId);

    if (!config) {
        showToast('Configuration not found');
        return;
    }

    // Apply parameters to the UI
    applyParametersToUI(config.parameters);
    showToast(`Loaded: ${config.name}`);
}

function applyParametersToUI(parameters) {
    const container = document.getElementById('dynamicParams');

    // Get all inputs in the container
    const numberInputs = container.querySelectorAll('input[type="number"]');
    const selects = container.querySelectorAll('select');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');

    // First, clear all inputs to their default state
    numberInputs.forEach(input => {
        input.value = '';
    });

    selects.forEach(select => {
        select.value = '';
    });

    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });

    // Then apply the values from the configuration
    Object.keys(parameters).forEach(key => {
        const value = parameters[key];

        // Skip null or undefined values (they've already been cleared)
        if (value === null || value === undefined) return;

        const input = container.querySelector(`#${key}`);
        if (input) {
            if (input.type === 'number') {
                input.value = value;
            } else if (input.type === 'checkbox') {
                input.checked = value;
            } else if (input.tagName === 'SELECT') {
                input.value = value;
            }
        }
    });
}

async function saveConfiguration() {
    const configName = document.getElementById('configName').value.trim();

    if (!configName) {
        showToast('Please enter a configuration name');
        return;
    }

    const parameters = getModelParameters();
    const configId = configName.toLowerCase().replace(/\s+/g, '_');

    // Ensure the structure exists
    if (!AppState.userConfig.savedConfigurations[AppState.selectedModel]) {
        AppState.userConfig.savedConfigurations[AppState.selectedModel] = {};
    }

    // Create new configuration
    const newConfig = {
        id: configId,
        name: configName,
        model: AppState.selectedModel,
        parameters: parameters,
        createdAt: new Date().toISOString()
    };

    AppState.userConfig.savedConfigurations[AppState.selectedModel][configId] = newConfig;

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Configuration "${configName}" saved successfully`);
        closeModal(document.getElementById('saveConfigModal'));
    }
}

function loadPrompt(promptId) {
    if (!AppState.userConfig) return;

    const prompt = AppState.userConfig.customPrompts.find(p => p.id === promptId);

    if (!prompt) {
        showToast('Prompt not found');
        return;
    }

    const promptInput = document.getElementById('promptInput');
    promptInput.value = prompt.text;
    AppState.customPrompt = prompt.text;
    showToast(`Loaded prompt: ${prompt.name}`);
}

function openSavePromptModal() {
    const promptInput = document.getElementById('promptInput');
    const currentPrompt = promptInput.value.trim();

    if (!currentPrompt) {
        showToast('Please enter a prompt before saving');
        return;
    }

    const modal = document.getElementById('savePromptModal');
    const promptName = document.getElementById('promptName');
    promptName.value = '';
    openModal(modal);
}

async function savePrompt() {
    const promptName = document.getElementById('promptName').value.trim();
    const promptInput = document.getElementById('promptInput');
    const promptText = promptInput.value.trim();

    if (!promptName) {
        showToast('Please enter a prompt name');
        return;
    }

    if (!promptText) {
        showToast('Prompt is empty');
        return;
    }

    const promptId = promptName.toLowerCase().replace(/\s+/g, '-');

    // Ensure customPrompts array exists
    if (!AppState.userConfig.customPrompts) {
        AppState.userConfig.customPrompts = [];
    }

    // Check if prompt already exists
    const existingIndex = AppState.userConfig.customPrompts.findIndex(p => p.id === promptId);

    const newPrompt = {
        id: promptId,
        name: promptName,
        text: promptText,
        createdAt: new Date().toISOString()
    };

    if (existingIndex >= 0) {
        // Update existing prompt
        AppState.userConfig.customPrompts[existingIndex] = newPrompt;
    } else {
        // Add new prompt
        AppState.userConfig.customPrompts.push(newPrompt);
    }

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Prompt "${promptName}" saved successfully`);
        closeModal(document.getElementById('savePromptModal'));
    }
}

async function deleteConfiguration(configId) {
    if (!AppState.userConfig) return;

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel];
    if (!modelConfigs || !modelConfigs[configId]) {
        showToast('Configuration not found');
        return;
    }

    const configName = modelConfigs[configId].name;

    // Delete the configuration
    delete AppState.userConfig.savedConfigurations[AppState.selectedModel][configId];

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Deleted: ${configName}`);
    }
}

async function deletePrompt(promptId) {
    if (!AppState.userConfig || !AppState.userConfig.customPrompts) return;

    const promptIndex = AppState.userConfig.customPrompts.findIndex(p => p.id === promptId);

    if (promptIndex === -1) {
        showToast('Prompt not found');
        return;
    }

    const promptName = AppState.userConfig.customPrompts[promptIndex].name;

    // Delete the prompt
    AppState.userConfig.customPrompts.splice(promptIndex, 1);

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Deleted: ${promptName}`);
    }
}

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

function initSearchHandlers() {
    // Upload search
    const uploadSearchInput = document.getElementById('uploadSearchInput');
    if (uploadSearchInput) {
        uploadSearchInput.addEventListener('input', (e) => {
            AppState.uploadSearchQuery = e.target.value.trim();
            AppState.uploadCurrentPage = 1; // Reset to first page when searching
            updateUploadGrid();
        });
    }

    // Results search
    const resultsSearchInput = document.getElementById('resultsSearchInput');
    if (resultsSearchInput) {
        resultsSearchInput.addEventListener('input', (e) => {
            AppState.resultsSearchQuery = e.target.value.trim();
            AppState.currentPage = 1; // Reset to first page when searching
            renderCurrentPage();
        });
    }
}

// ============================================================================
// Main Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    initTabNavigation();
    initThemeToggle();
    initUploadHandlers();
    initOptionsHandlers();
    initCopyFunctionality();
    initModalHandlers();
    initDownloadButton();
    initProcessingControls();
    initPaginationControls();
    initSearchHandlers();
    initConfigModals();
    initExportModal();
    initToastHoverBehavior();

    // Fetch available models from backend
    await fetchAvailableModels();

    // Load user configuration
    await loadUserConfig();

    console.log('AI Image Tagger initialized');
});

// ============================================================================
// Tab Navigation
// ============================================================================

function initTabNavigation() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabNavigation = document.querySelector('.tab-navigation');

    tabNavigation.setAttribute('data-active', '0');

    tabBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            tabNavigation.setAttribute('data-active', index.toString());

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${targetTab}-content`).classList.add('active');
        });
    });
}

// ============================================================================
// Theme Toggle
// ============================================================================

function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const htmlElement = document.documentElement;
    const curtain = document.getElementById('curtainOverlay');

    const currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') {
        htmlElement.setAttribute('data-theme', 'light');
    }

    themeToggle.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        // Theatrical curtain animation
        curtain.classList.add('closing');
        
        setTimeout(() => {
            // Change theme while curtains are fully closed
            if (newTheme === 'light') {
                htmlElement.setAttribute('data-theme', 'light');
            } else {
                htmlElement.removeAttribute('data-theme');
            }
            localStorage.setItem('theme', newTheme);
            
            // Wait 0.5s with curtains closed, then open
            setTimeout(() => {
                curtain.classList.remove('closing');
                curtain.classList.add('opening');
                
                setTimeout(() => curtain.classList.remove('opening'), 800);
            }, 300);
        }, 800);
    });
}

// ============================================================================
// Upload Handlers
// ============================================================================

function initUploadHandlers() {
    const folderBrowser = document.getElementById('folderBrowser');
    const folderPathInput = document.getElementById('folderPathInput');
    const browseFolderBtn = document.getElementById('browseFolderBtn');
    const scanFolderBtn = document.getElementById('scanFolderBtn');
    const clearQueueBtn = document.getElementById('clearQueueBtn');

    // Browse folder button - opens file picker
    browseFolderBtn.addEventListener('click', () => {
        folderBrowser.click();
    });

    // Handle folder selection
    folderBrowser.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            // Get the folder path from the first file
            const firstFile = e.target.files[0];
            // Extract folder path by removing the filename
            const fullPath = firstFile.webkitRelativePath || firstFile.name;
            const folderPath = fullPath.substring(0, fullPath.lastIndexOf('/'));

            // Since webkitRelativePath gives relative path, we need to get the actual path
            // Unfortunately, browsers don't expose the full filesystem path for security
            // So we'll scan using the FileList directly
            scanFolderFromFileList(e.target.files);
        }
    });

    // Scan folder button (if user manually enters path)
    if (scanFolderBtn) {
        scanFolderBtn.addEventListener('click', scanFolder);
    }

    // Clear queue button
    clearQueueBtn.addEventListener('click', clearQueue);
}

async function scanFolderFromFileList(files) {
    showToast('Scanning folder...', true, 0);

    try {
        // Clear existing queue
        AppState.uploadQueue = [];
        AppState.thumbnailCache.clear();

        let imageCount = 0;
        const fileArray = Array.from(files);
        const totalFiles = fileArray.length;

        // Filter and add images WITHOUT loading previews (lazy loading)
        for (let i = 0; i < fileArray.length; i++) {
            const file = fileArray[i];

            // Update progress (faster without loading images)
            if (i % 10 === 0 || i === totalFiles - 1) {
                const progress = (i + 1) / totalFiles;
                showToast(`Scanning files... ${i + 1}/${totalFiles}`, true, progress);
            }

            // Check if it's a supported image
            if (file.type && file.type.startsWith('image/')) {
                const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
                if (['.jpg', '.jpeg', '.png', '.webp', '.bmp'].includes(ext)) {
                    // DON'T load preview yet - just store file reference
                    AppState.uploadQueue.push({
                        id: generateId(),
                        filename: file.name,
                        size: file.size,
                        file: file, // Store the file object for lazy loading later
                        thumbnail: null // Will be loaded on-demand
                    });
                    imageCount++;
                }
            }
        }

        // Get folder name from first file
        if (files.length > 0 && files[0].webkitRelativePath) {
            const folderName = files[0].webkitRelativePath.split('/')[0];
            document.getElementById('folderPathInput').value = folderName;
        }

        updateUploadGrid();
        showToast(`Found ${imageCount} images`);

    } catch (error) {
        console.error('Error loading images:', error);
        showToast('Failed to load images');
    }
}

// ============================================================================
// Copy Functionality
// ============================================================================

function initCopyFunctionality() {
    // Copy functionality will be implemented when results are generated
    // Event delegation will be used to handle dynamically created result items
}

// ============================================================================
// Application State
// ============================================================================
const AppState = {
    uploadQueue: [], // Array of {file: File, preview: string, id: string}
    apiBaseUrl: 'http://localhost:5000',
    maxFileSize: 16 * 1024 * 1024, // 16MB
    supportedFormats: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
    selectedModel: 'blip',
    customPrompt: '',
    availableModels: [],
    resultsAutoFollow: true, // Track if we should auto-scroll to newest results
    processedResults: [], // Store processed results for downloading: {filename: string, caption: string}
    userConfig: null, // User configuration loaded from backend
    allResults: [], // All result items for virtual scrolling
    virtualScroll: {
        itemHeight: 200,
        buffer: 5,
        scrollTop: 0
    }
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

function isValidImageFile(file) {
    // Check file type
    if (!AppState.supportedFormats.includes(file.type)) {
        return { valid: false, error: 'Unsupported file format' };
    }

    // Check file size
    if (file.size > AppState.maxFileSize) {
        return { valid: false, error: 'File too large (max 16MB)' };
    }

    return { valid: true };
}

function createImagePreview(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ============================================================================
// Upload Queue Management
// ============================================================================

async function addFilesToQueue(files) {
    const fileArray = Array.from(files);
    let addedCount = 0;
    let errorCount = 0;
    const total = fileArray.length;

    // Show progress for large uploads
    if (total > 20) {
        showToast(`Loading ${total} images...`, true);
    }

    for (let i = 0; i < fileArray.length; i++) {
        const file = fileArray[i];
        const validation = isValidImageFile(file);

        if (!validation.valid) {
            console.warn(`Skipping ${file.name}: ${validation.error}`);
            errorCount++;
            continue;
        }

        // Check for duplicates
        const isDuplicate = AppState.uploadQueue.some(item =>
            item.file.name === file.name && item.file.size === file.size
        );

        if (isDuplicate) {
            console.warn(`Skipping duplicate: ${file.name}`);
            continue;
        }

        try {
            const preview = await createImagePreview(file);
            AppState.uploadQueue.push({
                id: generateId(),
                file: file,
                preview: preview
            });
            addedCount++;

            // Update progress for large uploads
            if (total > 20 && (i + 1) % 10 === 0) {
                showToast(`Loading ${i + 1}/${total} images...`, true);
            }
        } catch (error) {
            console.error(`Failed to create preview for ${file.name}:`, error);
            errorCount++;
        }
    }

    updateQueueUI();

    if (errorCount > 0) {
        showToast(`Added ${addedCount} images, ${errorCount} skipped`);
    } else if (addedCount > 0) {
        showToast(`Added ${addedCount} image${addedCount > 1 ? 's' : ''} to queue`);
    }
}

function removeFromQueue(id) {
    AppState.uploadQueue = AppState.uploadQueue.filter(item => item.id !== id);
    updateQueueUI();
    showToast('Image removed from queue');
}

function clearQueue() {
    AppState.uploadQueue = [];
    updateQueueUI();
    showToast('Queue cleared');
}

function updateQueueUI() {
    const uploadCard = document.getElementById('uploadCard');
    const queueContainer = document.getElementById('uploadQueueContainer');
    const queueList = document.getElementById('queueList');
    const queueStats = document.getElementById('queueStats');

    const queueSize = AppState.uploadQueue.length;

    if (queueSize === 0) {
        uploadCard.style.display = 'block';
        queueContainer.style.display = 'none';
        queueList.innerHTML = '';
        return;
    }

    // Show queue, hide upload card
    uploadCard.style.display = 'none';
    queueContainer.style.display = 'block';

    // Update stats
    const totalSize = AppState.uploadQueue.reduce((sum, item) => sum + item.file.size, 0);
    queueStats.textContent = `${queueSize} image${queueSize > 1 ? 's' : ''} ready (${formatFileSize(totalSize)})`;

    // Render queue items
    queueList.innerHTML = AppState.uploadQueue.map(item => `
        <div class="queue-item" data-id="${item.id}">
            <img src="${item.preview}" alt="${item.file.name}" class="queue-item-preview">
            <div class="queue-item-info">
                <div class="queue-item-name">${item.file.name}</div>
                <div class="queue-item-size">${formatFileSize(item.file.size)}</div>
            </div>
            <button class="queue-item-remove" data-id="${item.id}" title="Remove">×</button>
        </div>
    `).join('');

    // Add event listeners to remove buttons
    queueList.querySelectorAll('.queue-item-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeFromQueue(btn.dataset.id);
        });
    });

    // Add 3D tilt effect and click handlers to preview images
    queueList.querySelectorAll('.queue-item-preview').forEach(img => {
        const item = AppState.uploadQueue.find(i => {
            const queueItem = img.closest('.queue-item');
            return queueItem && i.id === queueItem.dataset.id;
        });

        // 3D tilt effect on hover
        img.addEventListener('mousemove', (e) => {
            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / centerY * 15; // Max 15 degrees
            const rotateY = (centerX - x) / centerX * 15;

            img.style.transform = `scale(1.15) perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });

        img.addEventListener('mouseleave', () => {
            img.style.transform = '';
        });

        // Click to open preview modal
        img.addEventListener('click', (e) => {
            e.stopPropagation();
            if (item) {
                openImagePreview(item.preview, item.file.name, formatFileSize(item.file.size));
            }
        });
    });
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

function showToast(message, keepVisible = false) {
    const statusToast = document.getElementById('statusToast');

    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }

    statusToast.textContent = message;
    statusToast.classList.add('show');

    if (!keepVisible && !isProcessing) {
        toastTimeout = setTimeout(() => {
            statusToast.classList.remove('show');
        }, 2000);
    }
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
// Upload Queue Processor
// ============================================================================
class UploadQueue {
    constructor(maxConcurrent = 3, batchSize = 10) {
        this.maxConcurrent = maxConcurrent;
        this.batchSize = batchSize;
        this.active = 0;
        this.processed = 0;
        this.total = 0;
    }

    async processAll(files, onProgress) {
        this.processed = 0;
        this.total = files.length;

        // Split into batches
        const batches = [];
        for (let i = 0; i < files.length; i += this.batchSize) {
            batches.push(files.slice(i, i + this.batchSize));
        }

        // Process batches with concurrency limit
        const promises = [];
        for (let i = 0; i < batches.length; i++) {
            // Wait if we've hit concurrency limit
            while (this.active >= this.maxConcurrent) {
                await new Promise(r => setTimeout(r, 100));
            }

            this.active++;
            const promise = this.processBatch(batches[i], onProgress)
                .finally(() => this.active--);
            promises.push(promise);
        }

        await Promise.all(promises);
    }

    async processBatch(batch, onProgress) {
        const formData = new FormData();
        const parameters = getModelParameters();

        batch.forEach(item => formData.append('images', item.file));
        formData.append('model', AppState.selectedModel);
        formData.append('parameters', JSON.stringify(parameters));

        if (AppState.customPrompt) {
            formData.append('prompt', AppState.customPrompt);
        }

        try {
            const response = await fetch(`${AppState.apiBaseUrl}/generate/batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // Process results
            for (let i = 0; i < data.results.length; i++) {
                const result = data.results[i];
                const item = batch[i];

                if (result.success) {
                    onProgress({
                        item,
                        result: {
                            caption: result.caption,
                            image_preview: result.image_preview
                        }
                    });
                } else {
                    onProgress({
                        item,
                        error: result.error
                    });
                }
            }
        } catch (error) {
            // Handle batch error
            batch.forEach(item => {
                onProgress({ item, error: error.message });
            });
        }
    }
}

async function processImages() {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const pauseBtn = document.getElementById('pauseBtn');

    // Create progress bar
    const progressBar = document.createElement('div');
    progressBar.style.cssText = 'width: 100%; background: var(--surface-color); border-radius: 8px; overflow: hidden; margin-bottom: 20px;';
    progressBar.innerHTML = '<div id="progressFill" style="height: 8px; width: 0%; background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); transition: width 0.3s;"></div>';

    resultsGrid.innerHTML = '';
    resultsGrid.appendChild(progressBar);

    const progressMsg = document.createElement('p');
    progressMsg.style.cssText = 'text-align: center; color: var(--text-secondary); margin-top: 10px;';
    progressMsg.textContent = 'Processing images...';
    resultsGrid.appendChild(progressMsg);

    downloadBtn.style.display = 'none';
    processingControls.style.display = 'flex';

    if (pauseBtn) {
        pauseBtn.classList.remove('paused');
        pauseBtn.title = 'Pause processing';
    }

    AppState.resultsAutoFollow = true;
    AppState.processedResults = [];
    AppState.allResults = [];

    isProcessing = true;
    isPaused = false;
    shouldStop = false;
    let processedCount = 0;
    const totalImages = AppState.uploadQueue.length;

    const queue = new UploadQueue(3, 10);
    let firstResult = true;

    await queue.processAll(AppState.uploadQueue, (data) => {
        if (firstResult) {
            resultsGrid.innerHTML = '';
            firstResult = false;
        }

        if (data.error) {
            console.error(`Error: ${data.item.file.name}:`, data.error);
        } else {
            addResultToGrid(data.item, data.result);
            AppState.processedResults.push({
                filename: data.item.file.name,
                caption: data.result.caption
            });
        }

        processedCount++;
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.width = `${(processedCount / totalImages) * 100}%`;
        }
        showToast(`Processed ${processedCount}/${totalImages} (${Math.round(processedCount/totalImages*100)}%)`, true);
    });

    isProcessing = false;
    isPaused = false;
    shouldStop = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
    }

    if (processedCount === totalImages) {
        showToast('All images processed successfully!');
    }
}

function addResultToGrid(item, result) {
    AppState.allResults.push({ item, result });

    // Only render if we have less than 100 items (use virtual scroll after)
    if (AppState.allResults.length < 100) {
        renderResultItem(item, result);
    } else {
        renderVirtualResults();
    }
}

function renderResultItem(item, result) {
    const resultsGrid = document.getElementById('resultsGrid');

    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';

    resultItem.innerHTML = `
        <div class="result-image">
            <img src="${item.preview}" alt="${item.file.name}">
        </div>
        <div class="result-text">
            <p>${result.caption}</p>
        </div>
    `;

    const resultImage = resultItem.querySelector('.result-image');
    resultImage.addEventListener('click', (e) => {
        if (e.target === resultImage || e.target.tagName === 'IMG') {
            openImagePreview(item.preview, result.caption, item.file.name);
        }
    });

    const resultText = resultItem.querySelector('.result-text');
    resultText.addEventListener('click', (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(result.caption)
            .then(() => showToast('Caption copied to clipboard!'))
            .catch(() => showToast('Failed to copy caption'));
    });

    resultsGrid.appendChild(resultItem);
}

function renderVirtualResults() {
    const resultsGrid = document.getElementById('resultsGrid');
    const viewportHeight = window.innerHeight;
    const scrollTop = AppState.virtualScroll.scrollTop;
    const itemHeight = AppState.virtualScroll.itemHeight;
    const buffer = AppState.virtualScroll.buffer;

    const start = Math.max(0, Math.floor(scrollTop / itemHeight) - buffer);
    const end = Math.min(AppState.allResults.length, Math.ceil((scrollTop + viewportHeight) / itemHeight) + buffer);

    // Clear and render only visible items
    resultsGrid.innerHTML = '';
    resultsGrid.style.paddingTop = `${start * itemHeight}px`;
    resultsGrid.style.paddingBottom = `${(AppState.allResults.length - end) * itemHeight}px`;

    for (let i = start; i < end; i++) {
        const { item, result } = AppState.allResults[i];
        renderResultItem(item, result);
    }
}

function initVirtualScroll() {
    const resultsTab = document.getElementById('results-content');
    let ticking = false;

    resultsTab?.addEventListener('scroll', () => {
        AppState.virtualScroll.scrollTop = resultsTab.scrollTop;

        if (!ticking && AppState.allResults.length >= 100) {
            window.requestAnimationFrame(() => {
                renderVirtualResults();
                ticking = false;
            });
            ticking = true;
        }
    });
}

// ============================================================================
// Results Grid Auto-Follow Behavior
// ============================================================================

function initResultsAutoFollow() {
}

// ============================================================================
// Download Functionality
// ============================================================================

async function downloadAllResults() {
    if (AppState.processedResults.length === 0) {
        showToast('No results to download');
        return;
    }

    const downloadBtn = document.getElementById('downloadAllBtn');
    const originalText = downloadBtn.innerHTML;

    try {
        // Show progress feedback
        downloadBtn.innerHTML = '<span class="download-icon">⏳</span> Creating ZIP...';
        downloadBtn.disabled = true;

        // Create a new JSZip instance
        const zip = new JSZip();

        // Add each description as a .txt file with matching name
        AppState.processedResults.forEach((result) => {
            // Get the exact filename and replace image extension with .txt
            const txtFilename = result.filename.replace(/\.[^/.]+$/, '.txt');

            // Add file to ZIP
            zip.file(txtFilename, result.caption);
        });

        // Generate the ZIP file
        showToast('Generating ZIP file...', true);
        const blob = await zip.generateAsync({
            type: 'blob',
            compression: 'DEFLATE',
            compressionOptions: { level: 9 }
        });

        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Name the ZIP file with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        a.download = `image_descriptions_${timestamp}.zip`;

        // Trigger download
        document.body.appendChild(a);
        a.click();

        // Cleanup
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast(`Downloaded ${AppState.processedResults.length} descriptions as ZIP`);

    } catch (error) {
        console.error('Error creating ZIP file:', error);
        showToast('Failed to create ZIP file');
    } finally {
        // Restore button state
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    }
}

function initDownloadButton() {
    const downloadBtn = document.getElementById('downloadAllBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadAllResults);
    }
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
    initResultsAutoFollow();
    initDownloadButton();
    initProcessingControls();
    initConfigModals();
    initVirtualScroll();

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
    const uploadCard = document.getElementById('uploadCard');
    const queueContainer = document.getElementById('uploadQueueContainer');
    const fileInput = document.getElementById('fileInput');
    const folderInput = document.getElementById('folderInput');
    const chooseFilesBtn = document.getElementById('chooseFilesBtn');
    const chooseFolderBtn = document.getElementById('chooseFolderBtn');
    const clearQueueBtn = document.getElementById('clearQueueBtn');
    const addMoreBtn = document.getElementById('addMoreBtn');

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Setup drag-and-drop for upload card (initial upload)
    setupDragAndDrop(uploadCard);

    // Setup drag-and-drop for queue container (add more to existing queue)
    setupDragAndDrop(queueContainer);

    function setupDragAndDrop(element) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight on drag over
        ['dragenter', 'dragover'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        element.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                addFilesToQueue(files);
            }
        }, false);
    }

    // Prevent default drag on body to avoid browser opening files
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // File input handlers
    chooseFilesBtn.addEventListener('click', () => fileInput.click());
    chooseFolderBtn.addEventListener('click', () => folderInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            addFilesToQueue(e.target.files);
            e.target.value = ''; // Reset input
        }
    });

    folderInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            addFilesToQueue(e.target.files);
            e.target.value = ''; // Reset input
        }
    });

    // Queue management buttons
    clearQueueBtn.addEventListener('click', clearQueue);
    addMoreBtn.addEventListener('click', () => fileInput.click());
}

// ============================================================================
// Copy Functionality
// ============================================================================

function initCopyFunctionality() {
    // Copy functionality will be implemented when results are generated
    // Event delegation will be used to handle dynamically created result items
}

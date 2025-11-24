// ============================================================================
// Upload Handlers
// ============================================================================

function initUploadHandlers() {
    const folderBrowser = document.getElementById('folderBrowser');
    const folderPathInput = document.getElementById('folderPathInput');
    const browseFolderBtn = document.getElementById('browseFolderBtn');
    const scanFolderBtn = document.getElementById('scanFolderBtn');
    const clearQueueBtn = document.getElementById('clearQueueBtn');
    const uploadCard = document.getElementById('uploadCard');

    // Initialize drag and drop
    initDragAndDrop(uploadCard);

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

            // Reset the input value to allow re-uploading the same files
            e.target.value = '';
        }
    });

    // Scan folder button (if user manually enters path)
    if (scanFolderBtn) {
        scanFolderBtn.addEventListener('click', scanFolder);
    }

    // Clear queue button
    clearQueueBtn.addEventListener('click', clearQueue);
}

// ============================================================================
// Drag and Drop Handlers
// ============================================================================

function initDragAndDrop(dropZone) {
    if (!dropZone) return;

    // Counter to handle nested elements (dragenter/dragleave fire for children)
    let dragCounter = 0;

    // Supported image types
    const supportedTypes = [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'image/bmp',
        'image/tiff'
    ];

    // Prevent default drag behaviors on document to avoid browser opening files
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Handle drag enter
    dropZone.addEventListener('dragenter', (e) => {
        preventDefaults(e);
        dragCounter++;
        if (dragCounter === 1) {
            dropZone.classList.add('drag-over');
        }
    }, false);

    // Handle drag over (needed to allow drop)
    dropZone.addEventListener('dragover', (e) => {
        preventDefaults(e);
        e.dataTransfer.dropEffect = 'copy';
    }, false);

    // Handle drag leave
    dropZone.addEventListener('dragleave', (e) => {
        preventDefaults(e);
        dragCounter--;
        if (dragCounter === 0) {
            dropZone.classList.remove('drag-over');
        }
    }, false);

    // Handle drop
    dropZone.addEventListener('drop', (e) => {
        preventDefaults(e);
        dragCounter = 0;
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length === 0) return;

        // Filter for image files only
        const imageFiles = Array.from(files).filter(file =>
            supportedTypes.includes(file.type) ||
            /\.(jpe?g|png|gif|webp|bmp|tiff?)$/i.test(file.name)
        );

        if (imageFiles.length === 0) {
            showToast('No valid image files found', 'warning');
            return;
        }

        // Convert filtered array back to FileList-like object for compatibility
        const dataTransfer = new DataTransfer();
        imageFiles.forEach(file => dataTransfer.items.add(file));

        scanFolderFromFileList(dataTransfer.files);

        if (imageFiles.length < files.length) {
            const skipped = files.length - imageFiles.length;
            showToast(`${imageFiles.length} images added (${skipped} non-image files skipped)`, 'info');
        }
    }, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

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

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
        // Use NEW session-based endpoint
        const response = await fetch(`${AppState.apiBaseUrl}/session/register-folder`, {
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

        // Add images to queue (NEW: using image_id instead of path)
        AppState.uploadQueue = data.images.map(img => ({
            image_id: img.image_id,
            filename: img.filename,
            size: img.size,
            uploaded: img.uploaded,
            width: img.width,
            height: img.height
        }));

        updateUploadGrid();
        updateInputNodes();
        showToast(`Found ${data.total} images`);

    } catch (error) {
        console.error('Error scanning folder:', error);
        showToast(error.message || 'Failed to scan folder', false, null, 'error');
    }
}

async function scanFolderFromFileList(files) {
    showToast('Processing files...', true, 0);

    try {
        // Clear existing queue
        AppState.uploadQueue = [];

        const fileArray = Array.from(files);
        const imageFiles = fileArray.filter(file => {
            if (!file.type || !file.type.startsWith('image/')) return false;
            const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
            return ['.jpg', '.jpeg', '.png', '.webp', '.bmp'].includes(ext);
        });

        if (imageFiles.length === 0) {
            showToast('No supported images found');
            return;
        }

        // Step 1: Pre-register files to get image_ids
        showToast(`Registering ${imageFiles.length} files...`, true, 0.1);

        const metadata = imageFiles.map(file => ({
            filename: file.name,
            size: file.size
        }));

        const registerResponse = await fetch(`${AppState.apiBaseUrl}/session/register-files`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: metadata })
        });

        if (!registerResponse.ok) {
            throw new Error('Failed to register files');
        }

        const { image_ids } = await registerResponse.json();

        // Step 2: Upload files in parallel batches for speed
        const BATCH_SIZE = 10;  // Files per batch
        const PARALLEL_BATCHES = 3;  // Number of batches to upload simultaneously
        let totalUploaded = 0;

        // Create all batches
        const batches = [];
        for (let i = 0; i < imageFiles.length; i += BATCH_SIZE) {
            batches.push({
                files: imageFiles.slice(i, i + BATCH_SIZE),
                ids: image_ids.slice(i, i + BATCH_SIZE),
                index: i
            });
        }

        // Upload batches in parallel chunks
        for (let chunkStart = 0; chunkStart < batches.length; chunkStart += PARALLEL_BATCHES) {
            const chunk = batches.slice(chunkStart, chunkStart + PARALLEL_BATCHES);

            const progress = 0.3 + (chunkStart / batches.length) * 0.6;
            const uploadedSoFar = chunkStart * BATCH_SIZE;
            showToast(`Uploading ${uploadedSoFar + 1}-${Math.min(uploadedSoFar + PARALLEL_BATCHES * BATCH_SIZE, imageFiles.length)}/${imageFiles.length}...`, true, progress);

            // Upload this chunk's batches in parallel
            const uploadPromises = chunk.map(async (batch) => {
                const formData = new FormData();
                batch.files.forEach(file => formData.append('files', file));
                batch.ids.forEach(id => formData.append('image_ids', id));

                const uploadResponse = await fetch(`${AppState.apiBaseUrl}/upload/batch`, {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    throw new Error(`Failed to upload batch`);
                }

                return await uploadResponse.json();
            });

            const results = await Promise.all(uploadPromises);
            totalUploaded += results.reduce((sum, r) => sum + r.uploaded, 0);
        }

        // Step 3: Update queue with image metadata
        AppState.uploadQueue = image_ids.map((image_id, i) => ({
            image_id: image_id,
            filename: imageFiles[i].name,
            size: imageFiles[i].size,
            uploaded: true,
            width: null,  // Will be populated by backend
            height: null
        }));

        // Get folder name from first file
        if (files.length > 0 && files[0].webkitRelativePath) {
            const folderName = files[0].webkitRelativePath.split('/')[0];
            document.getElementById('folderPathInput').value = folderName;
        }

        updateUploadGrid();
        updateInputNodes();
        showToast(`Uploaded ${totalUploaded} images`);

    } catch (error) {
        console.error('Error processing files:', error);
        showToast(error.message || 'Failed to process files', false, null, 'error');
    }
}

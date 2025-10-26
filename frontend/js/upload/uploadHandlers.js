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

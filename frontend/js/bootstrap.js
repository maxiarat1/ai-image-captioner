async function autoResumeSession() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/images?page=1&per_page=1000`);
        if (!response.ok) {
            console.warn('Failed to fetch session images');
            return;
        }

        const data = await response.json();

        if (data.total > 0) {
            AppState.uploadQueue = data.images;

            const imagesWithCaptions = data.images.filter(img => img.caption);
            if (imagesWithCaptions.length > 0) {
                AppState.allResults = imagesWithCaptions.map(img => ({
                    queueItem: img,
                    data: { caption: img.caption }
                }));

                AppState.processedResults = imagesWithCaptions.map(img => ({
                    filename: img.filename,
                    caption: img.caption,
                    path: img.filename
                }));

                if (typeof renderCurrentPage === 'function') {
                    renderCurrentPage();
                }

                const downloadBtn = document.getElementById('downloadAllBtn');
                if (downloadBtn) {
                    downloadBtn.style.display = 'inline-flex';
                }
            }

            const totalSize = data.images.reduce((sum, img) => sum + (img.size || 0), 0);
            const sizeText = formatFileSize(totalSize);

            if (typeof updateUploadGrid === 'function') {
                updateUploadGrid();
            }
            if (typeof updateInputNodes === 'function') {
                updateInputNodes();
            }

            const captionText = imagesWithCaptions.length > 0 ? ` (${imagesWithCaptions.length} with captions)` : '';
            showToast(`Session resumed: ${data.total} images (${sizeText})${captionText}`, false);

            console.log(`Auto-resumed session with ${data.total} images, ${imagesWithCaptions.length} with captions`);
        }
    } catch (error) {
        console.error('Error auto-resuming session:', error);
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    if (typeof fetchAvailableModels === 'function') {
        await fetchAvailableModels();
    }

    await autoResumeSession();

    initTabNavigation();
    initThemeToggle();

    initUploadHandlers();

    initNodeEditor();
    if (typeof NEContextMenu !== 'undefined') NEContextMenu.init();

    initCopyFunctionality();
    initExportModal();

    initProcessingControls();
    initPaginationControls();
    initSearchHandlers();

    initModalHandlers();
    initDownloadButton();
    initToastHoverBehavior();

    if (window.Logger) {
        Logger.info('AI Image Tagger with Node Editor initialized');
    }
});

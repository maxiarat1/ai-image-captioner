// ============================================================================
// Lazy Loading for Grids - Unified for Upload and Results tabs
// ============================================================================

// Unified lazy loading for both upload and results grids
function setupLazyLoadingForGrid(gridId) {
    const grid = document.getElementById(gridId);
    const containers = grid.querySelectorAll('.result-image[data-image-id]');

    if (containers.length === 0) {
        return;
    }

    const isResultsGrid = gridId === 'resultsGrid';

    const observer = new IntersectionObserver(async (entries) => {
        for (const entry of entries) {
            const container = entry.target;
            const image_id = container.dataset.imageId;

            if (!image_id) continue;

            if (entry.isIntersecting) {
                // LOAD: Element is visible, load full-resolution image from backend
                const fullImage = await loadFullImage(image_id);

                if (fullImage) {
                    // Replace only the thumbnail placeholder with the actual image
                    const placeholder = container.querySelector('.thumbnail-placeholder');
                    if (placeholder) {
                        const img = document.createElement('img');
                        img.src = fullImage;
                        img.alt = 'Image';
                        placeholder.replaceWith(img);
                        
                        // Add click handler to the image
                        img.addEventListener('click', async () => {
                            if (isResultsGrid) {
                                // Results: Find caption from AppState
                                const resultData = AppState.allResults.find(r => r.queueItem.image_id === image_id);
                                const caption = resultData?.data?.caption || '';
                                const filename = resultData?.queueItem?.filename || '';
                                openImagePreview(fullImage, caption, filename);
                            } else {
                                // Upload: Find item from upload queue
                                const item = AppState.uploadQueue.find(i => i.image_id === image_id);
                                if (item) {
                                    openImagePreview(fullImage, item.filename, formatFileSize(item.size));
                                }
                            }
                        });
                    }
                }
            }
            // Note: Browser HTTP cache handles caching automatically
        }
    }, {
        rootMargin: '500px', // Load well before visible for smoother scrolling
        threshold: 0
    });

    containers.forEach(container => observer.observe(container));
}

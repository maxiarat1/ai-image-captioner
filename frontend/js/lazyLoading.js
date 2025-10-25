// ============================================================================
// Lazy Loading for Grids - Unified for Upload and Results tabs
// ============================================================================

// Unified lazy loading for both upload and results grids
function setupLazyLoadingForGrid(gridId) {
    const grid = document.getElementById(gridId);
    // Support both upload and results containers
    const containers = grid.querySelectorAll('[data-image-id].upload-thumbnail-container, [data-image-id].result-thumbnail-container');

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
                    // Display image
                    container.innerHTML = `<img src="${fullImage}" alt="Image" style="width: 100%; height: 100%; object-fit: cover; border-radius: var(--radius-md);">`;

                    const img = container.querySelector('img');
                    if (img) {
                        // Check aspect ratio for stretched images (results only)
                        if (isResultsGrid) {
                            img.addEventListener('load', () => {
                                const aspectRatio = img.naturalWidth / img.naturalHeight;
                                if (aspectRatio > 2.5) {
                                    const resultItem = container.closest('.result-item');
                                    if (resultItem) {
                                        resultItem.classList.add('stretched-image');
                                    }
                                }
                            });
                        }

                        // Add click handler based on grid type
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

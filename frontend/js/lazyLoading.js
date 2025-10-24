// ============================================================================
// Lazy Loading for Grids - Simplified session-based version
// ============================================================================

// Unified lazy loading for both upload and results grids
function setupLazyLoadingForGrid(gridId) {
    const grid = document.getElementById(gridId);
    const containers = grid.querySelectorAll('.upload-thumbnail-container[data-image-id]');

    if (containers.length === 0) {
        return;
    }

    const observer = new IntersectionObserver(async (entries) => {
        for (const entry of entries) {
            const container = entry.target;
            const image_id = container.dataset.imageId;

            if (!image_id) continue;

            const item = AppState.uploadQueue.find(i => i.image_id === image_id);
            if (!item) continue;

            if (entry.isIntersecting) {
                // LOAD: Element is visible, load thumbnail from backend
                const thumbnail = await loadThumbnail(image_id);

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
            // Note: Browser HTTP cache handles caching automatically
        }
    }, {
        rootMargin: '500px', // Load well before visible for smoother scrolling
        threshold: 0
    });

    containers.forEach(container => observer.observe(container));
}

// ============================================================================
// Lazy Loading for Grids
// ============================================================================

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

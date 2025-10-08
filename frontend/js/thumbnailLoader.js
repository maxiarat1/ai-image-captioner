// ============================================================================
// Thumbnail Loading and Caching
// ============================================================================

async function loadThumbnail(path) {
    // Check cache first (Map maintains insertion order for LRU)
    if (AppState.thumbnailCache.has(path)) {
        // Move to end (most recently used)
        const thumbnail = AppState.thumbnailCache.get(path);
        AppState.thumbnailCache.delete(path);
        AppState.thumbnailCache.set(path, thumbnail);
        return thumbnail;
    }

    try {
        const response = await fetch(`${AppState.apiBaseUrl}/image/thumbnail?path=${encodeURIComponent(path)}`);
        if (!response.ok) throw new Error('Failed to load thumbnail');

        const data = await response.json();

        // Add to cache with LRU eviction
        if (AppState.thumbnailCache.size >= AppState.thumbnailCacheMaxSize) {
            // Remove oldest (first) entry
            const firstKey = AppState.thumbnailCache.keys().next().value;
            AppState.thumbnailCache.delete(firstKey);
        }

        AppState.thumbnailCache.set(path, data.thumbnail);
        return data.thumbnail;
    } catch (error) {
        console.error('Error loading thumbnail:', error);
        return null;
    }
}

async function loadThumbnailFromFile(file) {
    // Check cache first by file name
    const cacheKey = `file:${file.name}:${file.size}`;
    if (AppState.thumbnailCache.has(cacheKey)) {
        const thumbnail = AppState.thumbnailCache.get(cacheKey);
        AppState.thumbnailCache.delete(cacheKey);
        AppState.thumbnailCache.set(cacheKey, thumbnail);
        return thumbnail;
    }

    try {
        // Create thumbnail from File object
        const reader = new FileReader();
        const thumbnail = await new Promise((resolve, reject) => {
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });

        // Add to cache with LRU eviction
        if (AppState.thumbnailCache.size >= AppState.thumbnailCacheMaxSize) {
            const firstKey = AppState.thumbnailCache.keys().next().value;
            AppState.thumbnailCache.delete(firstKey);
        }

        AppState.thumbnailCache.set(cacheKey, thumbnail);
        return thumbnail;
    } catch (error) {
        console.error('Error loading thumbnail from file:', error);
        return null;
    }
}

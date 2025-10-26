// ============================================================================
// Image Loading - Unified session-based version
// ============================================================================

async function loadFullImage(image_id) {
    // Load full-resolution image for grid display and modal preview
    // Browser HTTP cache will handle caching automatically
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/image/${image_id}`);

        // 404 is expected during upload - image registered but not uploaded yet
        if (response.status === 404) {
            return null;
        }

        if (!response.ok) {
            throw new Error(`Failed to load image: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        return data.image;
    } catch (error) {
        // Only log non-404 errors (404 is normal during upload process)
        if (!error.message.includes('404')) {
            console.error('Error loading image:', error);
        }
        return null;
    }
}

// ============================================================================
// Image Loading - Unified session-based version
// ============================================================================

async function loadFullImage(image_id) {
    // Load full-resolution image for grid display and modal preview
    // Browser HTTP cache will handle caching automatically
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/image/${image_id}`);
        if (!response.ok) throw new Error('Failed to load image');

        const data = await response.json();
        return data.image;
    } catch (error) {
        console.error('Error loading image:', error);
        return null;
    }
}

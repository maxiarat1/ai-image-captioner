// ============================================================================
// Thumbnail Loading - Simplified session-based version
// ============================================================================

async function loadThumbnail(image_id) {
    // NEW: Simple endpoint using image_id
    // Browser HTTP cache will handle caching automatically
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/image/${image_id}/thumbnail`);
        if (!response.ok) throw new Error('Failed to load thumbnail');

        const data = await response.json();
        return data.thumbnail;
    } catch (error) {
        console.error('Error loading thumbnail:', error);
        return null;
    }
}

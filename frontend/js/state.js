// ============================================================================
// Application State
// ============================================================================
const AppState = {
    // NEW: Session-based upload queue (no File objects, just metadata)
    uploadQueue: [], // Array of {image_id, filename, size, uploaded, width, height}
    apiBaseUrl: 'http://localhost:5000',
    selectedModel: 'blip',
    customPrompt: '',
    availableModels: [],
    processedResults: [], // Store processed results: {filename, caption, path}
    userConfig: null,
    allResults: [], // All result items for pagination: {queueItem, data}
    currentPage: 1,
    itemsPerPage: 12,
    // REMOVED: thumbnailCache - now handled by backend
    uploadCurrentPage: 1, // Separate pagination for upload grid
    uploadSearchQuery: '', // Search query for upload tab
    resultsSearchQuery: '' // Search query for results tab
};

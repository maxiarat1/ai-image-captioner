// ============================================================================
// Application State
// ============================================================================
const AppState = {
    uploadQueue: [], // Array of {id, filename, size, path, file, thumbnail: null}
    apiBaseUrl: 'http://localhost:5000',
    selectedModel: 'blip',
    customPrompt: '',
    availableModels: [],
    processedResults: [], // Store processed results: {filename, caption, path}
    userConfig: null,
    allResults: [], // All result items for pagination: {queueItem, data}
    currentPage: 1,
    itemsPerPage: 12,
    thumbnailCache: new Map(), // Cache loaded thumbnails with LRU
    thumbnailCacheMaxSize: 300, // Max 300 thumbnails total (~45MB) - shared between tabs
    uploadCurrentPage: 1, // Separate pagination for upload grid
    uploadSearchQuery: '', // Search query for upload tab
    resultsSearchQuery: '' // Search query for results tab
};

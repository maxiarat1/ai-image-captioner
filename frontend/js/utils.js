// ============================================================================
// Utility Functions
// ============================================================================

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Fetch available models from backend
 * Updates AppState.availableModels with the list
 * @returns {Promise<Array>} List of available models
 */
async function fetchAvailableModels() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        const data = await response.json();

        // Store in AppState
        AppState.availableModels = data.models || [];

        console.log('Available models loaded:', AppState.availableModels);
        return AppState.availableModels;
    } catch (error) {
        console.error('Error fetching models:', error);
        // Fallback to hardcoded models if fetch fails
        AppState.availableModels = [
            { name: 'blip', loaded: false, description: 'BLIP - Fast captioning' },
            { name: 'r4b', loaded: false, description: 'R-4B - Advanced reasoning' },
            { name: 'qwen3vl-4b', loaded: false, description: 'Qwen3-VL 4B' },
            { name: 'qwen3vl-8b', loaded: false, description: 'Qwen3-VL 8B' },
            { name: 'wdvit', loaded: false, description: 'WD-ViT Tagger v3' },
            { name: 'wdeva02', loaded: false, description: 'WD-EVA02 Tagger v3' }
        ];
        return AppState.availableModels;
    }
}

/**
 * Get display name for a model (just the name, no description)
 * @param {string} modelName - Model identifier (e.g., 'blip', 'r4b')
 * @returns {string} Human-readable model name
 */
function getModelDisplayName(modelName) {
    const displayNames = {
        'blip': 'BLIP',
        'r4b': 'R-4B',
        'qwen3vl-4b': 'Qwen3-VL 4B',
        'qwen3vl-8b': 'Qwen3-VL 8B',
        'wdvit': 'WD-ViT v3',
        'wdeva02': 'WD-EVA02 v3'
    };
    return displayNames[modelName] || modelName.toUpperCase().replace(/-/g, ' ');
}

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

async function fetchAvailableModels() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        const data = await response.json();

        AppState.availableModels = data.models || [];

        console.log('Available models loaded:', AppState.availableModels);
        return AppState.availableModels;
    } catch (error) {
        console.error('Error fetching models:', error);
        AppState.availableModels = [
            { name: 'blip', loaded: false, description: 'BLIP - Fast captioning' },
            { name: 'r4b', loaded: false, description: 'R-4B - Advanced reasoning' },
            { name: 'wdvit', loaded: false, description: 'WD-ViT Tagger v3' },
            { name: 'wdeva02', loaded: false, description: 'WD-EVA02 Tagger v3' }
        ];
        return AppState.availableModels;
    }
}

function getModelDisplayName(modelName) {
    const displayNames = {
        'blip': 'BLIP',
        'r4b': 'R-4B',
        'wdvit': 'WD-ViT v3',
        'wdeva02': 'WD-EVA02 v3'
    };
    return displayNames[modelName] || modelName.toUpperCase().replace(/-/g, ' ');
}

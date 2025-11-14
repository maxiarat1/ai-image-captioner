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

        // Set default selected model to first available model
        if (AppState.availableModels.length > 0 && !AppState.selectedModel) {
            AppState.selectedModel = AppState.availableModels[0].name;
        }

        console.log('Available models loaded:', AppState.availableModels);
        return AppState.availableModels;
    } catch (error) {
        console.error('Error fetching models:', error);
        // Minimal fallback - empty list, will be populated from metadata API
        AppState.availableModels = [];
        return AppState.availableModels;
    }
}

function getModelDisplayName(modelName) {
    // Try to get display name from cached metadata (synchronous)
    if (ModelsAPI.metadata && ModelsAPI.metadata.models && ModelsAPI.metadata.models[modelName]) {
        const model = ModelsAPI.metadata.models[modelName];
        if (model.display_name) {
            return model.display_name;
        }
    }

    // Fallback: format the model name nicely
    return modelName
        .split('-')
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

async function getModelDisplayNameAsync(modelName) {
    // Async version for when you need to fetch fresh data
    try {
        const metadata = await ModelsAPI.getModel(modelName);
        if (metadata && metadata.display_name) {
            return metadata.display_name;
        }
    } catch (error) {
        console.warn('Could not fetch display name for model:', modelName);
    }

    return getModelDisplayName(modelName);
}

// Export for modules if needed (backwards compatible with global usage)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        generateId,
        formatFileSize,
        fetchAvailableModels,
        getModelDisplayName,
        getModelDisplayNameAsync
    };
}

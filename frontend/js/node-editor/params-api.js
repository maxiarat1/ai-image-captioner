// Functions for fetching model parameter metadata from the backend

async function fetchModelParameters(modelName) {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/model/info?model=${modelName}`);
        if (!response.ok) throw new Error('Failed to fetch model parameters');
        const data = await response.json();
        return data.parameters || [];
    } catch (error) {
        console.error('Error fetching model parameters:', error);
        return [];
    }
}

// ============================================================================
// Configuration Management
// ============================================================================

async function loadUserConfig() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/config`);
        if (!response.ok) {
            throw new Error('Failed to load user configuration');
        }
        const data = await response.json();
        AppState.userConfig = data.config;
        console.log('User configuration loaded:', AppState.userConfig);
    } catch (error) {
        console.error('Error loading user config:', error);
        showToast('Failed to load user configuration');
    }
}

async function saveUserConfig() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(AppState.userConfig)
        });

        if (!response.ok) {
            throw new Error('Failed to save user configuration');
        }

        const data = await response.json();
        console.log('Configuration saved successfully:', data);
        return true;
    } catch (error) {
        console.error('Error saving user config:', error);
        showToast('Failed to save configuration');
        return false;
    }
}

function loadConfiguration(configId) {
    if (!AppState.userConfig) return;

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel] || {};
    const config = Object.values(modelConfigs).find(c => c.id === configId);

    if (!config) {
        showToast('Configuration not found');
        return;
    }

    // Apply parameters to the UI
    applyParametersToUI(config.parameters);
    showToast(`Loaded: ${config.name}`);
}

function applyParametersToUI(parameters) {
    const container = document.getElementById('dynamicParams');

    // Get all inputs in the container
    const numberInputs = container.querySelectorAll('input[type="number"]');
    const selects = container.querySelectorAll('select');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');

    // First, clear all inputs to their default state
    numberInputs.forEach(input => {
        input.value = '';
    });

    selects.forEach(select => {
        select.value = '';
    });

    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });

    // Then apply the values from the configuration
    Object.keys(parameters).forEach(key => {
        const value = parameters[key];

        // Skip null or undefined values (they've already been cleared)
        if (value === null || value === undefined) return;

        const input = container.querySelector(`#${key}`);
        if (input) {
            if (input.type === 'number') {
                input.value = value;
            } else if (input.type === 'checkbox') {
                input.checked = value;
            } else if (input.tagName === 'SELECT') {
                input.value = value;
            }
        }
    });
}

async function saveConfiguration() {
    const configName = document.getElementById('configName').value.trim();

    if (!configName) {
        showToast('Please enter a configuration name');
        return;
    }

    const parameters = getModelParameters();
    const configId = configName.toLowerCase().replace(/\s+/g, '_');

    // Ensure the structure exists
    if (!AppState.userConfig.savedConfigurations[AppState.selectedModel]) {
        AppState.userConfig.savedConfigurations[AppState.selectedModel] = {};
    }

    // Create new configuration
    const newConfig = {
        id: configId,
        name: configName,
        model: AppState.selectedModel,
        parameters: parameters,
        createdAt: new Date().toISOString()
    };

    AppState.userConfig.savedConfigurations[AppState.selectedModel][configId] = newConfig;

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Configuration "${configName}" saved successfully`);
        closeModal(document.getElementById('saveConfigModal'));
    }
}

async function deleteConfiguration(configId) {
    if (!AppState.userConfig) return;

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel];
    if (!modelConfigs || !modelConfigs[configId]) {
        showToast('Configuration not found');
        return;
    }

    const configName = modelConfigs[configId].name;

    // Delete the configuration
    delete AppState.userConfig.savedConfigurations[AppState.selectedModel][configId];

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Deleted: ${configName}`);
    }
}

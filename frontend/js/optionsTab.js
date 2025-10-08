// ============================================================================
// Options Tab Functionality
// ============================================================================

async function fetchAvailableModels() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        AppState.availableModels = data.models;
        console.log('Available models:', AppState.availableModels);
    } catch (error) {
        console.error('Error fetching models:', error);
        showToast('Failed to load models from server');
    }
}

async function initOptionsHandlers() {
    const modelSelect = document.getElementById('modelSelect');
    const promptInput = document.getElementById('promptInput');
    const processImagesBtn = document.getElementById('processImagesBtn');
    const modelDescription = document.getElementById('modelDescription');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedContent = document.getElementById('advancedContent');

    // Model descriptions
    const modelDescriptions = {
        'blip': 'Fast, basic image captioning',
        'r4b': 'Advanced reasoning model with configurable parameters'
    };

    // Handle advanced settings toggle
    advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('active');
        advancedContent.classList.toggle('open');
    });

    // Handle model selection change
    modelSelect.addEventListener('change', async (e) => {
        AppState.selectedModel = e.target.value;
        modelDescription.textContent = modelDescriptions[e.target.value] || '';

        // Load parameters for the selected model
        await loadModelParameters(e.target.value);

        console.log('Model selected:', AppState.selectedModel);
    });

    // Handle prompt input change
    promptInput.addEventListener('input', (e) => {
        AppState.customPrompt = e.target.value.trim();
    });

    // Handle process images button
    processImagesBtn.addEventListener('click', async () => {
        if (AppState.uploadQueue.length === 0) {
            showToast('Please upload images first');
            return;
        }

        // Switch to Results tab
        const resultsTab = document.querySelector('.tab-btn[data-tab="results"]');
        resultsTab.click();

        // Start processing images
        await processImages();
    });

    // Load initial parameters for default model (BLIP)
    await loadModelParameters(AppState.selectedModel);
}

async function loadModelParameters(modelName) {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/model/info?model=${modelName}`);
        if (!response.ok) {
            throw new Error('Failed to fetch model parameters');
        }
        const data = await response.json();

        // Render parameters dynamically
        renderParameters(data.parameters);
    } catch (error) {
        console.error('Error loading model parameters:', error);
        showToast('Failed to load model parameters');
    }
}

function renderParameters(parameters) {
    const container = document.getElementById('dynamicParams');
    container.innerHTML = '';

    if (!parameters || parameters.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No parameters available for this model</p>';
        return;
    }

    parameters.forEach(param => {
        const paramGroup = document.createElement('div');
        paramGroup.className = 'param-group';

        // Make select and checkbox inputs full-width for better readability
        if (param.type === 'select' || param.type === 'checkbox') {
            paramGroup.classList.add('full-width');
        }

        // Create label
        const label = document.createElement('label');
        label.setAttribute('for', param.param_key);
        label.textContent = param.name;
        paramGroup.appendChild(label);

        // Create input based on type
        if (param.type === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.id = param.param_key;
            input.className = 'param-input';
            input.setAttribute('min', param.min);
            input.setAttribute('max', param.max);
            input.setAttribute('step', param.step);
            input.placeholder = 'Default';
            paramGroup.appendChild(input);
        } else if (param.type === 'select') {
            const select = document.createElement('select');
            select.id = param.param_key;
            select.className = 'param-select';

            // Add default option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Use default';
            select.appendChild(defaultOption);

            // Add other options
            param.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                select.appendChild(option);
            });

            paramGroup.appendChild(select);
        } else if (param.type === 'checkbox') {
            const checkboxLabel = document.createElement('label');
            checkboxLabel.className = 'checkbox-label';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = param.param_key;

            const span = document.createElement('span');
            span.textContent = param.name;

            checkboxLabel.appendChild(checkbox);
            checkboxLabel.appendChild(span);
            paramGroup.appendChild(checkboxLabel);
        }

        // Create description
        const description = document.createElement('p');
        description.className = 'param-description';
        description.textContent = param.description;
        paramGroup.appendChild(description);

        container.appendChild(paramGroup);
    });
}

function getModelParameters() {
    const parameters = {};
    const container = document.getElementById('dynamicParams');

    // Get all parameter inputs
    const numberInputs = container.querySelectorAll('input[type="number"]');
    const selects = container.querySelectorAll('select');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');

    // Collect number inputs
    numberInputs.forEach(input => {
        if (input.value) {
            const value = parseFloat(input.value);
            if (!isNaN(value)) {
                parameters[input.id] = value;
            }
        }
    });

    // Collect select values
    selects.forEach(select => {
        if (select.value) {
            parameters[select.id] = select.value;
        }
    });

    // Collect checkboxes
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            parameters[checkbox.id] = true;
        }
    });

    return parameters;
}

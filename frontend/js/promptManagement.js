// ============================================================================
// Prompt Management
// ============================================================================

function openLoadPromptModal() {
    const modal = document.getElementById('loadPromptModal');
    const promptList = document.getElementById('promptList');

    if (!AppState.userConfig || !AppState.userConfig.customPrompts || AppState.userConfig.customPrompts.length === 0) {
        promptList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No saved prompts available</p>';
        openModal(modal);
        return;
    }

    const prompts = AppState.userConfig.customPrompts;

    promptList.innerHTML = prompts.map(prompt => `
        <div class="config-item" data-prompt-id="${prompt.id}">
            <div class="config-item-info">
                <div class="config-item-name">${prompt.name}</div>
                <div class="config-item-date">${new Date(prompt.createdAt).toLocaleDateString()}</div>
            </div>
            <button class="config-item-delete" data-prompt-id="${prompt.id}" title="Delete prompt">×</button>
        </div>
    `).join('');

    // Add click handlers for loading
    promptList.querySelectorAll('.config-item').forEach(item => {
        item.addEventListener('click', (e) => {
            // Don't trigger if clicking delete button
            if (e.target.closest('.config-item-delete')) return;

            const promptId = item.dataset.promptId;
            loadPrompt(promptId);
            closeModal(modal);
        });
    });

    // Add click handlers for delete buttons
    promptList.querySelectorAll('.config-item-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const promptId = btn.dataset.promptId;
            await deletePrompt(promptId);
            openLoadPromptModal(); // Refresh the list
        });
    });

    openModal(modal);
}

function loadPrompt(promptId) {
    if (!AppState.userConfig) return;

    const prompt = AppState.userConfig.customPrompts.find(p => p.id === promptId);

    if (!prompt) {
        showToast('Prompt not found');
        return;
    }

    const promptInput = document.getElementById('promptInput');
    promptInput.value = prompt.text;
    AppState.customPrompt = prompt.text;
    showToast(`Loaded prompt: ${prompt.name}`);
}

function openSavePromptModal() {
    const promptInput = document.getElementById('promptInput');
    const currentPrompt = promptInput.value.trim();

    if (!currentPrompt) {
        showToast('Please enter a prompt before saving');
        return;
    }

    const modal = document.getElementById('savePromptModal');
    const promptName = document.getElementById('promptName');
    promptName.value = '';
    openModal(modal);
}

async function savePrompt() {
    const promptName = document.getElementById('promptName').value.trim();
    const promptInput = document.getElementById('promptInput');
    const promptText = promptInput.value.trim();

    if (!promptName) {
        showToast('Please enter a prompt name');
        return;
    }

    if (!promptText) {
        showToast('Prompt is empty');
        return;
    }

    const promptId = promptName.toLowerCase().replace(/\s+/g, '-');

    // Ensure customPrompts array exists
    if (!AppState.userConfig.customPrompts) {
        AppState.userConfig.customPrompts = [];
    }

    // Check if prompt already exists
    const existingIndex = AppState.userConfig.customPrompts.findIndex(p => p.id === promptId);

    const newPrompt = {
        id: promptId,
        name: promptName,
        text: promptText,
        createdAt: new Date().toISOString()
    };

    if (existingIndex >= 0) {
        // Update existing prompt
        AppState.userConfig.customPrompts[existingIndex] = newPrompt;
    } else {
        // Add new prompt
        AppState.userConfig.customPrompts.push(newPrompt);
    }

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Prompt "${promptName}" saved successfully`);
        closeModal(document.getElementById('savePromptModal'));
    }
}

async function deletePrompt(promptId) {
    if (!AppState.userConfig || !AppState.userConfig.customPrompts) return;

    const promptIndex = AppState.userConfig.customPrompts.findIndex(p => p.id === promptId);

    if (promptIndex === -1) {
        showToast('Prompt not found');
        return;
    }

    const promptName = AppState.userConfig.customPrompts[promptIndex].name;

    // Delete the prompt
    AppState.userConfig.customPrompts.splice(promptIndex, 1);

    // Save to backend
    const success = await saveUserConfig();

    if (success) {
        showToast(`Deleted: ${promptName}`);
    }
}

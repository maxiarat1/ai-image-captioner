// ============================================================================
// Configuration Modals
// ============================================================================

function initConfigModals() {
    const loadConfigModal = document.getElementById('loadConfigModal');
    const saveConfigModal = document.getElementById('saveConfigModal');
    const loadPromptModal = document.getElementById('loadPromptModal');
    const savePromptModal = document.getElementById('savePromptModal');

    const closeLoadConfig = document.getElementById('closeLoadConfig');
    const closeSaveConfig = document.getElementById('closeSaveConfig');
    const closeLoadPrompt = document.getElementById('closeLoadPrompt');
    const closeSavePrompt = document.getElementById('closeSavePrompt');

    const loadConfigBtn = document.getElementById('loadConfigBtn');
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    const loadPromptBtn = document.getElementById('loadPromptBtn');
    const savePromptBtn = document.getElementById('savePromptBtn');

    const cancelSaveConfig = document.getElementById('cancelSaveConfig');
    const confirmSaveConfig = document.getElementById('confirmSaveConfig');
    const cancelSavePrompt = document.getElementById('cancelSavePrompt');
    const confirmSavePrompt = document.getElementById('confirmSavePrompt');

    // Open modals
    loadConfigBtn?.addEventListener('click', () => openLoadConfigModal());
    saveConfigBtn?.addEventListener('click', () => openSaveConfigModal());
    loadPromptBtn?.addEventListener('click', () => openLoadPromptModal());
    savePromptBtn?.addEventListener('click', () => openSavePromptModal());

    // Close modals
    closeLoadConfig?.addEventListener('click', () => closeModal(loadConfigModal));
    closeSaveConfig?.addEventListener('click', () => closeModal(saveConfigModal));
    closeLoadPrompt?.addEventListener('click', () => closeModal(loadPromptModal));
    closeSavePrompt?.addEventListener('click', () => closeModal(savePromptModal));
    cancelSaveConfig?.addEventListener('click', () => closeModal(saveConfigModal));
    cancelSavePrompt?.addEventListener('click', () => closeModal(savePromptModal));

    // Close on backdrop click
    [loadConfigModal, saveConfigModal, loadPromptModal, savePromptModal].forEach(modal => {
        if (modal) {
            const backdrop = modal.querySelector('.config-modal-backdrop');
            backdrop?.addEventListener('click', (e) => {
                if (e.target === backdrop) {
                    closeModal(modal);
                }
            });
        }
    });

    // Close on Esc key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (loadConfigModal.classList.contains('active')) closeModal(loadConfigModal);
            if (saveConfigModal.classList.contains('active')) closeModal(saveConfigModal);
            if (loadPromptModal.classList.contains('active')) closeModal(loadPromptModal);
            if (savePromptModal.classList.contains('active')) closeModal(savePromptModal);
        }
    });

    // Save configuration and prompt
    confirmSaveConfig?.addEventListener('click', () => saveConfiguration());
    confirmSavePrompt?.addEventListener('click', () => savePrompt());
}

function openModal(modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal(modal) {
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

function openLoadConfigModal() {
    const modal = document.getElementById('loadConfigModal');
    const configList = document.getElementById('configList');

    if (!AppState.userConfig) {
        configList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No configurations available</p>';
        openModal(modal);
        return;
    }

    const modelConfigs = AppState.userConfig.savedConfigurations[AppState.selectedModel] || {};
    const configs = Object.values(modelConfigs);

    if (configs.length === 0) {
        configList.innerHTML = `<p style="text-align: center; color: var(--text-secondary);">No saved configurations for ${AppState.selectedModel.toUpperCase()}</p>`;
    } else {
        configList.innerHTML = configs.map(config => `
            <div class="config-item" data-config-id="${config.id}">
                <div class="config-item-info">
                    <div class="config-item-name">${config.name}</div>
                    <div class="config-item-date">${new Date(config.createdAt).toLocaleDateString()}</div>
                </div>
                <button class="config-item-delete" data-config-id="${config.id}" title="Delete configuration">×</button>
            </div>
        `).join('');

        // Add click handlers for loading
        configList.querySelectorAll('.config-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking delete button
                if (e.target.closest('.config-item-delete')) return;

                const configId = item.dataset.configId;
                loadConfiguration(configId);
                closeModal(modal);
            });
        });

        // Add click handlers for delete buttons
        configList.querySelectorAll('.config-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const configId = btn.dataset.configId;
                await deleteConfiguration(configId);
                openLoadConfigModal(); // Refresh the list
            });
        });
    }

    openModal(modal);
}

function openSaveConfigModal() {
    const modal = document.getElementById('saveConfigModal');
    const configName = document.getElementById('configName');
    configName.value = '';
    openModal(modal);
}

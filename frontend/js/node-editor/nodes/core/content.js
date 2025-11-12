// Node content generators moved from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

    NENodes.getNodeContent = function(node) {
        if (node.type === 'input') {
            const imageCount = AppState.uploadQueue ? AppState.uploadQueue.length : 0;
            return `
                <div class="node-info">
                    <span class="node-info-label">Images Ready</span>
                    <span class="node-info-value">${imageCount}</span>
                </div>
            `;
        }
        if (node.type === 'prompt') {
            return `<textarea id="node-${node.id}-prompt" name="prompt-${node.id}" data-key="text" placeholder="Enter prompt...">${node.data.text || ''}</textarea>`;
        }
        if (node.type === 'conjunction') {
            const template = node.data.template || '';
            const showPreview = node.data.showPreview || false;
            const preview = NENodes.resolveConjunctionTemplate ? NENodes.resolveConjunctionTemplate(node) : '';

            return `
                <div class="conjunction-template-wrapper">
                    <textarea
                        id="node-${node.id}-template"
                        name="template-${node.id}"
                        data-key="template"
                        class="conjunction-template"
                        placeholder="Enter prompt template (use {Prompt}, {AI_Model}, etc.)">${template}</textarea>
                    <div id="node-${node.id}-highlights" class="conjunction-highlights"></div>
                </div>
                <button class="btn-preview" data-node-id="${node.id}">
                    ${showPreview ? '▼ Hide Preview' : '▶ Show Preview'}
                </button>
                <div class="conjunction-preview ${showPreview ? '' : 'hidden'}" id="preview-${node.id}">
                    <div class="conjunction-preview-label">Resolved Text:</div>
                    <div class="conjunction-preview-content">${preview || '<em style="color: var(--text-secondary);">Empty template</em>'}</div>
                    <div class="conjunction-preview-label" style="margin-top: 8px;">Recent outputs:</div>
                    <div class="conjunction-preview-history" id="preview-${node.id}-history"></div>
                </div>
            `;
        }
        if (node.type === 'aimodel') {
            const showAdvanced = node.data.showAdvanced || false;
            const availableModels = AppState.availableModels || [];
            const currentModel = node.data.model;

            // Find current model's category and display info
            const currentModelObj = availableModels.find(m => m.name === currentModel);
            const currentCategory = typeof ModelCategories !== 'undefined'
                ? ModelCategories.getCategoryForModel(currentModel)
                : null;

            const currentDisplayName = currentModel
                ? (typeof getModelDisplayName === 'function'
                    ? getModelDisplayName(currentModel)
                    : currentModel.toUpperCase())
                : 'Select Model';

            const categoryIcon = currentCategory ? currentCategory.icon : '⚙️';
            const categoryColor = currentCategory ? currentCategory.color : '#6366f1';

            // Build categorized dropdown HTML
            let categoriesHtml = '';
            if (typeof ModelCategories !== 'undefined') {
                categoriesHtml = ModelCategories.categories.map(category => {
                    const categoryModels = category.models
                        .filter(modelName => availableModels.some(m => m.name === modelName))
                        .map(modelName => {
                            const modelObj = availableModels.find(m => m.name === modelName);
                            const displayName = typeof getModelDisplayName === 'function'
                                ? getModelDisplayName(modelName)
                                : modelName.toUpperCase();
                            const isSelected = modelName === currentModel ? 'selected' : '';
                            const description = modelObj ? modelObj.description : '';
                            return `
                                <div class="model-item ${isSelected}"
                                     data-model="${modelName}"
                                     title="${description}">
                                    ${displayName}
                                </div>
                            `;
                        }).join('');

                    if (!categoryModels) return '';

                    return `
                        <div class="model-category" data-category="${category.id}">
                            <div class="model-category-header" style="--category-color: ${category.color}">
                                <span class="model-category-icon">${category.icon}</span>
                                <span class="model-category-name">${category.name}</span>
                                <span class="model-category-arrow">›</span>
                            </div>
                            <div class="model-category-submenu">
                                ${categoryModels}
                            </div>
                        </div>
                    `;
                }).join('');
            }

            let html = `
                <div class="model-select-wrapper" id="model-select-${node.id}">
                    <button class="model-select-btn" data-node-id="${node.id}" style="--category-color: ${categoryColor}">
                        <span class="model-select-icon">${categoryIcon}</span>
                        <span class="model-select-text">${currentDisplayName}</span>
                        <span class="model-select-arrow">▼</span>
                    </button>
                    <div class="model-dropdown" id="model-dropdown-${node.id}">
                        ${categoriesHtml}
                    </div>
                </div>
                <button class="btn-advanced" data-node-id="${node.id}">
                    ${showAdvanced ? '▼ Hide Advanced' : '▶ Show Advanced'}
                </button>
                <div class="model-parameters ${showAdvanced ? '' : 'hidden'}" id="params-${node.id}">
                    <div style="text-align: center; color: var(--text-secondary); padding: 8px; font-size: 0.75rem;">
                        Loading parameters...
                    </div>
                </div>
            `;
            return html;
        }
        if (node.type === 'curate') {
            const showAdvanced = node.data.showAdvanced || false;
            // Show ports if explicitly enabled, otherwise show if ports already exist
            const showPorts = (typeof node.data.showPorts !== 'undefined') ? node.data.showPorts : (Array.isArray(node.data.ports) && node.data.ports.length > 0);
            const availableModels = AppState.availableModels || [];
            const currentModel = node.data.model;
            const modelType = node.data.modelType || 'vlm';
            const ports = node.data.ports || [];

            // Filter models based on curate model type
            const filteredModels = NENodes.filterModelsForCurateType ? NENodes.filterModelsForCurateType(availableModels, modelType) : availableModels;

            // Find current model's category and display info
            const currentCategory = typeof ModelCategories !== 'undefined'
                ? ModelCategories.getCategoryForModel(currentModel)
                : null;

            // If there are no models for the selected curate model type, show an empty
            // model button (prevents showing a stale/incorrect model name) and disable it.
            const hasFilteredModels = Array.isArray(filteredModels) && filteredModels.length > 0;

            const currentDisplayName = hasFilteredModels
                ? (currentModel
                    ? (typeof getModelDisplayName === 'function'
                        ? getModelDisplayName(currentModel)
                        : currentModel.toUpperCase())
                    : 'Select Model')
                : '';

            const categoryIcon = hasFilteredModels ? (currentCategory ? currentCategory.icon : '🔀') : '';
            const categoryColor = currentCategory ? currentCategory.color : '#a855f7';

            // Build categorized dropdown HTML with filtered models
            let categoriesHtml = '';
            if (typeof ModelCategories !== 'undefined') {
                categoriesHtml = ModelCategories.categories.map(category => {
                    const categoryModels = category.models
                        .filter(modelName => filteredModels.some(m => m.name === modelName))
                        .map(modelName => {
                            const modelObj = availableModels.find(m => m.name === modelName);
                            const displayName = typeof getModelDisplayName === 'function'
                                ? getModelDisplayName(modelName)
                                : modelName.toUpperCase();
                            const isSelected = modelName === currentModel ? 'selected' : '';
                            const description = modelObj ? modelObj.description : '';
                            return `
                                <div class="model-item ${isSelected}"
                                     data-model="${modelName}"
                                     title="${description}">
                                    ${displayName}
                                </div>
                            `;
                        }).join('');

                    if (!categoryModels) return '';

                    return `
                        <div class="model-category" data-category="${category.id}">
                            <div class="model-category-header" style="--category-color: ${category.color}">
                                <span class="model-category-icon">${category.icon}</span>
                                <span class="model-category-name">${category.name}</span>
                                <span class="model-category-arrow">›</span>
                            </div>
                            <div class="model-category-submenu">
                                ${categoryModels}
                            </div>
                        </div>
                    `;
                }).join('');
            }

            // Build simplified ports HTML
            const portsHtml = ports.map((port, index) => {
                const labelId = `curate-${node.id}-port-${port.id}-label`;
                const instructionId = `curate-${node.id}-port-${port.id}-instruction`;

                return `
                    <div class="curate-port-item" data-port-id="${port.id}">
                        <div class="curate-port-header">
                            <input type="text"
                                   id="${labelId}"
                                   name="${labelId}"
                                   class="curate-port-label-input"
                                   data-port-id="${port.id}"
                                   value="${port.label}"
                                   placeholder="Port ${index + 1}">
                            <button class="curate-port-delete" data-port-id="${port.id}" title="Delete port">×</button>
                        </div>
                        <textarea id="${instructionId}"
                                  name="${instructionId}"
                                  class="curate-port-instruction"
                                  data-port-id="${port.id}"
                                  placeholder="Describe routing criteria..."
                                  rows="2">${port.instruction || ''}</textarea>
                    </div>
                `;
            }).join('');

            let html = `
                <div class="curate-model-type-selector">
                    <select id="curate-${node.id}-model-type"
                            name="curate-${node.id}-model-type"
                            class="curate-model-type-select"
                            data-key="modelType">
                        <option value="vlm" ${modelType === 'vlm' ? 'selected' : ''}>🤖 Visual LLM</option>
                        <option value="classification" ${modelType === 'classification' ? 'selected' : ''}>🏷️ Image Classification</option>
                        <option value="zero_shot" ${modelType === 'zero_shot' ? 'selected' : ''}>🎯 Zero-Shot Classification</option>
                    </select>
                </div>

                <div class="model-select-wrapper" id="model-select-${node.id}">
                    <button class="model-select-btn ${!hasFilteredModels ? 'disabled empty' : ''}" data-node-id="${node.id}" style="--category-color: ${categoryColor}" ${!hasFilteredModels ? 'disabled' : ''}>
                        <span class="model-select-icon">${categoryIcon}</span>
                        <span class="model-select-text">${currentDisplayName}</span>
                        <span class="model-select-arrow">▼</span>
                    </button>
                    <div class="model-dropdown" id="model-dropdown-${node.id}">
                        ${categoriesHtml}
                    </div>
                </div>

                <button class="btn-ports-toggle" data-node-id="${node.id}">
                    ${showPorts ? '▼ Hide Routing Ports' : '▶ Show Routing Ports'}
                </button>
                <div class="curate-ports-list ${showPorts ? '' : 'hidden'}" id="curate-ports-${node.id}" onwheel="event.stopPropagation();">
                    ${portsHtml}
                    <div class="curate-add-port-wrapper">
                        <button class="curate-add-port-btn" data-node-id="${node.id}" title="Add port">+ Add Port</button>
                    </div>
                </div>

                <button class="btn-template-toggle" data-node-id="${node.id}">
                    ${node.data.showTemplate ? '▼ Hide Routing Template' : '▶ Show Routing Template'}
                </button>
                <div class="curate-template-section ${node.data.showTemplate ? '' : 'hidden'}" id="curate-template-${node.id}">
                    <div class="curate-port-references" id="curate-refs-${node.id}">
                        <div class="curate-refs-label">Port References (click to insert):</div>
                        <div class="curate-refs-list">
                            ${ports.map(port => `
                                <div class="curate-ref-item" data-ref-key="${port.refKey}" data-node-id="${node.id}" title="${port.label}: ${port.instruction || 'No instruction'}">
                                    <span class="curate-ref-key">{${port.refKey}}</span>
                                    <span class="curate-ref-label">${port.label}</span>
                                </div>
                            `).join('')}
                        </div>
                        <div class="curate-refs-help">
                            Use <code>{port_refKey}</code> for port label, <code>{port_refKey_instruction}</code> for criteria
                        </div>
                    </div>
                    <div class="curate-option-group">
                        <label class="curate-option-label" title="When enabled, images will be forwarded to downstream nodes connected to routing outputs. When disabled, only captions are passed.">
                            <input type="checkbox"
                                   id="curate-${node.id}-forward-images"
                                   class="curate-option-checkbox"
                                   ${node.data.forwardImages ? 'checked' : ''}>
                            Forward images to routed outputs
                        </label>
                    </div>
                    <div class="curate-template-wrapper">
                        <div class="curate-highlights" id="curate-${node.id}-highlights"></div>
                        <textarea id="curate-${node.id}-template"
                                  name="curate-${node.id}-template"
                                  class="curate-template"
                                  placeholder="Build your routing prompt template..."
                                  rows="8"
                                  onwheel="event.stopPropagation();">${node.data.template || ''}</textarea>
                    </div>
                </div>

                <button class="btn-advanced" data-node-id="${node.id}">
                    ${showAdvanced ? '▼ Hide Advanced' : '▶ Show Advanced'}
                </button>
                <div class="model-parameters ${showAdvanced ? '' : 'hidden'}" id="params-${node.id}">
                    <div style="text-align: center; color: var(--text-secondary); padding: 8px; font-size: 0.75rem;">
                        Loading parameters...
                    </div>
                </div>
            `;
            return html;
        }
        if (node.type === 'output') {
            const stats = node.data.stats || {};
            const processed = stats.processed || 0;
            const total = stats.total || 0;
            const success = stats.success || 0;
            const failed = stats.failed || 0;
            const progress = total > 0 ? Math.round((processed / total) * 100) : 0;
            const stage = stats.stage || '';
            const speed = stats.speed || '';
            const eta = stats.eta || '';
            const totalTime = stats.totalTime || '';
            const iterationsPerSecond = stats.iterationsPerSecond || '';
            const resultsReady = stats.resultsReady || 0;

            return `
                <div class="output-stats" id="output-stats-${node.id}">
                    ${total > 0 ? `
                        <div class="output-progress-container">
                            <div class="output-progress-bar">
                                <div class="output-progress-fill" style="width: ${progress}%"></div>
                            </div>
                            <div class="output-progress-text">${progress}%</div>
                        </div>

                        <div class="output-stat-row">
                            <span class="output-stat-label">Processed:</span>
                            <span class="output-stat-value">${processed}/${total}</span>
                        </div>

                        ${success > 0 || failed > 0 ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">Success:</span>
                                <span class="output-stat-value success">${success}</span>
                            </div>
                        ` : ''}

                        ${failed > 0 ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">Failed:</span>
                                <span class="output-stat-value error">${failed}</span>
                            </div>
                        ` : ''}

                        ${stage ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">Stage:</span>
                                <span class="output-stat-value">${stage}</span>
                            </div>
                        ` : ''}

                        ${speed ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">Speed:</span>
                                <span class="output-stat-value">${speed}</span>
                            </div>
                        ` : ''}

                        ${eta ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">ETA:</span>
                                <span class="output-stat-value">${eta}</span>
                            </div>
                        ` : ''}

                        ${totalTime ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">Total Time:</span>
                                <span class="output-stat-value">${totalTime}</span>
                            </div>
                        ` : ''}

                        ${iterationsPerSecond ? `
                            <div class="output-stat-row">
                                <span class="output-stat-label">It/s:</span>
                                <span class="output-stat-value">${iterationsPerSecond}</span>
                            </div>
                        ` : ''}
                    ` : ''}

                    ${resultsReady > 0 ? `
                        <div class="output-results-ready">
                            <span class="output-results-icon">✓</span>
                            <span>Results Ready (${resultsReady})</span>
                        </div>
                    ` : `
                        <div class="output-idle">
                            <span style="color: var(--text-secondary); font-size: 0.85rem;">Waiting for processing...</span>
                        </div>
                    `}
                </div>
            `;
        }
        return '';
    };

    // Global click handler to close model dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.model-select-wrapper')) {
            document.querySelectorAll('.model-select-wrapper.open').forEach(wrapper => {
                wrapper.classList.remove('open');
            });
        }
    });

    // Backward compatibility alias
    try { window.getNodeContent = NENodes.getNodeContent; } catch (e) {}

})();

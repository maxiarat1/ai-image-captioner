// Node core: creation, rendering, content, ports, and highlighting
(function() {
    const NENodes = {};

    // Create port element
    NENodes.createPort = function(node, portName, portIndex, isOutput) {
        const portWrapper = document.createElement('div');
        portWrapper.className = 'port-wrapper';

        const port = document.createElement('div');
        port.className = isOutput ? 'port port-out' : 'port port-in';
        port.dataset.node = node.id;
        port.dataset.port = portIndex;

        // For dynamic output ports (curate node), get the proper port type
        if (isOutput && typeof NENodes.getOutputPortType === 'function') {
            port.dataset.portType = NENodes.getOutputPortType(node, portIndex);
        } else {
            port.dataset.portType = portName;
        }

        const label = document.createElement('span');
        label.className = 'port-label';
        label.textContent = portName;

        if (isOutput) {
            port.onmousedown = (e) => NEConnections.startConnect(e, node.id, portIndex);
            portWrapper.appendChild(label);
            portWrapper.appendChild(port);
        } else {
            port.onmousedown = (e) => NEConnections.startConnectFromInput(e, node.id, portIndex);
            portWrapper.appendChild(port);
            portWrapper.appendChild(label);
        }

        return portWrapper;
    };

    // Add node
    NENodes.addNode = function(type) {
        const { wrapper } = NEUtils.getElements();
        const rect = wrapper.getBoundingClientRect();

        // Position node in center of visible viewport (in canvas space)
        const center = NEUtils.wrapperToCanvas(rect.width / 2, rect.height / 2);

        // Get default model from available models (first one if available, or let backend decide)
        const defaultModel = (AppState.availableModels && AppState.availableModels.length > 0)
            ? AppState.availableModels[0].name
            : AppState.selectedModel || '';

        const node = {
            id: NodeEditor.nextId++,
            type: type,
            x: center.x + (Math.random() - 0.5) * 200,
            y: center.y + (Math.random() - 0.5) * 200,
            label: '',
            data: type === 'prompt' ? { text: '' } :
                  type === 'aimodel' ? { model: defaultModel, parameters: {}, showAdvanced: false } :
                  type === 'conjunction' ? { connectedItems: [], template: '', showPreview: false } :
                  type === 'curate' ? {
                      modelType: 'vlm',  // 'vlm', 'classification', 'zero_shot'
                      model: defaultModel,
                      parameters: {},
                      ports: [
                          { id: 'port_1', label: 'Port 1', instruction: '' },
                          { id: 'port_2', label: 'Port 2', instruction: '' }
                      ],
                      showPorts: false,
                      showAdvanced: false
                  } : {}
        };

        NodeEditor.nodes.push(node);
        NENodes.renderNode(node);
    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Render node
    NENodes.renderNode = function(node) {
        const def = NODES[node.type];
    const el = document.createElement('div');
    // Add a per-type class to allow theming (e.g., gradient header colors per node type)
    el.className = 'node node-type-' + node.type;
        el.id = 'node-' + node.id;
        el.style.left = node.x + 'px';
        el.style.top = node.y + 'px';

        // Header
        const header = document.createElement('div');
        header.className = 'node-header';

        // Only show label input for nodes with outputs (can be referenced)
        const showLabelInput = def.outputs && def.outputs.length > 0;

        header.innerHTML = `
            <span class="node-type-label">${def.label}</span>
            ${showLabelInput ? `
                <input type="text"
                       class="node-label-input"
                       id="node-${node.id}-label"
                       placeholder="label..."
                       value="${node.label || ''}"
                       title="Custom label for references">
            ` : ''}
            <button class="node-del" data-id="${node.id}">×</button>
        `;
        header.onmousedown = (e) => {
            // Prevent drag when clicking on label input
            if (e.target.classList.contains('node-label-input')) {
                return;
            }
            NEDrag.startDrag(e, node);
        };
        el.appendChild(header);

        // Ports Section
        const portsSection = document.createElement('div');
        portsSection.className = 'node-ports-section';

        // Input ports
        const inputsContainer = document.createElement('div');
        inputsContainer.className = 'node-ports-in';
        def.inputs.forEach((portName, i) => {
            inputsContainer.appendChild(NENodes.createPort(node, portName, i, false));
        });
        portsSection.appendChild(inputsContainer);

        // Output ports
        const outputsContainer = document.createElement('div');
        outputsContainer.className = 'node-ports-out';
        def.outputs.forEach((portName, i) => {
            outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
        });
        portsSection.appendChild(outputsContainer);

        el.appendChild(portsSection);

        // For conjunction nodes, add references section before body
        if (node.type === 'conjunction') {
            const refsSection = document.createElement('div');
            refsSection.id = `node-${node.id}-refs-section`;
            refsSection.innerHTML = NENodes.getConjunctionReferencesHtml(node);
            el.appendChild(refsSection);
        }

        // Body (content section)
        const body = document.createElement('div');
        body.className = 'node-body';
        body.innerHTML = NENodes.getNodeContent(node);
        el.appendChild(body);

        document.getElementById('nodeCanvas').appendChild(el);

        // Delete handler
        el.querySelector('.node-del').onclick = (e) => {
            e.stopPropagation();
            if (typeof deleteNode === 'function') deleteNode(node.id);
        };

        // Input handlers
        const inputs = el.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.oninput = (e) => {
                const key = e.target.dataset.key;
                if (key) {
                    node.data[key] = e.target.value;

                    // If model selection changed, reload parameters
                    if (key === 'model' && node.type === 'aimodel') {
                        if (typeof loadModelParameters === 'function') {
                            loadModelParameters(node.id, e.target.value);
                        }
                    }

                    // If prompt text changed, update any connected conjunction nodes
                    if (key === 'text' && node.type === 'prompt') {
                        const connectedConjunctions = NodeEditor.connections
                            .filter(c => c.from === node.id)
                            .map(c => NodeEditor.nodes.find(n => n.id === c.to))
                            .filter(n => n && n.type === 'conjunction');

                        connectedConjunctions.forEach(conjNode => {
                            if (typeof updateConjunctionNode === 'function') updateConjunctionNode(conjNode.id);
                        });
                    }
                }
            };
        });

        // Model dropdown handlers (for AI model nodes)
        if (node.type === 'aimodel') {
            const modelBtn = el.querySelector('.model-select-btn');
            const modelDropdown = el.querySelector('.model-dropdown');
            const modelWrapper = el.querySelector('.model-select-wrapper');

            if (modelBtn && modelDropdown && modelWrapper) {
                // Toggle dropdown on button click
                modelBtn.onclick = (e) => {
                    e.stopPropagation();
                    const isOpen = modelWrapper.classList.contains('open');

                    // Close all other dropdowns
                    document.querySelectorAll('.model-select-wrapper.open').forEach(w => {
                        if (w !== modelWrapper) w.classList.remove('open');
                    });

                    modelWrapper.classList.toggle('open', !isOpen);
                };

                // Prevent dropdown from closing when clicking inside
                modelDropdown.onclick = (e) => {
                    e.stopPropagation();
                };

                // Prevent page scroll when scrolling inside dropdown
                modelDropdown.onwheel = (e) => {
                    e.stopPropagation();
                };

                // Handle category click to toggle submenu
                const categoryHeaders = el.querySelectorAll('.model-category-header');
                categoryHeaders.forEach(header => {
                    header.onclick = (e) => {
                        e.stopPropagation();
                        const category = header.parentElement;
                        category.classList.toggle('open');
                    };
                });

                // Handle model selection
                const modelItems = el.querySelectorAll('.model-item');
                modelItems.forEach(item => {
                    item.onclick = (e) => {
                        e.stopPropagation();
                        const modelName = item.dataset.model;
                        if (modelName && modelName !== node.data.model) {
                            node.data.model = modelName;

                            // Update button display
                            const category = typeof ModelCategories !== 'undefined'
                                ? ModelCategories.getCategoryForModel(modelName)
                                : null;
                            const displayName = typeof getModelDisplayName === 'function'
                                ? getModelDisplayName(modelName)
                                : modelName.toUpperCase();

                            const iconSpan = modelBtn.querySelector('.model-select-icon');
                            const textSpan = modelBtn.querySelector('.model-select-text');

                            if (iconSpan && category) iconSpan.textContent = category.icon;
                            if (textSpan) textSpan.textContent = displayName;
                            if (category) modelBtn.style.setProperty('--category-color', category.color);

                            // Update selected state
                            modelItems.forEach(mi => mi.classList.remove('selected'));
                            item.classList.add('selected');

                            // Close dropdown
                            modelWrapper.classList.remove('open');

                            // Reload model parameters
                            if (typeof loadModelParameters === 'function') {
                                loadModelParameters(node.id, modelName);
                            }
                        }
                    };
                });
            }
        }

        // Label input handler
        const labelInput = el.querySelector(`#node-${node.id}-label`);
        if (labelInput) {
            labelInput.oninput = (e) => {
                e.stopPropagation();
                node.label = e.target.value;

                // Update any connected conjunction nodes with new label
                const connectedConjunctions = NodeEditor.connections
                    .filter(c => c.from === node.id)
                    .map(c => NodeEditor.nodes.find(n => n.id === c.to))
                    .filter(n => n && n.type === 'conjunction');

                connectedConjunctions.forEach(conjNode => {
                    if (typeof updateConjunctionNode === 'function') updateConjunctionNode(conjNode.id);
                });
            };

            // Prevent propagation on click/focus to avoid triggering node drag
            labelInput.onclick = (e) => e.stopPropagation();
            labelInput.onmousedown = (e) => e.stopPropagation();
            labelInput.onfocus = (e) => e.stopPropagation();
        }

        // Curate node handlers
        if (node.type === 'curate') {
            // Attach handlers to port items
            NENodes.attachCurateHandlers(el, node);

            // Attach model dropdown and model type selector handlers
            NENodes.attachCurateModelDropdownHandlers(el, node);

            // Advanced toggle handler
            const advancedBtn = el.querySelector('.btn-advanced');
            if (advancedBtn) {
                advancedBtn.onclick = (e) => {
                    e.stopPropagation();
                    node.data.showAdvanced = !node.data.showAdvanced;

                    const paramsContainer = document.getElementById(`params-${node.id}`);
                    if (paramsContainer) {
                        if (node.data.showAdvanced) {
                            advancedBtn.textContent = '▼ Hide Advanced';
                            paramsContainer.classList.remove('hidden');
                            if (typeof loadModelParameters === 'function' && node.data.model) {
                                loadModelParameters(node.id, node.data.model);
                            }
                        } else {
                            advancedBtn.textContent = '▶ Show Advanced';
                            paramsContainer.classList.add('hidden');
                        }

                        setTimeout(() => {
                            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                        }, 300);
                    }
                };
            }

            // Load parameters if advanced is shown
            if (node.data.showAdvanced && node.data.model) {
                if (typeof loadModelParameters === 'function') {
                    loadModelParameters(node.id, node.data.model);
                }
            }
        }

        // Advanced toggle handler for AI model nodes
        if (node.type === 'aimodel') {
            const advancedBtn = el.querySelector('.btn-advanced');
            if (advancedBtn) {
                advancedBtn.onclick = (e) => {
                    e.stopPropagation();
                    node.data.showAdvanced = !node.data.showAdvanced;

                    // Update button text and parameters visibility
                    const paramsContainer = document.getElementById(`params-${node.id}`);
                    if (paramsContainer) {
                        if (node.data.showAdvanced) {
                            advancedBtn.textContent = '▼ Hide Advanced';
                            paramsContainer.classList.remove('hidden');
                            // Load parameters when showing advanced for the first time
                            if (typeof loadModelParameters === 'function' && node.data.model) {
                                loadModelParameters(node.id, node.data.model);
                            }
                        } else {
                            advancedBtn.textContent = '▶ Show Advanced';
                            paramsContainer.classList.add('hidden');
                        }

                        // Update connections after toggle animation completes
                        setTimeout(() => {
                            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                        }, 300);
                    }
                };
            }

            // Load parameters for AI model nodes (only if advanced is shown)
            if (node.data.showAdvanced && node.data.model) {
                if (typeof loadModelParameters === 'function') {
                    loadModelParameters(node.id, node.data.model);
                }
            }
        }

        // Conjunction template handler
        if (node.type === 'conjunction') {
            const templateTextarea = el.querySelector(`#node-${node.id}-template`);
            if (templateTextarea) {
                // Handle input and scroll events
                templateTextarea.oninput = () => {
                    node.data.template = templateTextarea.value;
                    NENodes.highlightPlaceholders(node.id);
                    NENodes.updateConjunctionPreview(node.id);
                };
                templateTextarea.onscroll = () => {
                    const highlightsDiv = document.getElementById(`node-${node.id}-highlights`);
                    if (highlightsDiv) {
                        highlightsDiv.scrollTop = templateTextarea.scrollTop;
                        highlightsDiv.scrollLeft = templateTextarea.scrollLeft;
                    }
                };

                // Initial highlight
                NENodes.highlightPlaceholders(node.id);
            }

            // Preview toggle handler
            const previewBtn = el.querySelector('.btn-preview');
            if (previewBtn) {
                previewBtn.onclick = (e) => {
                    e.stopPropagation();
                    node.data.showPreview = !node.data.showPreview;

                    // Update button text and preview visibility
                    const previewContainer = document.getElementById(`preview-${node.id}`);
                    if (previewContainer) {
                        if (node.data.showPreview) {
                            previewBtn.textContent = '▼ Hide Preview';
                            previewContainer.classList.remove('hidden');
                            // Update preview content
                            NENodes.updateConjunctionPreview(node.id);
                        } else {
                            previewBtn.textContent = '▶ Show Preview';
                            previewContainer.classList.add('hidden');
                        }

                        // Update connections after toggle animation completes
                        setTimeout(() => {
                            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                        }, 300);
                    }
                };
            }

            // Add click handlers to reference items to insert at cursor
            const refItems = el.querySelectorAll('.conjunction-ref-item');
            refItems.forEach(refItem => {
                refItem.onclick = (e) => {
                    e.stopPropagation();
                    const refKey = refItem.dataset.refKey;
                    const textarea = el.querySelector(`#node-${node.id}-template`);
                    if (!textarea || !refKey) return;

                    // Get cursor position
                    const start = textarea.selectionStart;
                    const end = textarea.selectionEnd;
                    const text = textarea.value;

                    // Insert reference at cursor position
                    const placeholder = `{${refKey}}`;
                    const newText = text.substring(0, start) + placeholder + text.substring(end);
                    textarea.value = newText;

                    // Update node data
                    node.data.template = newText;

                    // Set cursor after inserted text
                    const newCursorPos = start + placeholder.length;
                    textarea.setSelectionRange(newCursorPos, newCursorPos);

                    // Focus textarea and update highlights
                    textarea.focus();
                    NENodes.highlightPlaceholders(node.id);
                    NENodes.updateConjunctionPreview(node.id);
                };
            });
        }
    };

    // Get conjunction references HTML (separate from body)
    NENodes.getConjunctionReferencesHtml = function(node) {
        const items = node.data.connectedItems || [];

        if (items.length === 0) {
            return `
                <div class="conjunction-references-empty">
                    Connect prompts/captions to use as references
                </div>
            `;
        }

        const refsItems = items.map(item => {
            // Build tooltip with label info
            let tooltipText = item.sourceLabel;
            if (item.customLabel) {
                tooltipText += ` (${item.customLabel})`;
            }
            tooltipText += `: ${item.preview}`;

            return `
                <div class="conjunction-ref-item" data-ref-key="${item.refKey}" title="${tooltipText}">
                    <span class="conjunction-ref-key">{${item.refKey}}</span>
                </div>
            `;
        }).join('');

        return `
            <div class="conjunction-references">
                <div class="conjunction-references-title">Available References:</div>
                <div class="conjunction-ref-list">
                    ${refsItems}
                </div>
            </div>
        `;
    };

    // Get node content HTML
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
            const preview = NENodes.resolveConjunctionTemplate(node);

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
            const showPorts = node.data.showPorts || false;
            const availableModels = AppState.availableModels || [];
            const currentModel = node.data.model;
            const modelType = node.data.modelType || 'vlm';
            const ports = node.data.ports || [];

            // Filter models based on curate model type
            const filteredModels = NENodes.filterModelsForCurateType(availableModels, modelType);

            // Find current model's category and display info
            const currentCategory = typeof ModelCategories !== 'undefined'
                ? ModelCategories.getCategoryForModel(currentModel)
                : null;

            const currentDisplayName = currentModel
                ? (typeof getModelDisplayName === 'function'
                    ? getModelDisplayName(currentModel)
                    : currentModel.toUpperCase())
                : 'Select Model';

            const categoryIcon = currentCategory ? currentCategory.icon : '🔀';
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
                    <label for="curate-${node.id}-model-type" class="curate-label">Model Type:</label>
                    <select id="curate-${node.id}-model-type"
                            name="curate-${node.id}-model-type"
                            class="curate-model-type-select"
                            data-key="modelType">
                        <option value="vlm" ${modelType === 'vlm' ? 'selected' : ''}>🤖 Visual LLM (instruction-based)</option>
                        <option value="classification" ${modelType === 'classification' ? 'selected' : ''}>🏷️ Image Classification (pre-trained)</option>
                        <option value="zero_shot" ${modelType === 'zero_shot' ? 'selected' : ''}>🎯 Zero-Shot Classification (CLIP)</option>
                    </select>
                </div>

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

                                <button class="btn-ports-toggle" data-node-id="${node.id}">
                    ${showPorts ? '▼ Hide Routing Ports' : '▶ Show Routing Ports'}
                </button>
                <div class="curate-ports-list ${showPorts ? '' : 'hidden'}" id="curate-ports-${node.id}">
                    ${portsHtml}
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

    // Conjunction template helpers moved to core/conjunction.js

    // Update conjunction preview
    NENodes.updateConjunctionPreview = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'conjunction') return;

        const previewContent = document.querySelector(`#preview-${nodeId} .conjunction-preview-content`);
        if (!previewContent) return;

        const preview = NENodes.resolveConjunctionTemplate(node);
        previewContent.innerHTML = preview || '<em style="color: var(--text-secondary);">Empty template</em>';

        // Update recent outputs history (last 5)
        const historyEl = document.getElementById(`preview-${nodeId}-history`);
        if (historyEl) {
            const hist = Array.isArray(node.data.history) ? node.data.history.slice(-5) : [];
            if (hist.length === 0) {
                historyEl.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem;">No recent outputs</div>';
            } else {
                const escape = (s) => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
                // Show newest first
                const items = hist.slice().reverse().map(t => `<li class="conjunction-history-item" style="margin: 4px 0; line-height: 1.2;">${escape(t)}</li>`).join('');
                historyEl.innerHTML = `<ul class="conjunction-history-list" style="padding-left: 16px; margin: 4px 0 0 0;">${items}</ul>`;
            }
        }
    };

    // Update output node statistics
    NENodes.updateOutputStats = function(nodeId, stats) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'output') return;

        // Update node data
        node.data.stats = { ...node.data.stats, ...stats };

        // Update the display
        const statsContainer = document.getElementById(`output-stats-${nodeId}`);
        if (statsContainer) {
            const body = document.getElementById(`node-${nodeId}`).querySelector('.node-body');
            if (body) {
                body.innerHTML = NENodes.getNodeContent(node);
            }
        }
    };

    // Reset output node statistics
    NENodes.resetOutputStats = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'output') return;

        node.data.stats = {};

        const body = document.getElementById(`node-${nodeId}`).querySelector('.node-body');
        if (body) {
            body.innerHTML = NENodes.getNodeContent(node);
        }
    };

    // Filter models based on curate node type
    NENodes.filterModelsForCurateType = function(availableModels, curateType) {
        if (!availableModels || availableModels.length === 0) return [];

        // Model categories for each curate type
        const modelMappings = {
            'vlm': ['multimodal', 'general'],  // Visual LLMs - BLIP2, LLaVA, Janus, etc.
            'classification': ['anime'],  // Classification models - WD14, ViT, etc.
            'zero_shot': ['multimodal']  // Zero-shot - CLIP-based models
        };

        const allowedCategories = modelMappings[curateType] || ['multimodal'];

        // Filter models by category
        return availableModels.filter(model => {
            const category = model.category || 'general';
            return allowedCategories.includes(category);
        });
    };

    // Dynamic Port Management API

    // Add output port to a node (for nodes with allowDynamicOutputs)
    NENodes.addOutputPort = function(nodeId, portConfig) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return;

        const nodeDef = NODES[node.type];
        if (!nodeDef || !nodeDef.allowDynamicOutputs) {
            console.warn('Node type does not support dynamic outputs');
            return;
        }

        // Initialize ports array if it doesn't exist
        if (!node.data.ports) {
            node.data.ports = [];
        }

        // Generate port ID if not provided
        const portId = portConfig.id || `port_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const newPort = {
            id: portId,
            label: portConfig.label || `Port ${node.data.ports.length + 1}`,
            instruction: portConfig.instruction || '',
            isDefault: portConfig.isDefault || false
        };

        // If this is set as default, unset other defaults
        if (newPort.isDefault) {
            node.data.ports.forEach(p => p.isDefault = false);
        }

        node.data.ports.push(newPort);

        // Re-render ports section
        NENodes.updateNodePorts(nodeId);

        // Update connections
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

        return portId;
    };

    // Remove output port from a node
    NENodes.removeOutputPort = function(nodeId, portId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || !node.data.ports) return;

        const portIndex = node.data.ports.findIndex(p => p.id === portId);
        if (portIndex === -1) return;

        // Remove any connections to this port
        if (typeof NEConnections !== 'undefined') {
            const connectionsToRemove = NodeEditor.connections.filter(
                c => c.from === nodeId && c.fromPort === portIndex
            );
            connectionsToRemove.forEach(c => NEConnections.removeConnection(c.id));
        }

        // Remove port
        node.data.ports.splice(portIndex, 1);

        // Update port indices in remaining connections
        if (typeof NEConnections !== 'undefined') {
            NodeEditor.connections.forEach(c => {
                if (c.from === nodeId && c.fromPort > portIndex) {
                    c.fromPort--;
                }
            });
        }

        // Re-render ports section
        NENodes.updateNodePorts(nodeId);

        // Update connections
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Update node's ports section (re-render ports without re-rendering entire node)
    NENodes.updateNodePorts = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return;

        const nodeEl = document.getElementById(`node-${nodeId}`);
        if (!nodeEl) return;

        const portsSection = nodeEl.querySelector('.node-ports-section');
        if (!portsSection) return;

        const nodeDef = NODES[node.type];

        // Clear and rebuild ports section
        portsSection.innerHTML = '';

        // Input ports
        const inputsContainer = document.createElement('div');
        inputsContainer.className = 'node-ports-in';
        nodeDef.inputs.forEach((portName, i) => {
            inputsContainer.appendChild(NENodes.createPort(node, portName, i, false));
        });
        portsSection.appendChild(inputsContainer);

        // Output ports - handle dynamic outputs
        const outputsContainer = document.createElement('div');
        outputsContainer.className = 'node-ports-out';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            // Use dynamic ports from node data
            node.data.ports.forEach((portConfig, i) => {
                const portName = portConfig.label || `Port ${i + 1}`;
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        } else {
            // Use static ports from definition
            nodeDef.outputs.forEach((portName, i) => {
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        }

        portsSection.appendChild(outputsContainer);
    };

    // Get output port name for a node (handles both static and dynamic ports)
    NENodes.getOutputPortName = function(node, portIndex) {
        const nodeDef = NODES[node.type];
        if (!nodeDef) return '';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            const port = node.data.ports[portIndex];
            return port ? port.label : '';
        }

        return nodeDef.outputs[portIndex] || '';
    };

    // Get output port type for connections (returns 'route' for curate nodes)
    NENodes.getOutputPortType = function(node, portIndex) {
        if (node.type === 'curate') {
            return 'route';
        }

        const nodeDef = NODES[node.type];
        if (!nodeDef) return '';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            return 'route';  // All dynamic outputs are route type
        }

        return nodeDef.outputs[portIndex] || '';
    };

    // Attach model dropdown handlers for curate node
    NENodes.attachCurateModelDropdownHandlers = function(nodeEl, node) {
        if (!node || node.type !== 'curate') return;

        const modelBtn = nodeEl.querySelector('.model-select-btn');
        const modelDropdown = nodeEl.querySelector('.model-dropdown');
        const modelWrapper = nodeEl.querySelector('.model-select-wrapper');

        if (!modelBtn || !modelDropdown || !modelWrapper) return;

        modelBtn.onclick = (e) => {
            e.stopPropagation();
            const isOpen = modelWrapper.classList.contains('open');
            document.querySelectorAll('.model-select-wrapper.open').forEach(w => {
                if (w !== modelWrapper) w.classList.remove('open');
            });
            modelWrapper.classList.toggle('open', !isOpen);
        };

        modelDropdown.onclick = (e) => e.stopPropagation();
        modelDropdown.onwheel = (e) => e.stopPropagation();

        const categoryHeaders = nodeEl.querySelectorAll('.model-category-header');
        categoryHeaders.forEach(header => {
            header.onclick = (e) => {
                e.stopPropagation();
                header.parentElement.classList.toggle('open');
            };
        });

        const modelItems = nodeEl.querySelectorAll('.model-item');
        modelItems.forEach(item => {
            item.onclick = (e) => {
                e.stopPropagation();
                const modelName = item.dataset.model;
                if (modelName && modelName !== node.data.model) {
                    node.data.model = modelName;

                    const category = typeof ModelCategories !== 'undefined'
                        ? ModelCategories.getCategoryForModel(modelName)
                        : null;
                    const displayName = typeof getModelDisplayName === 'function'
                        ? getModelDisplayName(modelName)
                        : modelName.toUpperCase();

                    const iconSpan = modelBtn.querySelector('.model-select-icon');
                    const textSpan = modelBtn.querySelector('.model-select-text');

                    if (iconSpan && category) iconSpan.textContent = category.icon;
                    if (textSpan) textSpan.textContent = displayName;
                    if (category) modelBtn.style.setProperty('--category-color', category.color);

                    modelItems.forEach(mi => mi.classList.remove('selected'));
                    item.classList.add('selected');
                    modelWrapper.classList.remove('open');

                    if (typeof loadModelParameters === 'function') {
                        loadModelParameters(node.id, modelName);
                    }
                }
            };
        });

        // Model type selector handler (needs to be re-attached after re-render)
        const modelTypeSelect = nodeEl.querySelector('.curate-model-type-select');
        if (modelTypeSelect) {
            modelTypeSelect.onchange = (e) => {
                node.data.modelType = e.target.value;

                // Re-render entire node body to update model dropdown and port fields
                const body = nodeEl.querySelector('.node-body');
                if (body) body.innerHTML = NENodes.getNodeContent(node);

                // Re-attach all handlers including model dropdown
                NENodes.attachCurateHandlers(nodeEl, node);
                NENodes.attachCurateModelDropdownHandlers(nodeEl, node);
            };
        }
    };

    // Attach event handlers to curate node port items
    NENodes.attachCurateHandlers = function(nodeEl, node) {
        if (!node || node.type !== 'curate') return;

        const ports = node.data.ports || [];

        // Port label input handlers
        const labelInputs = nodeEl.querySelectorAll('.curate-port-label-input');
        labelInputs.forEach(input => {
            input.onclick = (e) => e.stopPropagation();
            input.onmousedown = (e) => e.stopPropagation();
            input.onfocus = (e) => e.stopPropagation();
            input.oninput = (e) => {
                const portId = e.target.dataset.portId;
                const port = ports.find(p => p.id === portId);
                if (port) {
                    port.label = e.target.value;
                    NENodes.updateNodePorts(node.id);
                    NENodes.checkAndAddNewPort(nodeEl, node, portId);
                }
            };
        });

        // Port instruction textarea handlers
        const instructionTextareas = nodeEl.querySelectorAll('.curate-port-instruction');
        instructionTextareas.forEach(textarea => {
            textarea.onclick = (e) => e.stopPropagation();
            textarea.onmousedown = (e) => e.stopPropagation();
            textarea.onfocus = (e) => e.stopPropagation();
            textarea.oninput = (e) => {
                const portId = e.target.dataset.portId;
                const port = ports.find(p => p.id === portId);
                if (port) {
                    port.instruction = e.target.value;
                    NENodes.checkAndAddNewPort(nodeEl, node, portId);
                }
            };
        });

        // Port delete button handlers
        const deleteButtons = nodeEl.querySelectorAll('.curate-port-delete');
        deleteButtons.forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                const portId = btn.dataset.portId;

                // Don't allow deleting if only 2 ports remain
                if (ports.length <= 2) {
                    alert('At least two ports are required.');
                    return;
                }

                NENodes.removeOutputPort(node.id, portId);
                // Re-render
                const body = nodeEl.querySelector('.node-body');
                if (body) body.innerHTML = NENodes.getNodeContent(node);
                NENodes.attachCurateHandlers(nodeEl, node);
                NENodes.attachCurateModelDropdownHandlers(nodeEl, node);
            };
        });

        // Ports toggle handler
        const portsBtn = nodeEl.querySelector('.btn-ports-toggle');
        if (portsBtn) {
            portsBtn.onclick = (e) => {
                e.stopPropagation();
                node.data.showPorts = !node.data.showPorts;

                const portsContainer = document.getElementById(`curate-ports-${node.id}`);
                if (portsContainer) {
                    if (node.data.showPorts) {
                        portsBtn.textContent = '▼ Hide Routing Ports';
                        portsContainer.classList.remove('hidden');
                    } else {
                        portsBtn.textContent = '▶ Show Routing Ports';
                        portsContainer.classList.add('hidden');
                    }

                    setTimeout(() => {
                        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                    }, 300);
                }
            };
        }
    };

    // Helper function to check if we should auto-add a new port
    NENodes.checkAndAddNewPort = function(nodeEl, node, currentPortId) {
        const ports = node.data.ports || [];
        const lastPort = ports[ports.length - 1];

        // If user is typing in the last port and it has content, add a new empty port
        if (lastPort && currentPortId === lastPort.id) {
            const hasContent = lastPort.label || lastPort.instruction;
            if (hasContent) {
                const portNumber = ports.length + 1;
                const newPortId = `port_${portNumber}`;

                // Check if we already added this port
                if (!ports.find(p => p.id === newPortId)) {
                    NENodes.addOutputPort(node.id, {
                        id: newPortId,
                        label: `Port ${portNumber}`,
                        instruction: ''
                    });

                    // Re-render
                    const body = nodeEl.querySelector('.node-body');
                    if (body) body.innerHTML = NENodes.getNodeContent(node);
                    NENodes.attachCurateHandlers(nodeEl, node);
                    NENodes.attachCurateModelDropdownHandlers(nodeEl, node);
                }
            }
        }
    };

    // Global click handler to close model dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        // Check if click is outside any model dropdown
        if (!e.target.closest('.model-select-wrapper')) {
            document.querySelectorAll('.model-select-wrapper.open').forEach(wrapper => {
                wrapper.classList.remove('open');
            });
        }
    });

    // Expose
    window.NENodes = NENodes;
    // Backward compatibility aliases
    window.createPort = NENodes.createPort;
    window.addNode = NENodes.addNode;
    window.renderNode = NENodes.renderNode;
    window.getConjunctionReferencesHtml = NENodes.getConjunctionReferencesHtml;
    window.getNodeContent = NENodes.getNodeContent;
    window.highlightPlaceholders = NENodes.highlightPlaceholders;
    window.resolveConjunctionTemplate = NENodes.resolveConjunctionTemplate;
    window.updateConjunctionPreview = NENodes.updateConjunctionPreview;
    window.updateOutputStats = NENodes.updateOutputStats;
    window.resetOutputStats = NENodes.resetOutputStats;
})();

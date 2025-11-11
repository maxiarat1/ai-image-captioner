// Node rendering and creation split out from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

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
                          { id: 'port_1', label: 'Port 1', instruction: '', refKey: 'port_1' },
                          { id: 'port_2', label: 'Port 2', instruction: '', refKey: 'port_2' }
                      ],
                      template: `Analyze this image and determine which category best describes it.

Available categories:
{port_1}: {port_1_instruction}
{port_2}: {port_2_instruction}

Respond with ONLY the exact category name. Do not add explanations.`,
                      showPorts: true,
                      showTemplate: false,
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

        // Output ports - handle dynamic outputs for curate nodes
        const outputsContainer = document.createElement('div');
        outputsContainer.className = 'node-ports-out';
        
        if (def.allowDynamicOutputs && node.data.ports) {
            // Use dynamic ports from node data
            node.data.ports.forEach((portConfig, i) => {
                const portName = portConfig.label || `Port ${i + 1}`;
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        } else {
            // Use static ports from definition
            def.outputs.forEach((portName, i) => {
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        }
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

    try {
        window.addNode = NENodes.addNode;
        window.renderNode = NENodes.renderNode;
    } catch (e) {}

})();

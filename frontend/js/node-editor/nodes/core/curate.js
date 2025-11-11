// Curate node UI handlers split out from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

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

    try {
        window.attachCurateHandlers = NENodes.attachCurateHandlers;
        window.attachCurateModelDropdownHandlers = NENodes.attachCurateModelDropdownHandlers;
    } catch (e) {}

})();

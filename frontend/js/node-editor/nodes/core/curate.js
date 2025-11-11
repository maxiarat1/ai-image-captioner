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

                // Animate out the specific port element, then remove from DOM and update data
                const portEl = nodeEl.querySelector(`.curate-port-item[data-port-id="${portId}"]`);
                if (portEl) {
                    // add animate-out class and remove after animation
                    portEl.classList.add('animate-out');
                    portEl.addEventListener('animationend', () => {
                        // Update underlying data and connection ports
                        NENodes.removeOutputPort(node.id, portId);
                        // Remove the DOM element
                        try { portEl.remove(); } catch (err) { /* ignore */ }
                        // Update connections visuals
                        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                    }, { once: true });
                } else {
                    // Fallback: if element not found, just remove the port from data
                    NENodes.removeOutputPort(node.id, portId);
                }
            };
        });

        // Add port button handler (explicit add instead of auto-adding on typing)
        const addPortBtn = nodeEl.querySelector('.curate-add-port-btn');
        if (addPortBtn) {
            addPortBtn.onclick = (e) => {
                e.stopPropagation();
                const portsArr = node.data.ports || [];
                const portNumber = portsArr.length + 1;

                // Add to data first (returns new port id)
                const newPortId = NENodes.addOutputPort(node.id, {
                    label: `Port ${portNumber}`,
                    instruction: ''
                });

                // Update connection ports visuals
                if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();

                // Create the DOM node for the new port and append it before the add button wrapper
                const portsContainer = nodeEl.querySelector(`#curate-ports-${node.id}`);
                const addWrapper = portsContainer ? portsContainer.querySelector('.curate-add-port-wrapper') : null;
                if (portsContainer) {
                    const portHtml = `
                    <div class="curate-port-item animate-in" data-port-id="${newPortId}">
                        <div class="curate-port-header">
                            <input type="text"
                                   id="curate-${node.id}-port-${newPortId}-label"
                                   name="curate-${node.id}-port-${newPortId}-label"
                                   class="curate-port-label-input"
                                   data-port-id="${newPortId}"
                                   value="Port ${portNumber}"
                                   placeholder="Port ${portNumber}">
                            <button class="curate-port-delete" data-port-id="${newPortId}" title="Delete port">×</button>
                        </div>
                        <textarea id="curate-${node.id}-port-${newPortId}-instruction"
                                  name="curate-${node.id}-port-${newPortId}-instruction"
                                  class="curate-port-instruction"
                                  data-port-id="${newPortId}"
                                  placeholder="Describe routing criteria..."
                                  rows="2"></textarea>
                    </div>
                    `;

                    const temp = document.createElement('div');
                    temp.innerHTML = portHtml.trim();
                    const newPortEl = temp.firstChild;

                    if (addWrapper) portsContainer.insertBefore(newPortEl, addWrapper);
                    else portsContainer.appendChild(newPortEl);

                    // Attach handlers only to the new element
                    const input = newPortEl.querySelector('.curate-port-label-input');
                    const textarea = newPortEl.querySelector('.curate-port-instruction');
                    const deleteBtn = newPortEl.querySelector('.curate-port-delete');

                    if (input) {
                        input.onclick = (ev) => ev.stopPropagation();
                        input.onmousedown = (ev) => ev.stopPropagation();
                        input.onfocus = (ev) => ev.stopPropagation();
                        input.oninput = (ev) => {
                            const portId = ev.target.dataset.portId;
                            const port = (node.data.ports || []).find(p => p.id === portId);
                            if (port) {
                                port.label = ev.target.value;
                                NENodes.updateNodePorts(node.id);
                            }
                        };
                    }

                    if (textarea) {
                        textarea.onclick = (ev) => ev.stopPropagation();
                        textarea.onmousedown = (ev) => ev.stopPropagation();
                        textarea.onfocus = (ev) => ev.stopPropagation();
                        textarea.oninput = (ev) => {
                            const portId = ev.target.dataset.portId;
                            const port = (node.data.ports || []).find(p => p.id === portId);
                            if (port) {
                                port.instruction = ev.target.value;
                            }
                        };
                    }

                    if (deleteBtn) {
                        deleteBtn.onclick = (ev) => {
                            ev.stopPropagation();
                            const pid = deleteBtn.dataset.portId;
                            const portElToRemove = nodeEl.querySelector(`.curate-port-item[data-port-id="${pid}"]`);
                            if (portElToRemove) {
                                portElToRemove.classList.add('animate-out');
                                portElToRemove.addEventListener('animationend', () => {
                                    NENodes.removeOutputPort(node.id, pid);
                                    try { portElToRemove.remove(); } catch (err) {}
                                    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                                }, { once: true });
                            } else {
                                NENodes.removeOutputPort(node.id, pid);
                            }
                        };
                    }
                }
            };
        }

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

    // Auto-add-on-typing behavior removed. Use the explicit Add Port button instead.

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

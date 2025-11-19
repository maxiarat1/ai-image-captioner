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
                    // Update refKey based on new label (without re-rendering)
                    NENodes.updatePortRefKey(node.id, portId);
                }
            };
            // Update connection port labels only when done editing
            input.onblur = () => {
                NENodes.updateNodePorts(node.id);
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

                // Helper function to complete the deletion
                const completeDelete = () => {
                    // Update underlying data and connection ports
                    NENodes.removeOutputPort(node.id, portId);
                    // Remove the DOM element
                    const portElToRemove = nodeEl.querySelector(`.curate-port-item[data-port-id="${portId}"]`);
                    if (portElToRemove) {
                        try { portElToRemove.remove(); } catch (err) { console.error("Failed to remove port element:", err); }
                    }
                    // Update connections visuals
                    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();

                    // Update reference chips after deletion
                    const refsContainer = nodeEl.querySelector(`#curate-refs-${node.id} .curate-refs-list`);
                    if (refsContainer) {
                        const remainingPorts = node.data.ports || [];
                        const refsHtml = remainingPorts.map((p, index) => {
                            const safeLabel = p.label || `Port ${index + 1}`;
                            const combinedText = p.instruction
                                ? `${safeLabel}: ${p.instruction}`
                                : safeLabel;
                            return `
                                <div class="curate-ref-item"
                                     data-ref-key="${p.refKey}"
                                     data-node-id="${node.id}"
                                     title="${combinedText}">
                                    <div class="curate-ref-top">
                                        <span class="curate-ref-key">{${p.refKey}}</span>
                                        <span class="curate-ref-label">${safeLabel}</span>
                                    </div>
                                </div>
                            `;
                        }).join('');
                        refsContainer.innerHTML = refsHtml;

                        // Re-attach click handlers for the reference chips
                        const refItems = refsContainer.querySelectorAll('.curate-ref-item');
                        refItems.forEach(refItem => {
                            refItem.onclick = (evt) => {
                                evt.stopPropagation();
                                const refKey = refItem.dataset.refKey;
                                const textarea = nodeEl.querySelector(`#curate-${node.id}-template`);

                                if (textarea && refKey) {
                                    const start = textarea.selectionStart;
                                    const end = textarea.selectionEnd;
                                    const text = textarea.value;

                                    const placeholder = `{${refKey}}`;
                                    const newText = text.substring(0, start) + placeholder + text.substring(end);
                                    textarea.value = newText;
                                    node.data.template = newText;

                                    const newCursorPos = start + placeholder.length;
                                    textarea.setSelectionRange(newCursorPos, newCursorPos);

                                    textarea.focus();
                                    NENodes.highlightCuratePlaceholders(node.id);
                                }
                            };
                        });
                    }

                    // Update template highlighting
                    NENodes.highlightCuratePlaceholders(node.id);
                };

                // Animate out the specific port element, then remove from DOM and update data
                const portEl = nodeEl.querySelector(`.curate-port-item[data-port-id="${portId}"]`);
                if (portEl) {
                    // add animate-out class and remove after animation
                    portEl.classList.add('animate-out');

                    // Use both animationend and a fallback timeout
                    let deleted = false;
                    const doDelete = () => {
                        if (!deleted) {
                            deleted = true;
                            completeDelete();
                        }
                    };

                    portEl.addEventListener('animationend', doDelete, { once: true });
                    // Fallback timeout in case animation doesn't fire
                    setTimeout(doDelete, 300);
                } else {
                    // Fallback: if element not found, just complete the deletion
                    completeDelete();
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

                // Guard: if port wasn't added properly, exit
                if (!newPortId) {
                    console.error('Failed to add output port');
                    return;
                }

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

                    // Scroll the new port into view
                    newPortEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

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
                                // Update refKey based on new label (without re-rendering)
                                NENodes.updatePortRefKey(node.id, portId);
                            }
                        };
                        // Update connection port labels only when done editing
                        input.onblur = () => {
                            NENodes.updateNodePorts(node.id);
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

                            // Don't allow deleting if only 2 ports remain
                            const currentPorts = node.data.ports || [];
                            if (currentPorts.length <= 2) {
                                alert('At least two ports are required.');
                                return;
                            }

                            // Helper function to complete the deletion
                            const completeDelete = () => {
                                NENodes.removeOutputPort(node.id, pid);
                                const portElToRemove = nodeEl.querySelector(`.curate-port-item[data-port-id="${pid}"]`);
                                if (portElToRemove) {
                                    try { portElToRemove.remove(); } catch (err) { console.error("Failed to remove port element:", err); }
                                }
                                if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();

                                // Update reference chips after deletion
                                const refsContainer = nodeEl.querySelector(`#curate-refs-${node.id} .curate-refs-list`);
                                if (refsContainer) {
                                    const remainingPorts = node.data.ports || [];
                                    const refsHtml = remainingPorts.map((p, index) => {
                                        const safeLabel = p.label || `Port ${index + 1}`;
                                        const combinedText = p.instruction
                                            ? `${safeLabel}: ${p.instruction}`
                                            : safeLabel;
                                        return `
                                            <div class="curate-ref-item"
                                                 data-ref-key="${p.refKey}"
                                                 data-node-id="${node.id}"
                                                 title="${combinedText}">
                                                <div class="curate-ref-top">
                                                    <span class="curate-ref-key">{${p.refKey}}</span>
                                                    <span class="curate-ref-label">${safeLabel}</span>
                                                </div>
                                            </div>
                                        `;
                                    }).join('');
                                    refsContainer.innerHTML = refsHtml;

                                    // Re-attach click handlers for the reference chips
                                    const refItems = refsContainer.querySelectorAll('.curate-ref-item');
                                    refItems.forEach(refItem => {
                                        refItem.onclick = (evt) => {
                                            evt.stopPropagation();
                                            const refKey = refItem.dataset.refKey;
                                            const textarea = nodeEl.querySelector(`#curate-${node.id}-template`);

                                            if (textarea && refKey) {
                                                const start = textarea.selectionStart;
                                                const end = textarea.selectionEnd;
                                                const text = textarea.value;

                                                const placeholder = `{${refKey}}`;
                                                const newText = text.substring(0, start) + placeholder + text.substring(end);
                                                textarea.value = newText;
                                                node.data.template = newText;

                                                const newCursorPos = start + placeholder.length;
                                                textarea.setSelectionRange(newCursorPos, newCursorPos);

                                                textarea.focus();
                                                NENodes.highlightCuratePlaceholders(node.id);
                                            }
                                        };
                                    });
                                }

                                // Update template highlighting
                                NENodes.highlightCuratePlaceholders(node.id);
                            };

                            const portElToRemove = nodeEl.querySelector(`.curate-port-item[data-port-id="${pid}"]`);
                            if (portElToRemove) {
                                portElToRemove.classList.add('animate-out');

                                // Use both animationend and a fallback timeout
                                let deleted = false;
                                const doDelete = () => {
                                    if (!deleted) {
                                        deleted = true;
                                        completeDelete();
                                    }
                                };

                                portElToRemove.addEventListener('animationend', doDelete, { once: true });
                                setTimeout(doDelete, 300);
                            } else {
                                completeDelete();
                            }
                        };
                    }

                    // Update the reference chips to include the new port
                    const newPort = (node.data.ports || []).find(p => p.id === newPortId);
                    if (newPort) {
                        const refsContainer = nodeEl.querySelector(`#curate-refs-${node.id} .curate-refs-list`);
                        if (refsContainer) {
                            const ports = node.data.ports || [];
                            const refsHtml = ports.map((p, index) => {
                                const safeLabel = p.label || `Port ${index + 1}`;
                                const combinedText = p.instruction
                                    ? `${safeLabel}: ${p.instruction}`
                                    : safeLabel;
                                return `
                                    <div class="curate-ref-item"
                                         data-ref-key="${p.refKey}"
                                         data-node-id="${node.id}"
                                         title="${combinedText}">
                                        <div class="curate-ref-top">
                                            <span class="curate-ref-key">{${p.refKey}}</span>
                                            <span class="curate-ref-label">${safeLabel}</span>
                                        </div>
                                    </div>
                                `;
                            }).join('');
                            refsContainer.innerHTML = refsHtml;

                            // Re-attach click handlers for the reference chips
                            const refItems = refsContainer.querySelectorAll('.curate-ref-item');
                            refItems.forEach(refItem => {
                                refItem.onclick = (evt) => {
                                    evt.stopPropagation();
                                    const refKey = refItem.dataset.refKey;
                                    const textarea = nodeEl.querySelector(`#curate-${node.id}-template`);

                                    if (textarea && refKey) {
                                        const start = textarea.selectionStart;
                                        const end = textarea.selectionEnd;
                                        const text = textarea.value;

                                        // Insert {refKey} at cursor
                                        const placeholder = `{${refKey}}`;
                                        const newText = text.substring(0, start) + placeholder + text.substring(end);
                                        textarea.value = newText;
                                        node.data.template = newText;

                                        // Update cursor position
                                        const newCursorPos = start + placeholder.length;
                                        textarea.setSelectionRange(newCursorPos, newCursorPos);

                                        textarea.focus();
                                        NENodes.highlightCuratePlaceholders(node.id);
                                    }
                                };
                            });
                        }

                        // Update template highlighting
                        NENodes.highlightCuratePlaceholders(node.id);
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

        // Template toggle handler
        const templateBtn = nodeEl.querySelector('.btn-template-toggle');
        if (templateBtn) {
            templateBtn.onclick = (e) => {
                e.stopPropagation();
                node.data.showTemplate = !node.data.showTemplate;

                const templateSection = document.getElementById(`curate-template-${node.id}`);
                if (templateSection) {
                    if (node.data.showTemplate) {
                        templateBtn.textContent = '▼ Hide Routing Template';
                        templateSection.classList.remove('hidden');
                        // Highlight placeholders on show
                        NENodes.highlightCuratePlaceholders(node.id);
                    } else {
                        templateBtn.textContent = '▶ Show Routing Template';
                        templateSection.classList.add('hidden');
                    }
                }
            };
        }

        // Template textarea handlers
        const templateTextarea = nodeEl.querySelector(`#curate-${node.id}-template`);
        if (templateTextarea) {
            templateTextarea.onclick = (e) => e.stopPropagation();
            templateTextarea.onmousedown = (e) => e.stopPropagation();
            templateTextarea.onfocus = (e) => e.stopPropagation();
            templateTextarea.oninput = (e) => {
                node.data.template = e.target.value;
                NENodes.highlightCuratePlaceholders(node.id);
            };
            templateTextarea.onscroll = (e) => {
                const highlightsDiv = document.getElementById(`curate-${node.id}-highlights`);
                if (highlightsDiv) {
                    highlightsDiv.scrollTop = e.target.scrollTop;
                    highlightsDiv.scrollLeft = e.target.scrollLeft;
                }
            };
        }

        // Port reference chip click-to-insert handlers
        const refItems = nodeEl.querySelectorAll('.curate-ref-item');
        refItems.forEach(refItem => {
            refItem.onclick = (e) => {
                e.stopPropagation();
                const refKey = refItem.dataset.refKey;
                const textarea = nodeEl.querySelector(`#curate-${node.id}-template`);

                if (textarea && refKey) {
                    const start = textarea.selectionStart;
                    const end = textarea.selectionEnd;
                    const text = textarea.value;

                    // Insert {refKey} at cursor
                    const placeholder = `{${refKey}}`;
                    const newText = text.substring(0, start) + placeholder + text.substring(end);
                    textarea.value = newText;
                    node.data.template = newText;

                    // Update cursor position
                    const newCursorPos = start + placeholder.length;
                    textarea.setSelectionRange(newCursorPos, newCursorPos);

                    textarea.focus();
                    NENodes.highlightCuratePlaceholders(node.id);
                }
            };
        });
    };

    // Auto-add-on-typing behavior removed. Use the explicit Add Port button instead.

    // Highlight placeholders in curate template
    NENodes.highlightCuratePlaceholders = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'curate') return;

        const textarea = document.getElementById(`curate-${nodeId}-template`);
        const highlightsDiv = document.getElementById(`curate-${nodeId}-highlights`);

        if (!textarea || !highlightsDiv) return;

        const text = textarea.value;
        const ports = node.data.ports || [];

        // Build list of valid placeholders
        const validKeys = ['caption'];  // Input caption placeholder
        ports.forEach(port => {
            validKeys.push(port.refKey);
            validKeys.push(`${port.refKey}_label`);
            validKeys.push(`${port.refKey}_instruction`);
        });

        // Regex to find all placeholders
        const regex = /\{([^}]+)\}/g;
        // Escape HTML first
        let escapedText = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Then highlight placeholders
        let highlightedText = escapedText.replace(regex, (match, key) => {
            const isValid = validKeys.includes(key);
            const className = isValid ? 'placeholder-valid' : 'placeholder-invalid';
            return `<mark class="${className}">${match}</mark>`;
        });

        // Preserve formatting
        highlightedText = highlightedText
            .replace(/\n/g, '<br>')
            .replace(/ /g, '&nbsp;');
        highlightsDiv.innerHTML = highlightedText;

        // Sync scroll position
        highlightsDiv.scrollTop = textarea.scrollTop;
        highlightsDiv.scrollLeft = textarea.scrollLeft;
    };

    // Update port refKey when label changes
    NENodes.updatePortRefKey = function(nodeId, portId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'curate') return;

        const port = (node.data.ports || []).find(p => p.id === portId);
        if (!port) return;

        // Generate new refKey from label
        const newRefKey = typeof NEUtils !== 'undefined' && NEUtils.sanitizeLabel
            ? NEUtils.sanitizeLabel(port.label)
            : port.label.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '').substring(0, 30);

        // Check for duplicates
        const usedRefKeys = (node.data.ports || [])
            .filter(p => p.id !== portId)
            .map(p => p.refKey);

        let finalRefKey = newRefKey;
        let counter = 2;
        while (usedRefKeys.includes(finalRefKey)) {
            finalRefKey = `${newRefKey}_${counter}`;
            counter++;
        }

        port.refKey = finalRefKey;

        // Update only the port reference chips display (not the entire body)
        const nodeEl = document.getElementById(`node-${nodeId}`);
        if (nodeEl) {
            const refsContainer = nodeEl.querySelector(`#curate-refs-${nodeId} .curate-refs-list`);
            if (refsContainer) {
                const ports = node.data.ports || [];
                const refsHtml = ports.map((p, index) => {
                    const safeLabel = p.label || `Port ${index + 1}`;
                    const combinedText = p.instruction
                        ? `${safeLabel}: ${p.instruction}`
                        : safeLabel;
                    return `
                        <div class="curate-ref-item"
                             data-ref-key="${p.refKey}"
                             data-node-id="${nodeId}"
                             title="${combinedText}">
                            <div class="curate-ref-top">
                                <span class="curate-ref-key">{${p.refKey}}</span>
                                <span class="curate-ref-label">${safeLabel}</span>
                            </div>
                        </div>
                    `;
                }).join('');
                refsContainer.innerHTML = refsHtml;

                // Re-attach click handlers for the reference chips
                const refItems = refsContainer.querySelectorAll('.curate-ref-item');
                refItems.forEach(refItem => {
                    refItem.onclick = (e) => {
                        e.stopPropagation();
                        const refKey = refItem.dataset.refKey;
                        const textarea = nodeEl.querySelector(`#curate-${nodeId}-template`);

                        if (textarea && refKey) {
                            const start = textarea.selectionStart;
                            const end = textarea.selectionEnd;
                            const text = textarea.value;

                            // Insert {refKey} at cursor
                            const placeholder = `{${refKey}}`;
                            const newText = text.substring(0, start) + placeholder + text.substring(end);
                            textarea.value = newText;
                            node.data.template = newText;

                            // Update cursor position
                            const newCursorPos = start + placeholder.length;
                            textarea.setSelectionRange(newCursorPos, newCursorPos);

                            textarea.focus();
                            NENodes.highlightCuratePlaceholders(nodeId);
                        }
                    };
                });
            }
        }

        // Update template highlighting
        NENodes.highlightCuratePlaceholders(nodeId);
    };

    // Filter models based on curate node type
    NENodes.filterModelsForCurateType = function(availableModels, curateType) {
        if (!availableModels || availableModels.length === 0) return [];

        // For VLM mode: only show models with vlm_capable = true AND curate_suitable !== false
        if (curateType === 'vlm') {
            return availableModels.filter(model =>
                model.vlm_capable === true && model.curate_suitable !== false
            );
        }

        // For classification and zero_shot: disabled for now (return empty array)
        if (curateType === 'classification' || curateType === 'zero_shot') {
            return [];
        }

        // Default fallback: VLM models only (also filter out non-curate-suitable)
        return availableModels.filter(model =>
            model.vlm_capable === true && model.curate_suitable !== false
        );
    };

    try {
        window.attachCurateHandlers = NENodes.attachCurateHandlers;
        window.attachCurateModelDropdownHandlers = NENodes.attachCurateModelDropdownHandlers;
    } catch (e) {}

})();

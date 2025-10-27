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
        port.dataset.portType = portName;

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

        const node = {
            id: NodeEditor.nextId++,
            type: type,
            x: center.x + (Math.random() - 0.5) * 200,
            y: center.y + (Math.random() - 0.5) * 200,
            label: '',
            data: type === 'prompt' ? { text: '' } :
                  type === 'aimodel' ? { model: 'blip', parameters: {}, showAdvanced: false } :
                  type === 'conjunction' ? { connectedItems: [], template: '', showPreview: false } : {}
        };

        NodeEditor.nodes.push(node);
        NENodes.renderNode(node);
    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Render node
    NENodes.renderNode = function(node) {
        const def = NODES[node.type];
        const el = document.createElement('div');
        el.className = 'node';
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
                            if (typeof loadModelParameters === 'function') {
                                loadModelParameters(node.id, node.data.model || 'blip');
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
            if (node.data.showAdvanced) {
                if (typeof loadModelParameters === 'function') {
                    loadModelParameters(node.id, node.data.model || 'blip');
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

            // Build model options dynamically from AppState.availableModels
            const availableModels = AppState.availableModels || [];
            const modelOptions = availableModels.length > 0
                ? availableModels.map(model => {
                    const displayName = typeof getModelDisplayName === 'function'
                        ? getModelDisplayName(model.name)
                        : model.name.toUpperCase();
                    const selected = node.data.model === model.name ? 'selected' : '';
                    const tooltip = model.description || '';
                    return `<option value="${model.name}" ${selected} title="${tooltip}">${displayName}</option>`;
                }).join('\n                    ')
                : `
                    <option value="blip" ${node.data.model === 'blip' ? 'selected' : ''}>BLIP</option>
                    <option value="r4b" ${node.data.model === 'r4b' ? 'selected' : ''}>R-4B</option>
                    <option value="wdvit" ${node.data.model === 'wdvit' ? 'selected' : ''}>WD-ViT v3</option>
                    <option value="wdeva02" ${node.data.model === 'wdeva02' ? 'selected' : ''}>WD-EVA02 v3</option>
                `;

            let html = `
                <select id="node-${node.id}-model" name="model-${node.id}" data-key="model" class="model-select">
                    ${modelOptions}
                </select>
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

    // Highlight placeholders in conjunction template
    NENodes.highlightPlaceholders = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'conjunction') return;

        const textarea = document.getElementById(`node-${nodeId}-template`);
        const highlightsDiv = document.getElementById(`node-${nodeId}-highlights`);
        if (!textarea || !highlightsDiv) return;

        const text = textarea.value;
        const validKeys = (node.data.connectedItems || []).map(item => item.refKey);

        // Find all placeholders and mark them as valid or invalid
        const regex = /\{([^}]+)\}/g;
        let highlightedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        highlightedText = highlightedText.replace(regex, (match, key) => {
            const isValid = validKeys.includes(key);
            const className = isValid ? 'placeholder-valid' : 'placeholder-invalid';
            return `<mark class="${className}">${match}</mark>`;
        });

        // Add line breaks for proper alignment
        highlightedText = highlightedText.replace(/\n/g, '<br>');

        highlightsDiv.innerHTML = highlightedText;

        // Sync scroll
        highlightsDiv.scrollTop = textarea.scrollTop;
        highlightsDiv.scrollLeft = textarea.scrollLeft;
    };

    // Resolve conjunction template with actual values
    NENodes.resolveConjunctionTemplate = function(node) {
        if (!node || node.type !== 'conjunction') return '';

        const template = node.data.template || '';
        if (!template) return '';

        const items = node.data.connectedItems || [];
        const refMap = {};

        // Build reference map
        items.forEach(item => {
            refMap[item.refKey] = item.content;
        });

        // Replace placeholders with actual content
        const resolved = template.replace(/\{([^}]+)\}/g, (match, key) => {
            return refMap[key] !== undefined ? refMap[key] : match;
        });

        // Escape HTML for display
        return resolved.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
    };

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

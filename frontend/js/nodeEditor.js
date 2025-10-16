// ============================================================================
// Node Editor - Simple & Barebone
// ============================================================================

// State
const NodeEditor = {
    nodes: [],
    connections: [],
    nextId: 1,
    dragging: null,
    connecting: null,
    panning: null,
    transform: {
        x: 0,
        y: 0,
        scale: 1
    }
};

// Node types
const NODES = {
    input: { label: 'Input', inputs: [], outputs: ['images'] },
    prompt: { label: 'Prompt', inputs: [], outputs: ['text'] },
    conjunction: { label: 'Conjunction', inputs: ['text', 'captions'], outputs: ['text'] },
    aimodel: { label: 'AI Model', inputs: ['images', 'prompt'], outputs: ['captions'] },
    output: { label: 'Output', inputs: ['data'], outputs: [] }
};

// ============================================================================
// Helper Functions
// ============================================================================

// Get frequently used DOM elements
const getElements = () => ({
    canvas: document.getElementById('nodeCanvas'),
    wrapper: document.querySelector('.node-canvas-wrapper'),
    svg: document.getElementById('connectionsSVG')
});

// Convert screen coordinates to canvas space
function screenToCanvas(screenX, screenY) {
    const { canvas } = getElements();
    const rect = canvas.getBoundingClientRect();
    return {
        x: (screenX - rect.left) / NodeEditor.transform.scale,
        y: (screenY - rect.top) / NodeEditor.transform.scale
    };
}

// Convert wrapper-relative screen position to canvas local coordinates
function wrapperToCanvas(screenX, screenY) {
    return {
        x: (screenX - NodeEditor.transform.x) / NodeEditor.transform.scale,
        y: (screenY - NodeEditor.transform.y) / NodeEditor.transform.scale
    };
}

// Create port element
function createPort(node, portName, portIndex, isOutput) {
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
        port.onmousedown = (e) => startConnect(e, node.id, portIndex);
        portWrapper.appendChild(label);
        portWrapper.appendChild(port);
    } else {
        portWrapper.appendChild(port);
        portWrapper.appendChild(label);
    }

    return portWrapper;
}

// Sanitize label to create valid reference key
function sanitizeLabel(label) {
    if (!label || typeof label !== 'string') return '';

    // Replace spaces with underscores, remove special chars except underscores
    return label
        .trim()
        .replace(/\s+/g, '_')
        .replace(/[^a-zA-Z0-9_]/g, '')
        .substring(0, 30); // Limit length
}

// ============================================================================
// Initialization & Setup
// ============================================================================

// Initialize
function initNodeEditor() {
    setupToolbar();
    initCanvasPanning();
    createConnectionGradient();
    createMinimap();

    document.getElementById('executeGraphBtn').onclick = executeGraph;
    document.getElementById('clearGraphBtn').onclick = clearGraph;
    document.getElementById('fullscreenBtn').onclick = openFullscreen;

    // Setup fullscreen modal handlers
    document.getElementById('closeNodeFullscreen').onclick = closeFullscreen;
    document.getElementById('nodeFullscreenBackdrop').onclick = (e) => {
        if (e.target.id === 'nodeFullscreenBackdrop') {
            closeFullscreen();
        }
    };

    // Close on Esc key
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('nodeFullscreenModal');
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeFullscreen();
        }
    });
}

// Open fullscreen modal
function openFullscreen() {
    const modal = document.getElementById('nodeFullscreenModal');
    const container = document.getElementById('fullscreenCanvasContainer');
    const toolbar = document.getElementById('nodeToolbar');
    const wrapper = document.querySelector('.node-canvas-wrapper');

    // Move toolbar and wrapper into modal
    container.appendChild(toolbar);
    container.appendChild(wrapper);

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Re-center canvas for fullscreen dimensions
    setTimeout(() => {
        const rect = container.getBoundingClientRect();
        NodeEditor.transform.x = rect.width / 2 - 2500;
        NodeEditor.transform.y = rect.height / 2 - 2500;
        applyTransform();
        updateConnections();
        updateMinimap();
    }, 50);

    // Add zoom support to fullscreen container
    const fullscreenZoomHandler = (e) => {
        e.preventDefault();
        handleZoom(e);
    };
    container.addEventListener('wheel', fullscreenZoomHandler);
    // Store handler for cleanup
    container._zoomHandler = fullscreenZoomHandler;
}

// Close fullscreen modal
function closeFullscreen() {
    const modal = document.getElementById('nodeFullscreenModal');
    const container = document.getElementById('fullscreenCanvasContainer');

    // Remove zoom event listener from fullscreen container
    if (container._zoomHandler) {
        container.removeEventListener('wheel', container._zoomHandler);
        delete container._zoomHandler;
    }

    // Add closing class to trigger animation
    modal.classList.add('closing');

    // Wait for animation to complete using animationend event
    const handleAnimationEnd = (e) => {
        // Only handle the modal's own animation, not child animations
        if (e.target === modal) {
            modal.removeEventListener('animationend', handleAnimationEnd);

            // Animation complete - now safe to move elements and cleanup
            const editorContainer = document.querySelector('.node-editor-container');
            const toolbar = document.getElementById('nodeToolbar');
            const wrapper = document.querySelector('.node-canvas-wrapper');

            modal.classList.remove('active', 'closing');
            // Move toolbar and wrapper back to editor container
            editorContainer.appendChild(toolbar);
            editorContainer.appendChild(wrapper);
            document.body.style.overflow = '';

            // Re-center canvas for normal wrapper dimensions
            setTimeout(() => {
                const rect = wrapper.getBoundingClientRect();
                NodeEditor.transform.x = rect.width / 2 - 2500;
                NodeEditor.transform.y = rect.height / 2 - 2500;
                applyTransform();
                updateConnections();
                updateMinimap();
            }, 50);
        }
    };

    modal.addEventListener('animationend', handleAnimationEnd);
}

// Setup toolbar
function setupToolbar() {
    const toolbar = document.getElementById('nodeToolbar');
    toolbar.innerHTML = '';

    for (const [type, def] of Object.entries(NODES)) {
        const btn = document.createElement('button');
        btn.textContent = def.label;
        btn.style.borderLeft = `3px solid ${def.color}`;
        btn.onclick = () => addNode(type);
        toolbar.appendChild(btn);
    }
}

// Apply transform to canvas
function applyTransform() {
    const canvas = document.getElementById('nodeCanvas');
    if (!canvas) return;

    const { x, y, scale } = NodeEditor.transform;
    canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
}

// Initialize canvas panning and zoom
function initCanvasPanning() {
    const wrapper = document.querySelector('.node-canvas-wrapper');
    const canvas = document.getElementById('nodeCanvas');
    if (!wrapper || !canvas) return;

    // Center the canvas initially (canvas is 5000x5000, center at 2500, 2500)
    const rect = wrapper.getBoundingClientRect();
    NodeEditor.transform.x = rect.width / 2 - 2500;
    NodeEditor.transform.y = rect.height / 2 - 2500;
    applyTransform();

    // Mouse down - start panning on canvas background only
    canvas.addEventListener('mousedown', (e) => {
        // Only pan on left click on canvas or SVG (not on nodes)
        const isNode = e.target.closest('.node');
        if (e.button === 0 && !isNode) {
            e.preventDefault();
            NodeEditor.panning = {
                startX: e.clientX - NodeEditor.transform.x,
                startY: e.clientY - NodeEditor.transform.y
            };
            wrapper.classList.add('panning');
        }
    });

    // Mouse move - do panning
    document.addEventListener('mousemove', (e) => {
        if (NodeEditor.panning) {
            NodeEditor.transform.x = e.clientX - NodeEditor.panning.startX;
            NodeEditor.transform.y = e.clientY - NodeEditor.panning.startY;
            applyTransform();
            updateMinimap();
        }
    });

    // Mouse up - stop panning
    document.addEventListener('mouseup', () => {
        if (NodeEditor.panning) {
            wrapper.classList.remove('panning');
            NodeEditor.panning = null;
        }
    });

    // Mouse wheel - zoom
    wrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        handleZoom(e);
    });
}

// Handle zoom
function handleZoom(e) {
    const wrapper = document.querySelector('.node-canvas-wrapper');
    const rect = wrapper.getBoundingClientRect();

    // Mouse position relative to wrapper
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Current mouse position in canvas space (before zoom)
    const canvasX = (mouseX - NodeEditor.transform.x) / NodeEditor.transform.scale;
    const canvasY = (mouseY - NodeEditor.transform.y) / NodeEditor.transform.scale;

    // Calculate zoom delta
    const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
    let newScale = NodeEditor.transform.scale * zoomDelta;

    // Limit zoom range
    newScale = Math.max(0.1, Math.min(3, newScale));

    // Adjust transform to zoom toward cursor
    NodeEditor.transform.scale = newScale;
    NodeEditor.transform.x = mouseX - canvasX * newScale;
    NodeEditor.transform.y = mouseY - canvasY * newScale;

    applyTransform();
    updateConnections();
    updateMinimap();
}

// ============================================================================
// Minimap
// ============================================================================

// Create minimap
function createMinimap() {
    const wrapper = document.querySelector('.node-canvas-wrapper');

    const minimap = document.createElement('div');
    minimap.className = 'node-minimap';
    minimap.id = 'nodeMinimap';

    const minimapCanvas = document.createElement('div');
    minimapCanvas.className = 'minimap-canvas';
    minimapCanvas.id = 'minimapCanvas';

    const viewport = document.createElement('div');
    viewport.className = 'minimap-viewport';
    viewport.id = 'minimapViewport';

    minimapCanvas.appendChild(viewport);
    minimap.appendChild(minimapCanvas);
    wrapper.appendChild(minimap);

    updateMinimap();
}

// Update minimap
function updateMinimap() {
    const minimapCanvas = document.getElementById('minimapCanvas');
    const viewport = document.getElementById('minimapViewport');

    if (!minimapCanvas || !viewport) return;

    // Canvas is 5000x5000, minimap is 200x200, so scale is 0.04
    const scale = 200 / 5000;

    // Clear existing nodes
    const existingNodes = minimapCanvas.querySelectorAll('.minimap-node');
    existingNodes.forEach(n => n.remove());

    // Draw nodes
    NodeEditor.nodes.forEach(node => {
        const dot = document.createElement('div');
        dot.className = 'minimap-node';
        dot.style.left = (node.x * scale) + 'px';
        dot.style.top = (node.y * scale) + 'px';
        dot.style.width = '6px';
        dot.style.height = '6px';
        minimapCanvas.appendChild(dot);
    });

    // Update viewport indicator
    const canvas = document.getElementById('nodeCanvas');
    const wrapper = canvas ? canvas.parentElement : null;
    if (!wrapper) return;

    const rect = wrapper.getBoundingClientRect();
    const viewportWidth = rect.width / NodeEditor.transform.scale;
    const viewportHeight = rect.height / NodeEditor.transform.scale;
    const viewportX = -NodeEditor.transform.x / NodeEditor.transform.scale;
    const viewportY = -NodeEditor.transform.y / NodeEditor.transform.scale;

    viewport.style.left = (viewportX * scale) + 'px';
    viewport.style.top = (viewportY * scale) + 'px';
    viewport.style.width = (viewportWidth * scale) + 'px';
    viewport.style.height = (viewportHeight * scale) + 'px';
}

// ============================================================================
// Node Management
// ============================================================================

// Add node
function addNode(type) {
    const { wrapper } = getElements();
    const rect = wrapper.getBoundingClientRect();

    // Position node in center of visible viewport (in canvas space)
    const center = wrapperToCanvas(rect.width / 2, rect.height / 2);

    const node = {
        id: NodeEditor.nextId++,
        type: type,
        x: center.x + (Math.random() - 0.5) * 200,
        y: center.y + (Math.random() - 0.5) * 200,
        label: '',
        data: type === 'prompt' ? { text: '' } :
              type === 'aimodel' ? { model: 'blip', parameters: {}, showAdvanced: false } :
              type === 'conjunction' ? { connectedItems: [], template: '' } : {}
    };

    NodeEditor.nodes.push(node);
    renderNode(node);
    updateMinimap();
}

// Render node
function renderNode(node) {
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
        startDrag(e, node);
    };
    el.appendChild(header);

    // Ports Section
    const portsSection = document.createElement('div');
    portsSection.className = 'node-ports-section';

    // Input ports
    const inputsContainer = document.createElement('div');
    inputsContainer.className = 'node-ports-in';
    def.inputs.forEach((portName, i) => {
        inputsContainer.appendChild(createPort(node, portName, i, false));
    });
    portsSection.appendChild(inputsContainer);

    // Output ports
    const outputsContainer = document.createElement('div');
    outputsContainer.className = 'node-ports-out';
    def.outputs.forEach((portName, i) => {
        outputsContainer.appendChild(createPort(node, portName, i, true));
    });
    portsSection.appendChild(outputsContainer);

    el.appendChild(portsSection);

    // For conjunction nodes, add references section before body
    if (node.type === 'conjunction') {
        const refsSection = document.createElement('div');
        refsSection.id = `node-${node.id}-refs-section`;
        refsSection.innerHTML = getConjunctionReferencesHtml(node);
        el.appendChild(refsSection);
    }

    // Body (content section)
    const body = document.createElement('div');
    body.className = 'node-body';
    body.innerHTML = getNodeContent(node);
    el.appendChild(body);

    document.getElementById('nodeCanvas').appendChild(el);

    // Delete handler
    el.querySelector('.node-del').onclick = (e) => {
        e.stopPropagation();
        deleteNode(node.id);
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
                    loadModelParameters(node.id, e.target.value);
                }

                // If prompt text changed, update any connected conjunction nodes
                if (key === 'text' && node.type === 'prompt') {
                    const connectedConjunctions = NodeEditor.connections
                        .filter(c => c.from === node.id)
                        .map(c => NodeEditor.nodes.find(n => n.id === c.to))
                        .filter(n => n && n.type === 'conjunction');

                    connectedConjunctions.forEach(conjNode => {
                        updateConjunctionNode(conjNode.id);
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
                updateConjunctionNode(conjNode.id);
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
                        loadModelParameters(node.id, node.data.model || 'blip');
                    } else {
                        advancedBtn.textContent = '▶ Show Advanced';
                        paramsContainer.classList.add('hidden');
                    }

                    // Update connections after toggle animation completes
                    setTimeout(() => {
                        updateConnections();
                    }, 300);
                }
            };
        }

        // Load parameters for AI model nodes (only if advanced is shown)
        if (node.data.showAdvanced) {
            loadModelParameters(node.id, node.data.model || 'blip');
        }
    }

    // Conjunction template handler
    if (node.type === 'conjunction') {
        const templateTextarea = el.querySelector(`#node-${node.id}-template`);
        if (templateTextarea) {
            // Handle input and scroll events
            templateTextarea.oninput = () => {
                node.data.template = templateTextarea.value;
                highlightPlaceholders(node.id);
            };
            templateTextarea.onscroll = () => {
                const highlightsDiv = document.getElementById(`node-${node.id}-highlights`);
                if (highlightsDiv) {
                    highlightsDiv.scrollTop = templateTextarea.scrollTop;
                    highlightsDiv.scrollLeft = templateTextarea.scrollLeft;
                }
            };

            // Initial highlight
            highlightPlaceholders(node.id);
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
                highlightPlaceholders(node.id);
            };
        });
    }
}

// Get conjunction references HTML (separate from body)
function getConjunctionReferencesHtml(node) {
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
}

// Get node content HTML
function getNodeContent(node) {
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
        `;
    }
    if (node.type === 'aimodel') {
        const showAdvanced = node.data.showAdvanced || false;
        let html = `
            <select id="node-${node.id}-model" name="model-${node.id}" data-key="model" class="model-select">
                <option value="blip" ${node.data.model === 'blip' ? 'selected' : ''}>BLIP</option>
                <option value="r4b" ${node.data.model === 'r4b' ? 'selected' : ''}>R-4B</option>
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
    return '';
}

// Update all input nodes with current queue count
function updateInputNodes() {
    NodeEditor.nodes.forEach(node => {
        if (node.type === 'input') {
            const el = document.getElementById('node-' + node.id);
            if (el) {
                const body = el.querySelector('.node-body');
                if (body) {
                    body.innerHTML = getNodeContent(node);
                }
            }
        }
    });
}

// Update conjunction node with connected items
function updateConjunctionNode(nodeId) {
    const node = NodeEditor.nodes.find(n => n.id === nodeId);
    if (!node || node.type !== 'conjunction') return;

    // Find all incoming connections
    const incomingConnections = NodeEditor.connections.filter(c => c.to === nodeId);

    // Gather connected items with reference keys
    const connectedItems = [];
    const usedRefKeys = new Set(); // Track all used reference keys to avoid duplicates
    const labelCounts = {}; // Track count of each node type for auto-numbering

    incomingConnections.forEach(conn => {
        const sourceNode = NodeEditor.nodes.find(n => n.id === conn.from);
        if (!sourceNode) return;

        const sourceDef = NODES[sourceNode.type];
        const portType = sourceDef.outputs[conn.fromPort];
        const baseLabel = sourceDef.label.replace(/\s+/g, '_');

        // Generate reference key with priority: custom label > content preview > auto-numbered
        let refKey;
        let shouldSetLabel = false;

        if (sourceNode.label && sourceNode.label.trim()) {
            // Priority 1: Use existing label (custom or previously auto-generated)
            refKey = sanitizeLabel(sourceNode.label);
        } else if (sourceNode.type === 'prompt' && sourceNode.data.text && sourceNode.data.text.trim()) {
            // Priority 2: Use content preview for prompts
            const preview = sourceNode.data.text.trim().substring(0, 20);
            refKey = sanitizeLabel(preview);
        } else {
            // Priority 3: Auto-generate with numbering
            labelCounts[baseLabel] = (labelCounts[baseLabel] || 0) + 1;
            refKey = labelCounts[baseLabel] === 1 ? baseLabel : `${baseLabel}${labelCounts[baseLabel]}`;
            shouldSetLabel = true;
        }

        // Handle duplicate keys by appending numbers
        const originalRefKey = refKey;
        let counter = 2;
        while (usedRefKeys.has(refKey)) {
            refKey = `${originalRefKey}_${counter}`;
            counter++;
            shouldSetLabel = true; // Update label if we had to deduplicate
        }

        // Mark this refKey as used
        usedRefKeys.add(refKey);

        // Update node label to match the final refKey
        if (shouldSetLabel) {
            sourceNode.label = refKey;
            // Update the label input field
            const labelInput = document.getElementById(`node-${sourceNode.id}-label`);
            if (labelInput) {
                labelInput.value = refKey;
            }
        }

        let content = '';
        let preview = '';

        if (sourceNode.type === 'prompt') {
            content = sourceNode.data.text || '';
            preview = content.substring(0, 60) + (content.length > 60 ? '...' : '');
        } else if (sourceNode.type === 'aimodel') {
            content = '[Generated Captions]';
            preview = 'Captions from AI Model (generated at runtime)';
        } else {
            content = `[${sourceDef.label} Output]`;
            preview = `Output from ${sourceDef.label}`;
        }

        connectedItems.push({
            sourceId: sourceNode.id,
            sourceLabel: sourceDef.label,
            customLabel: sourceNode.label || '',
            refKey: refKey,
            portType: portType,
            content: content,
            preview: preview
        });
    });

    // Update node data
    node.data.connectedItems = connectedItems;

    // Update node display - only update the references section
    const el = document.getElementById('node-' + nodeId);
    if (el) {
        const refsSection = document.getElementById(`node-${nodeId}-refs-section`);
        if (refsSection) {
            refsSection.innerHTML = getConjunctionReferencesHtml(node);

            // Re-attach click handlers to reference items
            const refItems = refsSection.querySelectorAll('.conjunction-ref-item');
            refItems.forEach(refItem => {
                refItem.onclick = (e) => {
                    e.stopPropagation();
                    const refKey = refItem.dataset.refKey;
                    const textarea = el.querySelector(`#node-${nodeId}-template`);
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
                    highlightPlaceholders(nodeId);
                };
            });

            // Update placeholder highlighting
            highlightPlaceholders(nodeId);
        }
    }
}

// Highlight placeholders in conjunction template
function highlightPlaceholders(nodeId) {
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
}

// Fetch model parameters from API
async function fetchModelParameters(modelName) {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/model/info?model=${modelName}`);
        if (!response.ok) throw new Error('Failed to fetch model parameters');
        const data = await response.json();
        return data.parameters || [];
    } catch (error) {
        console.error('Error fetching model parameters:', error);
        return [];
    }
}

// Build parameter input HTML based on parameter definition
function buildParameterInput(param, currentValue, nodeId) {
    const value = currentValue !== undefined ? currentValue : '';
    const inputId = `node-${nodeId}-param-${param.param_key}`;
    const inputName = `${param.param_key}-${nodeId}`;

    if (param.type === 'number') {
        return `
            <div class="param-group">
                <label class="param-label" for="${inputId}" title="${param.description}">${param.name}</label>
                <input type="number"
                       id="${inputId}"
                       name="${inputName}"
                       class="param-input"
                       data-param-key="${param.param_key}"
                       min="${param.min}"
                       max="${param.max}"
                       step="${param.step}"
                       value="${value}"
                       placeholder="${param.min}-${param.max}">
            </div>
        `;
    } else if (param.type === 'select') {
        const options = param.options.map(opt =>
            `<option value="${opt.value}" ${value === opt.value ? 'selected' : ''}>${opt.label}</option>`
        ).join('');
        return `
            <div class="param-group">
                <label class="param-label" for="${inputId}" title="${param.description}">${param.name}</label>
                <select id="${inputId}"
                        name="${inputName}"
                        class="param-input"
                        data-param-key="${param.param_key}">
                    <option value="">Default</option>
                    ${options}
                </select>
            </div>
        `;
    } else if (param.type === 'checkbox') {
        return `
            <div class="param-group">
                <label class="param-label param-checkbox-label" for="${inputId}" title="${param.description}">
                    <input type="checkbox"
                           id="${inputId}"
                           name="${inputName}"
                           class="param-checkbox"
                           data-param-key="${param.param_key}"
                           ${value ? 'checked' : ''}>
                    ${param.name}
                </label>
            </div>
        `;
    }
    return '';
}

// Load and display model parameters
async function loadModelParameters(nodeId, modelName) {
    const node = NodeEditor.nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    const paramsContainer = document.getElementById(`params-${nodeId}`);
    if (!paramsContainer) return;
    
    // Fetch parameters
    const parameters = await fetchModelParameters(modelName);
    
    if (parameters.length === 0) {
        paramsContainer.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 4px; font-size: 0.75rem;">No parameters available</div>';
        return;
    }
    
    // Build parameters UI
    const currentParams = node.data.parameters || {};
    const paramsHtml = parameters.map(param => buildParameterInput(param, currentParams[param.param_key], nodeId)).join('');
    paramsContainer.innerHTML = paramsHtml;
    
    // Add event listeners for parameter inputs
    paramsContainer.querySelectorAll('.param-input, .param-checkbox').forEach(input => {
        input.addEventListener('input', (e) => {
            const paramKey = e.target.dataset.paramKey;
            if (!paramKey) return;
            
            if (e.target.type === 'checkbox') {
                node.data.parameters[paramKey] = e.target.checked;
            } else if (e.target.type === 'number') {
                const value = parseFloat(e.target.value);
                if (!isNaN(value)) {
                    node.data.parameters[paramKey] = value;
                } else {
                    delete node.data.parameters[paramKey];
                }
            } else {
                const value = e.target.value;
                if (value) {
                    node.data.parameters[paramKey] = value;
                } else {
                    delete node.data.parameters[paramKey];
                }
            }
        });
    });
}

// ============================================================================
// Node Dragging
// ============================================================================

function startDrag(e, node) {
    if (e.target.classList.contains('node-del')) return;

    const canvasPos = screenToCanvas(e.clientX, e.clientY);
    const el = document.getElementById('node-' + node.id);
    el.classList.add('dragging');

    NodeEditor.dragging = {
        node: node,
        offsetX: canvasPos.x - node.x,
        offsetY: canvasPos.y - node.y
    };

    document.onmousemove = drag;
    document.onmouseup = stopDrag;
}

function drag(e) {
    if (!NodeEditor.dragging) return;

    const canvasPos = screenToCanvas(e.clientX, e.clientY);
    const node = NodeEditor.dragging.node;
    node.x = canvasPos.x - NodeEditor.dragging.offsetX;
    node.y = canvasPos.y - NodeEditor.dragging.offsetY;

    const el = document.getElementById('node-' + node.id);
    el.style.left = node.x + 'px';
    el.style.top = node.y + 'px';

    updateConnections();
    updateMinimap();
}

function stopDrag() {
    if (NodeEditor.dragging) {
        const el = document.getElementById('node-' + NodeEditor.dragging.node.id);
        el.classList.remove('dragging');
    }

    NodeEditor.dragging = null;
    document.onmousemove = null;
    document.onmouseup = null;
}

// ============================================================================
// Connection Creation
// ============================================================================

function startConnect(e, nodeId, portIndex) {
    e.stopPropagation();
    e.preventDefault();

    const { canvas, svg } = getElements();
    canvas.classList.add('connecting');

    // Create temporary connection line
    const tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    tempLine.id = 'temp-connection';
    tempLine.setAttribute('stroke', 'url(#connection-gradient)');
    tempLine.setAttribute('stroke-width', '3');
    tempLine.setAttribute('stroke-linecap', 'round');
    tempLine.setAttribute('stroke-dasharray', '5,5');
    tempLine.style.opacity = '0.6';
    svg.appendChild(tempLine);

    // Get starting port position in canvas local coordinates
    const portEl = document.querySelector(`#node-${nodeId} .port-out[data-port="${portIndex}"]`);
    const container = canvas.parentElement;
    const containerRect = container.getBoundingClientRect();
    const portRect = portEl.getBoundingClientRect();

    const startPos = wrapperToCanvas(
        portRect.left - containerRect.left + portRect.width / 2,
        portRect.top - containerRect.top + portRect.height / 2
    );

    NodeEditor.connecting = {
        from: nodeId,
        port: portIndex,
        startX: startPos.x,
        startY: startPos.y
    };

    document.onmousemove = updateTempConnection;
    document.onmouseup = endConnect;
}

function updateTempConnection(e) {
    if (!NodeEditor.connecting) return;

    const { canvas } = getElements();
    const tempLine = document.getElementById('temp-connection');

    if (!tempLine) return;

    // Convert mouse position to canvas local coordinates
    const container = canvas.parentElement;
    const containerRect = container.getBoundingClientRect();
    const mousePos = wrapperToCanvas(
        e.clientX - containerRect.left,
        e.clientY - containerRect.top
    );

    tempLine.setAttribute('x1', NodeEditor.connecting.startX);
    tempLine.setAttribute('y1', NodeEditor.connecting.startY);
    tempLine.setAttribute('x2', mousePos.x);
    tempLine.setAttribute('y2', mousePos.y);
}

function endConnect(e) {
    if (!NodeEditor.connecting) return;

    const { canvas } = getElements();
    canvas.classList.remove('connecting');

    // Remove temporary line
    const tempLine = document.getElementById('temp-connection');
    if (tempLine) tempLine.remove();

    const target = e.target;
    if (target.classList.contains('port-in')) {
        const toNode = parseInt(target.dataset.node);
        const toPort = parseInt(target.dataset.port);

        if (toNode !== NodeEditor.connecting.from) {
            addConnection(NodeEditor.connecting.from, NodeEditor.connecting.port, toNode, toPort);
        }
    }

    NodeEditor.connecting = null;
    document.onmousemove = null;
    document.onmouseup = null;
}

// ============================================================================
// Connection Management
// ============================================================================

// Add connection
function addConnection(fromNode, fromPort, toNode, toPort) {
    // Check if exists
    const exists = NodeEditor.connections.some(c =>
        c.from === fromNode && c.fromPort === fromPort &&
        c.to === toNode && c.toPort === toPort
    );

    if (exists) return;

    const conn = { id: NodeEditor.nextId++, from: fromNode, fromPort, to: toNode, toPort };
    NodeEditor.connections.push(conn);
    renderConnection(conn);

    // Update conjunction node if the target is a conjunction
    const targetNode = NodeEditor.nodes.find(n => n.id === toNode);
    if (targetNode && targetNode.type === 'conjunction') {
        updateConjunctionNode(toNode);
    }
}

// Render connection
function renderConnection(conn) {
    const svg = document.getElementById('connectionsSVG');
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.id = 'conn-' + conn.id;
    line.setAttribute('stroke', 'url(#connection-gradient)');
    line.setAttribute('stroke-width', '3');
    line.setAttribute('stroke-linecap', 'round');
    line.style.cursor = 'pointer';
    line.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))';
    line.style.transition = 'all 0.15s ease';

    line.onmouseenter = () => {
        line.setAttribute('stroke-width', '4');
        line.style.filter = 'drop-shadow(0 4px 12px rgba(99, 102, 241, 0.6))';
    };
    line.onmouseleave = () => {
        line.setAttribute('stroke-width', '3');
        line.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))';
    };
    line.onclick = () => deleteConnection(conn.id);

    svg.appendChild(line);
    updateConnectionLine(conn.id);
}

// Create gradient definition for connections
function createConnectionGradient() {
    const svg = document.getElementById('connectionsSVG');
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.id = 'connection-gradient';
    gradient.setAttribute('x1', '0%');
    gradient.setAttribute('y1', '0%');
    gradient.setAttribute('x2', '100%');
    gradient.setAttribute('y2', '0%');

    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', '#2196F3');

    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', '#4CAF50');

    gradient.appendChild(stop1);
    gradient.appendChild(stop2);
    defs.appendChild(gradient);
    svg.appendChild(defs);
}

// Update connection line
function updateConnectionLine(connId) {
    const conn = NodeEditor.connections.find(c => c.id === connId);
    if (!conn) return;

    const line = document.getElementById('conn-' + connId);
    if (!line) return;

    // Find ports using data attributes
    const fromEl = document.querySelector(`#node-${conn.from} .port-out[data-port="${conn.fromPort}"]`);
    const toEl = document.querySelector(`#node-${conn.to} .port-in[data-port="${conn.toPort}"]`);

    if (!fromEl || !toEl) return;

    // Get the correct wrapper (normal mode or fullscreen mode)
    const canvas = document.getElementById('nodeCanvas');
    const container = canvas.parentElement;
    const containerRect = container.getBoundingClientRect();
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();

    // Calculate screen positions relative to container, then convert to canvas local coordinates
    const pos1 = wrapperToCanvas(
        fromRect.left - containerRect.left + fromRect.width / 2,
        fromRect.top - containerRect.top + fromRect.height / 2
    );
    const pos2 = wrapperToCanvas(
        toRect.left - containerRect.left + toRect.width / 2,
        toRect.top - containerRect.top + toRect.height / 2
    );

    line.setAttribute('x1', pos1.x);
    line.setAttribute('y1', pos1.y);
    line.setAttribute('x2', pos2.x);
    line.setAttribute('y2', pos2.y);
}

// Update all connections
function updateConnections() {
    NodeEditor.connections.forEach(c => updateConnectionLine(c.id));
}

// ============================================================================
// Node & Connection Deletion
// ============================================================================

function deleteNode(nodeId) {
    // Find conjunction nodes that were connected to this node
    const affectedConjunctions = new Set();
    NodeEditor.connections.forEach(c => {
        if (c.from === nodeId) {
            const targetNode = NodeEditor.nodes.find(n => n.id === c.to);
            if (targetNode && targetNode.type === 'conjunction') {
                affectedConjunctions.add(c.to);
            }
        }
    });

    // Remove connections
    NodeEditor.connections = NodeEditor.connections.filter(c => {
        if (c.from === nodeId || c.to === nodeId) {
            const line = document.getElementById('conn-' + c.id);
            if (line) line.remove();
            return false;
        }
        return true;
    });

    // Remove node
    NodeEditor.nodes = NodeEditor.nodes.filter(n => n.id !== nodeId);
    const el = document.getElementById('node-' + nodeId);
    if (el) el.remove();
    updateMinimap();

    // Update affected conjunction nodes to remove references
    affectedConjunctions.forEach(conjId => {
        updateConjunctionNode(conjId);
    });
}

// Delete connection
function deleteConnection(connId) {
    // Find the connection before deleting to check if it was connected to a conjunction
    const conn = NodeEditor.connections.find(c => c.id === connId);
    const targetNode = conn ? NodeEditor.nodes.find(n => n.id === conn.to) : null;

    NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== connId);
    const line = document.getElementById('conn-' + connId);
    if (line) line.remove();

    // Update conjunction node if the deleted connection was connected to one
    if (targetNode && targetNode.type === 'conjunction') {
        updateConjunctionNode(targetNode.id);
    }
}

function clearGraph() {
    if (!confirm('Clear all?')) return;

    NodeEditor.nodes.forEach(n => {
        const el = document.getElementById('node-' + n.id);
        if (el) el.remove();
    });

    NodeEditor.connections.forEach(c => {
        const line = document.getElementById('conn-' + c.id);
        if (line) line.remove();
    });

    NodeEditor.nodes = [];
    NodeEditor.connections = [];
    updateMinimap();
}

// ============================================================================
// Graph Execution
// ============================================================================

async function executeGraph() {
    // Find nodes
    const inputNode = NodeEditor.nodes.find(n => n.type === 'input');
    const aiNode = NodeEditor.nodes.find(n => n.type === 'aimodel');
    const outputNode = NodeEditor.nodes.find(n => n.type === 'output');

    if (!inputNode || !aiNode || !outputNode) {
        showToast('Need Input, AI Model, and Output nodes');
        return;
    }

    // Check connections
    const hasInput = NodeEditor.connections.some(c => c.to === aiNode.id);
    const hasOutput = NodeEditor.connections.some(c => c.from === aiNode.id);

    if (!hasInput || !hasOutput) {
        showToast('AI Model must be connected');
        return;
    }

    if (AppState.uploadQueue.length === 0) {
        showToast('Upload images first');
        return;
    }

    // Switch to results
    document.querySelector('.tab-btn[data-tab="results"]').click();

    // Process
    await processGraph(aiNode);
}

// Process graph
async function processGraph(aiNode) {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const paginationControls = document.getElementById('paginationControls');

    resultsGrid.innerHTML = '';
    paginationControls.style.display = 'none';
    downloadBtn.style.display = 'none';
    processingControls.style.display = 'flex';

    AppState.processedResults = [];
    AppState.allResults = [];
    AppState.currentPage = 1;
    isProcessing = true;

    const total = AppState.uploadQueue.length;
    let count = 0;

    // Get prompt if connected (from prompt or conjunction node)
    let prompt = '';

    // Check for direct prompt connection
    const promptNode = NodeEditor.nodes.find(n => n.type === 'prompt');
    const hasPromptConn = promptNode && NodeEditor.connections.some(c =>
        c.from === promptNode.id && c.to === aiNode.id
    );
    if (hasPromptConn) {
        prompt = promptNode.data.text || '';
    }

    // Check for conjunction node connection (takes precedence)
    const conjunctionNode = NodeEditor.nodes.find(n => n.type === 'conjunction');
    const hasConjunctionConn = conjunctionNode && NodeEditor.connections.some(c =>
        c.from === conjunctionNode.id && c.to === aiNode.id
    );
    if (hasConjunctionConn) {
        // Use the template and resolve placeholders
        let template = conjunctionNode.data.template || '';
        const items = conjunctionNode.data.connectedItems || [];

        // Create a map of reference keys to content
        const refMap = {};
        items.forEach(item => {
            refMap[item.refKey] = item.content;
        });

        // Replace all placeholders with actual content
        prompt = template.replace(/\{([^}]+)\}/g, (match, key) => {
            return refMap[key] !== undefined ? refMap[key] : match;
        });
    }

    // Process images
    for (const item of AppState.uploadQueue) {
        if (shouldStop) break;
        while (isPaused && !shouldStop) {
            await new Promise(r => setTimeout(r, 100));
        }

        const formData = new FormData();
        if (item.file) formData.append('image', item.file);
        else if (item.path) formData.append('image_path', item.path);

        formData.append('model', aiNode.data.model);
        formData.append('parameters', JSON.stringify(aiNode.data.parameters || {}));
        if (prompt) formData.append('prompt', prompt);

        try {
            const res = await fetch(`${AppState.apiBaseUrl}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(res.statusText);

            const data = await res.json();
            AppState.allResults.push({ queueItem: item, data });
            AppState.processedResults.push({
                filename: item.filename,
                caption: data.caption,
                path: item.path || item.filename
            });

            await addResultItemToCurrentPage(item, data);
        } catch (err) {
            console.error(err);
        }

        count++;
        showToast(`Processed ${count}/${total}`, true, count / total);
    }

    shouldStop = false;
    isPaused = false;
    isProcessing = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
        showToast('Done!');
    }
}

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
    const canvas = document.getElementById('nodeCanvas');
    const minimap = document.getElementById('nodeMinimap');

    // Move canvas and minimap into modal
    container.appendChild(canvas);
    if (minimap) container.appendChild(minimap);

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
            const wrapper = document.querySelector('.node-canvas-wrapper');
            const canvas = document.getElementById('nodeCanvas');
            const minimap = document.getElementById('nodeMinimap');

            modal.classList.remove('active', 'closing');
            wrapper.appendChild(canvas);
            if (minimap) wrapper.appendChild(minimap);
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
        data: type === 'prompt' ? { text: '' } :
              type === 'aimodel' ? { model: 'blip' } : {}
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
    header.innerHTML = `
        <span>${def.label}</span>
        <button class="node-del" data-id="${node.id}">×</button>
    `;
    header.onmousedown = (e) => startDrag(e, node);
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
            if (key) node.data[key] = e.target.value;
        };
    });
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
        return `<textarea data-key="text" placeholder="Enter prompt...">${node.data.text || ''}</textarea>`;
    }
    if (node.type === 'aimodel') {
        return `
            <select data-key="model">
                <option value="blip" ${node.data.model === 'blip' ? 'selected' : ''}>BLIP</option>
                <option value="r4b" ${node.data.model === 'r4b' ? 'selected' : ''}>R-4B</option>
            </select>
        `;
    }
    return '';
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

    // Get starting port position
    const portEl = document.querySelector(`#node-${nodeId} .port-out[data-port="${portIndex}"]`);
    const canvasRect = canvas.getBoundingClientRect();
    const portRect = portEl.getBoundingClientRect();

    NodeEditor.connecting = {
        from: nodeId,
        port: portIndex,
        startX: portRect.left - canvasRect.left + portRect.width / 2,
        startY: portRect.top - canvasRect.top + portRect.height / 2
    };

    document.onmousemove = updateTempConnection;
    document.onmouseup = endConnect;
}

function updateTempConnection(e) {
    if (!NodeEditor.connecting) return;

    const { canvas } = getElements();
    const canvasRect = canvas.getBoundingClientRect();
    const tempLine = document.getElementById('temp-connection');

    if (!tempLine) return;

    tempLine.setAttribute('x1', NodeEditor.connecting.startX);
    tempLine.setAttribute('y1', NodeEditor.connecting.startY);
    tempLine.setAttribute('x2', e.clientX - canvasRect.left);
    tempLine.setAttribute('y2', e.clientY - canvasRect.top);
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
}

// Delete connection
function deleteConnection(connId) {
    NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== connId);
    const line = document.getElementById('conn-' + connId);
    if (line) line.remove();
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

    // Get prompt if connected
    const promptNode = NodeEditor.nodes.find(n => n.type === 'prompt');
    const hasPromptConn = promptNode && NodeEditor.connections.some(c =>
        c.from === promptNode.id && c.to === aiNode.id
    );
    const prompt = hasPromptConn ? promptNode.data.text : '';

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
        formData.append('parameters', '{}');
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

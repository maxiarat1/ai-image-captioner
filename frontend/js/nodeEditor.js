// ============================================================================
// Node Editor - Simple & Barebone
// ============================================================================

// State
const NodeEditor = {
    nodes: [],
    connections: [],
    nextId: 1,
    dragging: null,
    connecting: null
};

// Node types
const NODES = {
    input: { label: 'Input', color: '#4CAF50', inputs: [], outputs: ['out'] },
    prompt: { label: 'Prompt', color: '#2196F3', inputs: [], outputs: ['out'] },
    aimodel: { label: 'AI Model', color: '#9C27B0', inputs: ['img', 'txt'], outputs: ['out'] },
    output: { label: 'Output', color: '#FF9800', inputs: ['in'], outputs: [] }
};

// Initialize
function initNodeEditor() {
    setupToolbar();

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

    // Move canvas into modal
    container.appendChild(canvas);

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Update connections after DOM change
    setTimeout(() => updateConnections(), 50);
}

// Close fullscreen modal
function closeFullscreen() {
    const modal = document.getElementById('nodeFullscreenModal');

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

            modal.classList.remove('active', 'closing');
            wrapper.appendChild(canvas);
            document.body.style.overflow = '';

            setTimeout(() => updateConnections(), 50);
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

// Add node
function addNode(type) {
    const def = NODES[type];
    const canvas = document.getElementById('nodeCanvas');
    const rect = canvas.getBoundingClientRect();

    const node = {
        id: NodeEditor.nextId++,
        type: type,
        x: 100 + Math.random() * 300,
        y: 100 + Math.random() * 200,
        data: type === 'prompt' ? { text: '' } :
              type === 'aimodel' ? { model: 'blip' } : {}
    };

    NodeEditor.nodes.push(node);
    renderNode(node);
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
    header.style.background = def.color;
    header.innerHTML = `
        <span>${def.label}</span>
        <button class="node-del" data-id="${node.id}">×</button>
    `;
    header.onmousedown = (e) => startDrag(e, node);
    el.appendChild(header);

    // Body
    const body = document.createElement('div');
    body.className = 'node-body';
    body.innerHTML = getNodeContent(node);
    el.appendChild(body);

    // Ports
    if (def.inputs.length) {
        const inputs = document.createElement('div');
        inputs.className = 'node-ports-in';
        def.inputs.forEach((port, i) => {
            const p = document.createElement('div');
            p.className = 'port port-in';
            p.dataset.node = node.id;
            p.dataset.port = i;
            inputs.appendChild(p);
        });
        el.appendChild(inputs);
    }

    if (def.outputs.length) {
        const outputs = document.createElement('div');
        outputs.className = 'node-ports-out';
        def.outputs.forEach((port, i) => {
            const p = document.createElement('div');
            p.className = 'port port-out';
            p.dataset.node = node.id;
            p.dataset.port = i;
            p.onmousedown = (e) => startConnect(e, node.id, i);
            outputs.appendChild(p);
        });
        el.appendChild(outputs);
    }

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

// Drag node
function startDrag(e, node) {
    if (e.target.classList.contains('node-del')) return;

    NodeEditor.dragging = {
        node: node,
        offsetX: e.clientX - node.x,
        offsetY: e.clientY - node.y
    };

    document.onmousemove = drag;
    document.onmouseup = stopDrag;
}

function drag(e) {
    if (!NodeEditor.dragging) return;

    const node = NodeEditor.dragging.node;
    node.x = e.clientX - NodeEditor.dragging.offsetX;
    node.y = e.clientY - NodeEditor.dragging.offsetY;

    const el = document.getElementById('node-' + node.id);
    el.style.left = node.x + 'px';
    el.style.top = node.y + 'px';

    updateConnections();
}

function stopDrag() {
    NodeEditor.dragging = null;
    document.onmousemove = null;
    document.onmouseup = null;
}

// Connect nodes
function startConnect(e, nodeId, portIndex) {
    e.stopPropagation();
    NodeEditor.connecting = { from: nodeId, port: portIndex };

    document.onmouseup = endConnect;
}

function endConnect(e) {
    if (!NodeEditor.connecting) return;

    const target = e.target;
    if (target.classList.contains('port-in')) {
        const toNode = parseInt(target.dataset.node);
        const toPort = parseInt(target.dataset.port);

        if (toNode !== NodeEditor.connecting.from) {
            addConnection(NodeEditor.connecting.from, NodeEditor.connecting.port, toNode, toPort);
        }
    }

    NodeEditor.connecting = null;
    document.onmouseup = null;
}

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
    line.setAttribute('stroke', '#666');
    line.setAttribute('stroke-width', '2');
    line.style.cursor = 'pointer';
    line.onclick = () => deleteConnection(conn.id);

    svg.appendChild(line);
    updateConnectionLine(conn.id);
}

// Update connection line
function updateConnectionLine(connId) {
    const conn = NodeEditor.connections.find(c => c.id === connId);
    if (!conn) return;

    const line = document.getElementById('conn-' + connId);
    if (!line) return;

    const fromEl = document.querySelector(`#node-${conn.from} .port-out:nth-child(${conn.fromPort + 1})`);
    const toEl = document.querySelector(`#node-${conn.to} .port-in:nth-child(${conn.toPort + 1})`);

    if (!fromEl || !toEl) return;

    const canvas = document.getElementById('nodeCanvas');
    const canvasRect = canvas.getBoundingClientRect();
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();

    const x1 = fromRect.left - canvasRect.left + fromRect.width / 2;
    const y1 = fromRect.top - canvasRect.top + fromRect.height / 2;
    const x2 = toRect.left - canvasRect.left + toRect.width / 2;
    const y2 = toRect.top - canvasRect.top + toRect.height / 2;

    line.setAttribute('x1', x1);
    line.setAttribute('y1', y1);
    line.setAttribute('x2', x2);
    line.setAttribute('y2', y2);
}

// Update all connections
function updateConnections() {
    NodeEditor.connections.forEach(c => updateConnectionLine(c.id));
}

// Delete node
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
}

// Delete connection
function deleteConnection(connId) {
    NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== connId);
    const line = document.getElementById('conn-' + connId);
    if (line) line.remove();
}

// Clear graph
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
}

// Execute graph
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

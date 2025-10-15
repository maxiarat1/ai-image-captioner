// ============================================================================
// Node Editor - Visual Programming Interface
// ============================================================================

// Node Editor State
const NodeEditorState = {
    nodes: [],
    connections: [],
    nextNodeId: 1,
    nextConnectionId: 1,
    selectedNode: null,
    draggingNode: null,
    connectingFrom: null,
    dragOffset: { x: 0, y: 0 },
    canvas: null,
    svgLayer: null
};

// Node type definitions
const NODE_TYPES = {
    INPUT: {
        type: 'input',
        label: 'Input',
        color: '#4CAF50',
        description: 'Uploaded images',
        inputs: [],
        outputs: ['images']
    },
    PROMPT: {
        type: 'prompt',
        label: 'Prompt',
        color: '#2196F3',
        description: 'Text prompt for AI',
        inputs: [],
        outputs: ['text']
    },
    AI_MODEL: {
        type: 'aimodel',
        label: 'AI Model',
        color: '#9C27B0',
        description: 'Process with AI',
        inputs: ['images', 'text'],
        outputs: ['captions']
    },
    OUTPUT: {
        type: 'output',
        label: 'Output',
        color: '#FF9800',
        description: 'Export results',
        inputs: ['captions'],
        outputs: []
    }
};

// Initialize node editor
function initNodeEditor() {
    NodeEditorState.canvas = document.getElementById('nodeCanvas');
    NodeEditorState.svgLayer = document.getElementById('connectionsSVG');

    if (!NodeEditorState.canvas) {
        console.error('Node canvas not found');
        return;
    }

    // Setup toolbar buttons
    setupNodeToolbar();

    // Setup canvas interactions
    setupCanvasInteractions();

    // Setup execute button
    const executeBtn = document.getElementById('executeGraphBtn');
    if (executeBtn) {
        executeBtn.addEventListener('click', executeNodeGraph);
    }

    // Setup clear graph button
    const clearBtn = document.getElementById('clearGraphBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearNodeGraph);
    }

    // Create default nodes for beginners
    createDefaultGraph();
}

// Setup node type toolbar
function setupNodeToolbar() {
    const toolbar = document.getElementById('nodeToolbar');
    if (!toolbar) return;

    toolbar.innerHTML = '';

    Object.values(NODE_TYPES).forEach(nodeType => {
        const button = document.createElement('button');
        button.className = 'node-toolbar-btn';
        button.style.borderLeft = `4px solid ${nodeType.color}`;
        button.innerHTML = `
            <div class="node-btn-label">${nodeType.label}</div>
            <div class="node-btn-desc">${nodeType.description}</div>
        `;
        button.addEventListener('click', () => createNode(nodeType.type));
        toolbar.appendChild(button);
    });
}

// Create a new node
function createNode(type, position = null) {
    const nodeType = Object.values(NODE_TYPES).find(nt => nt.type === type);
    if (!nodeType) return;

    const node = {
        id: `node-${NodeEditorState.nextNodeId++}`,
        type: type,
        label: nodeType.label,
        color: nodeType.color,
        position: position || getCanvasCenter(),
        data: getDefaultNodeData(type),
        inputs: nodeType.inputs,
        outputs: nodeType.outputs
    };

    NodeEditorState.nodes.push(node);
    renderNode(node);

    return node;
}

// Get default data for each node type
function getDefaultNodeData(type) {
    switch (type) {
        case 'input':
            return { source: 'uploadQueue' };
        case 'prompt':
            return { text: '' };
        case 'aimodel':
            return {
                model: 'blip',
                parameters: {}
            };
        case 'output':
            return { format: 'results' };
        default:
            return {};
    }
}

// Get canvas center position
function getCanvasCenter() {
    const canvas = NodeEditorState.canvas;
    const rect = canvas.getBoundingClientRect();
    return {
        x: rect.width / 2 - 100,
        y: rect.height / 2 - 60 + Math.random() * 40 - 20 // Small random offset
    };
}

// Render a node to the canvas
function renderNode(node) {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'node';
    nodeEl.id = node.id;
    nodeEl.style.left = `${node.position.x}px`;
    nodeEl.style.top = `${node.position.y}px`;
    nodeEl.style.borderTopColor = node.color;

    // Node header
    const header = document.createElement('div');
    header.className = 'node-header';
    header.style.backgroundColor = node.color;
    header.innerHTML = `
        <span class="node-title">${node.label}</span>
        <button class="node-delete-btn" onclick="deleteNode('${node.id}')">&times;</button>
    `;
    nodeEl.appendChild(header);

    // Node body with inputs
    const body = document.createElement('div');
    body.className = 'node-body';
    body.appendChild(createNodeContent(node));
    nodeEl.appendChild(body);

    // Input ports
    if (node.inputs.length > 0) {
        const inputsContainer = document.createElement('div');
        inputsContainer.className = 'node-inputs';
        node.inputs.forEach((inputName, index) => {
            const port = document.createElement('div');
            port.className = 'node-port node-port-input';
            port.dataset.nodeId = node.id;
            port.dataset.portName = inputName;
            port.dataset.portIndex = index;
            port.title = inputName;
            inputsContainer.appendChild(port);
        });
        nodeEl.appendChild(inputsContainer);
    }

    // Output ports
    if (node.outputs.length > 0) {
        const outputsContainer = document.createElement('div');
        outputsContainer.className = 'node-outputs';
        node.outputs.forEach((outputName, index) => {
            const port = document.createElement('div');
            port.className = 'node-port node-port-output';
            port.dataset.nodeId = node.id;
            port.dataset.portName = outputName;
            port.dataset.portIndex = index;
            port.title = outputName;
            port.addEventListener('mousedown', startConnection);
            outputsContainer.appendChild(port);
        });
        nodeEl.appendChild(outputsContainer);
    }

    // Make draggable
    header.addEventListener('mousedown', startDrag);

    NodeEditorState.canvas.appendChild(nodeEl);
}

// Create node-specific content
function createNodeContent(node) {
    const content = document.createElement('div');
    content.className = 'node-content';

    switch (node.type) {
        case 'input':
            content.innerHTML = `
                <div class="node-field">
                    <label>Source:</label>
                    <select class="node-select" onchange="updateNodeData('${node.id}', 'source', this.value)">
                        <option value="uploadQueue" ${node.data.source === 'uploadQueue' ? 'selected' : ''}>Upload Queue</option>
                    </select>
                </div>
            `;
            break;

        case 'prompt':
            content.innerHTML = `
                <div class="node-field">
                    <label>Prompt Text:</label>
                    <textarea class="node-textarea"
                              placeholder="Enter prompt..."
                              oninput="updateNodeData('${node.id}', 'text', this.value)">${node.data.text || ''}</textarea>
                </div>
            `;
            break;

        case 'aimodel':
            content.innerHTML = `
                <div class="node-field">
                    <label>Model:</label>
                    <select class="node-select" onchange="updateNodeData('${node.id}', 'model', this.value)">
                        <option value="blip" ${node.data.model === 'blip' ? 'selected' : ''}>BLIP (Fast)</option>
                        <option value="r4b" ${node.data.model === 'r4b' ? 'selected' : ''}>R-4B (Advanced)</option>
                    </select>
                </div>
            `;
            break;

        case 'output':
            content.innerHTML = `
                <div class="node-field">
                    <label>Output:</label>
                    <select class="node-select" onchange="updateNodeData('${node.id}', 'format', this.value)">
                        <option value="results" ${node.data.format === 'results' ? 'selected' : ''}>Results Tab</option>
                        <option value="export" ${node.data.format === 'export' ? 'selected' : ''}>Export ZIP</option>
                    </select>
                </div>
            `;
            break;
    }

    return content;
}

// Update node data
function updateNodeData(nodeId, key, value) {
    const node = NodeEditorState.nodes.find(n => n.id === nodeId);
    if (node) {
        node.data[key] = value;
    }
}

// Start dragging a node
function startDrag(e) {
    e.preventDefault();
    const nodeEl = e.target.closest('.node');
    const nodeId = nodeEl.id;
    const node = NodeEditorState.nodes.find(n => n.id === nodeId);

    if (!node) return;

    NodeEditorState.draggingNode = node;
    NodeEditorState.dragOffset = {
        x: e.clientX - node.position.x,
        y: e.clientY - node.position.y
    };

    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDrag);
}

function drag(e) {
    if (!NodeEditorState.draggingNode) return;

    const node = NodeEditorState.draggingNode;
    node.position.x = e.clientX - NodeEditorState.dragOffset.x;
    node.position.y = e.clientY - NodeEditorState.dragOffset.y;

    const nodeEl = document.getElementById(node.id);
    nodeEl.style.left = `${node.position.x}px`;
    nodeEl.style.top = `${node.position.y}px`;

    updateConnections();
}

function stopDrag() {
    NodeEditorState.draggingNode = null;
    document.removeEventListener('mousemove', drag);
    document.removeEventListener('mouseup', stopDrag);
}

// Start creating a connection
function startConnection(e) {
    e.stopPropagation();
    const port = e.target;

    NodeEditorState.connectingFrom = {
        nodeId: port.dataset.nodeId,
        portName: port.dataset.portName,
        portIndex: port.dataset.portIndex,
        element: port
    };

    const tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    tempLine.id = 'temp-connection';
    tempLine.setAttribute('stroke', '#666');
    tempLine.setAttribute('stroke-width', '2');
    tempLine.setAttribute('stroke-dasharray', '5,5');

    const rect = port.getBoundingClientRect();
    const canvasRect = NodeEditorState.canvas.getBoundingClientRect();
    const startX = rect.left - canvasRect.left + rect.width / 2;
    const startY = rect.top - canvasRect.top + rect.height / 2;

    tempLine.setAttribute('x1', startX);
    tempLine.setAttribute('y1', startY);
    tempLine.setAttribute('x2', startX);
    tempLine.setAttribute('y2', startY);

    NodeEditorState.svgLayer.appendChild(tempLine);

    document.addEventListener('mousemove', updateTempConnection);
    document.addEventListener('mouseup', endConnection);
}

function updateTempConnection(e) {
    const tempLine = document.getElementById('temp-connection');
    if (!tempLine) return;

    const canvasRect = NodeEditorState.canvas.getBoundingClientRect();
    const x = e.clientX - canvasRect.left;
    const y = e.clientY - canvasRect.top;

    tempLine.setAttribute('x2', x);
    tempLine.setAttribute('y2', y);
}

function endConnection(e) {
    const tempLine = document.getElementById('temp-connection');
    if (tempLine) {
        tempLine.remove();
    }

    // Check if mouse is over an input port
    const targetPort = document.elementFromPoint(e.clientX, e.clientY);

    if (targetPort && targetPort.classList.contains('node-port-input') && NodeEditorState.connectingFrom) {
        createConnection(
            NodeEditorState.connectingFrom.nodeId,
            NodeEditorState.connectingFrom.portName,
            targetPort.dataset.nodeId,
            targetPort.dataset.portName
        );
    }

    NodeEditorState.connectingFrom = null;
    document.removeEventListener('mousemove', updateTempConnection);
    document.removeEventListener('mouseup', endConnection);
}

// Create a connection between nodes
function createConnection(fromNodeId, fromPort, toNodeId, toPort) {
    // Validate connection
    if (fromNodeId === toNodeId) {
        showToast('Cannot connect node to itself');
        return;
    }

    // Check if connection already exists
    const existing = NodeEditorState.connections.find(c =>
        c.from.nodeId === fromNodeId &&
        c.from.port === fromPort &&
        c.to.nodeId === toNodeId &&
        c.to.port === toPort
    );

    if (existing) {
        showToast('Connection already exists');
        return;
    }

    const connection = {
        id: `conn-${NodeEditorState.nextConnectionId++}`,
        from: { nodeId: fromNodeId, port: fromPort },
        to: { nodeId: toNodeId, port: toPort }
    };

    NodeEditorState.connections.push(connection);
    renderConnection(connection);
}

// Render a connection
function renderConnection(connection) {
    const fromNode = NodeEditorState.nodes.find(n => n.id === connection.from.nodeId);
    const toNode = NodeEditorState.nodes.find(n => n.id === connection.to.nodeId);

    if (!fromNode || !toNode) return;

    const fromEl = document.querySelector(`#${connection.from.nodeId} .node-port-output[data-port-name="${connection.from.port}"]`);
    const toEl = document.querySelector(`#${connection.to.nodeId} .node-port-input[data-port-name="${connection.to.port}"]`);

    if (!fromEl || !toEl) return;

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.id = connection.id;
    line.classList.add('connection-line');
    line.setAttribute('stroke', fromNode.color);
    line.setAttribute('stroke-width', '3');
    line.style.cursor = 'pointer';

    line.addEventListener('click', () => deleteConnection(connection.id));

    NodeEditorState.svgLayer.appendChild(line);
    updateConnectionLine(connection.id);
}

// Update connection line position
function updateConnectionLine(connectionId) {
    const connection = NodeEditorState.connections.find(c => c.id === connectionId);
    if (!connection) return;

    const line = document.getElementById(connectionId);
    if (!line) return;

    const fromEl = document.querySelector(`#${connection.from.nodeId} .node-port-output[data-port-name="${connection.from.port}"]`);
    const toEl = document.querySelector(`#${connection.to.nodeId} .node-port-input[data-port-name="${connection.to.port}"]`);

    if (!fromEl || !toEl) return;

    const canvasRect = NodeEditorState.canvas.getBoundingClientRect();
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
    NodeEditorState.connections.forEach(conn => {
        updateConnectionLine(conn.id);
    });
}

// Delete a node
function deleteNode(nodeId) {
    // Remove connections
    NodeEditorState.connections = NodeEditorState.connections.filter(conn => {
        if (conn.from.nodeId === nodeId || conn.to.nodeId === nodeId) {
            const line = document.getElementById(conn.id);
            if (line) line.remove();
            return false;
        }
        return true;
    });

    // Remove node
    NodeEditorState.nodes = NodeEditorState.nodes.filter(n => n.id !== nodeId);

    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) nodeEl.remove();
}

// Delete a connection
function deleteConnection(connectionId) {
    NodeEditorState.connections = NodeEditorState.connections.filter(c => c.id !== connectionId);
    const line = document.getElementById(connectionId);
    if (line) line.remove();
}

// Clear entire graph
function clearNodeGraph() {
    if (!confirm('Clear all nodes and connections?')) return;

    NodeEditorState.nodes.forEach(node => {
        const nodeEl = document.getElementById(node.id);
        if (nodeEl) nodeEl.remove();
    });

    NodeEditorState.connections.forEach(conn => {
        const line = document.getElementById(conn.id);
        if (line) line.remove();
    });

    NodeEditorState.nodes = [];
    NodeEditorState.connections = [];

    showToast('Graph cleared');
}

// Create a default graph for beginners
function createDefaultGraph() {
    // Create nodes with specific positions
    const inputNode = createNode('input', { x: 100, y: 200 });
    const promptNode = createNode('prompt', { x: 100, y: 350 });
    const aiModelNode = createNode('aimodel', { x: 400, y: 250 });
    const outputNode = createNode('output', { x: 700, y: 250 });

    // Wait for DOM to update, then create connections
    setTimeout(() => {
        createConnection(inputNode.id, 'images', aiModelNode.id, 'images');
        createConnection(promptNode.id, 'text', aiModelNode.id, 'text');
        createConnection(aiModelNode.id, 'captions', outputNode.id, 'captions');
    }, 100);
}

// Setup canvas interactions
function setupCanvasInteractions() {
    // Resize SVG layer to match canvas
    const resizeSVG = () => {
        const canvas = NodeEditorState.canvas;
        const svg = NodeEditorState.svgLayer;
        svg.setAttribute('width', canvas.scrollWidth);
        svg.setAttribute('height', canvas.scrollHeight);
        updateConnections();
    };

    resizeSVG();
    window.addEventListener('resize', resizeSVG);
}

// Execute the node graph
async function executeNodeGraph() {
    // Validate graph
    const validation = validateGraph();
    if (!validation.valid) {
        showToast(validation.error);
        return;
    }

    // Check if we have images
    if (AppState.uploadQueue.length === 0) {
        showToast('Please upload images first');
        return;
    }

    // Switch to Results tab
    const resultsTab = document.querySelector('.tab-btn[data-tab="results"]');
    resultsTab.click();

    // Execute the graph
    await processNodesGraph();
}

// Validate the node graph
function validateGraph() {
    // Find required nodes
    const inputNode = NodeEditorState.nodes.find(n => n.type === 'input');
    const aiModelNode = NodeEditorState.nodes.find(n => n.type === 'aimodel');
    const outputNode = NodeEditorState.nodes.find(n => n.type === 'output');

    if (!inputNode) {
        return { valid: false, error: 'Input node is required' };
    }

    if (!aiModelNode) {
        return { valid: false, error: 'AI Model node is required' };
    }

    if (!outputNode) {
        return { valid: false, error: 'Output node is required' };
    }

    // Check if AI Model has input connection
    const hasImageInput = NodeEditorState.connections.some(c =>
        c.to.nodeId === aiModelNode.id && c.to.port === 'images'
    );

    if (!hasImageInput) {
        return { valid: false, error: 'AI Model needs an image input connection' };
    }

    // Check if AI Model has output connection
    const hasOutput = NodeEditorState.connections.some(c =>
        c.from.nodeId === aiModelNode.id
    );

    if (!hasOutput) {
        return { valid: false, error: 'AI Model needs an output connection' };
    }

    return { valid: true };
}

// Process the node graph (similar to processImages but using graph)
async function processNodesGraph() {
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadBtn = document.getElementById('downloadAllBtn');
    const processingControls = document.getElementById('processingControls');
    const paginationControls = document.getElementById('paginationControls');

    // Reset state
    resultsGrid.innerHTML = '';
    paginationControls.style.display = 'none';
    downloadBtn.style.display = 'none';
    processingControls.style.display = 'flex';
    AppState.processedResults = [];
    AppState.allResults = [];
    AppState.currentPage = 1;
    isProcessing = true;

    const totalImages = AppState.uploadQueue.length;
    let processedCount = 0;

    // Get AI Model node configuration
    const aiModelNode = NodeEditorState.nodes.find(n => n.type === 'aimodel');
    const promptNode = NodeEditorState.nodes.find(n => n.type === 'prompt');

    const model = aiModelNode.data.model || 'blip';
    const prompt = promptNode ? promptNode.data.text : '';

    // Process each image
    for (const queueItem of AppState.uploadQueue) {
        // Check if should stop
        if (shouldStop) {
            showToast('Processing stopped');
            break;
        }

        // Wait while paused
        while (isPaused && !shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Build request
        const formData = new FormData();

        if (queueItem.file) {
            formData.append('image', queueItem.file);
        } else if (queueItem.path) {
            formData.append('image_path', queueItem.path);
        }

        formData.append('model', model);
        formData.append('parameters', JSON.stringify({}));

        if (prompt) {
            formData.append('prompt', prompt);
        }

        try {
            const response = await fetch(`${AppState.apiBaseUrl}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // Store result
            AppState.allResults.push({ queueItem, data });

            AppState.processedResults.push({
                filename: queueItem.filename,
                caption: data.caption,
                path: queueItem.path || queueItem.filename
            });

            await addResultItemToCurrentPage(queueItem, data);

        } catch (error) {
            console.error(`Error processing ${queueItem.filename}:`, error);
        }

        processedCount++;
        const progress = processedCount / totalImages;
        showToast(`Processed ${processedCount}/${totalImages}`, true, progress);
    }

    // Reset flags
    shouldStop = false;
    isPaused = false;

    isProcessing = false;
    processingControls.style.display = 'none';

    if (AppState.processedResults.length > 0) {
        downloadBtn.style.display = 'inline-flex';
        showToast('All images processed!');
    }
}

/**
 * CONNECTION SYSTEM GUIDE
 *
 * Port types are defined in state.js NODES object.
 * For dynamic ports (e.g., Curate), use getOutputPortType() helper.
 *
 * To add new node type:
 * 1. Define in state.js: inputs: [...], outputs: [...]
 * 2. Add port colors in PORT_COLORS below
 * 3. Update arePortsCompatible() for connection rules
 * 4. For dynamic ports: add logic in ports.js getOutputPortType()
 */

// Connections: creation, rendering, updates, and deletion
(function() {
    const NEConnections = {};

    // Port type to color mapping
    const PORT_COLORS = {
        'images': '#FF6B6B',
        'text': '#4ECDC4',
        'prompt': '#4ECDC4',
        'captions': '#4ECDC4',
        'data': '#FFD93D',
        'route': '#A855F7'  // Purple for routing ports
    };

    // Connection state
    let connectionState = {
        active: false,
        fromNode: null,
        fromPort: null,
        startX: 0,
        startY: 0,
        portColor: null,
        tempPath: null,
        mouseMoveHandler: null,
        mouseUpHandler: null
    };

    /**
     * Get color for a port type
     */
    function getPortColor(portType) {
        return PORT_COLORS[portType] || '#888888';
    }

    /**
     * Check if two port types are compatible
     */
    function arePortsCompatible(fromPortType, toPortType) {
        // Route ports can accept images or text (they're flexible)
        if (toPortType === 'route') {
            return true;  // Route ports accept any type
        }

        // Route output can connect to images or text inputs
        if (fromPortType === 'route') {
            return toPortType === 'images' ||
                   toPortType === 'text' ||
                   toPortType === 'prompt' ||
                   toPortType === 'captions' ||
                   toPortType === 'data';
        }

        // Images can only connect to images
        if (fromPortType === 'images' || toPortType === 'images') {
            return fromPortType === 'images' && toPortType === 'images';
        }

        // Text-based port compatibility rules
        // 'captions' can connect to anything text-based (allows AI chaining)
        if (fromPortType === 'captions' && ['text', 'prompt', 'captions'].includes(toPortType)) {
            return true;
        }

        // 'text' and 'prompt' can connect to each other but NOT to 'captions'
        if (['text', 'prompt'].includes(fromPortType) && ['text', 'prompt'].includes(toPortType)) {
            return true;
        }

        // Data port accepts anything
        if (toPortType === 'data') {
            return true;
        }

        return false;
    }

    /**
     * Get port position in canvas coordinates
     * Properly handles viewport transform by measuring in screen space and converting to canvas space
     */
    function getPortPosition(nodeId, portIndex, isOutput) {
        // Get node data (in canvas coordinates)
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return null;

        // Get port and node elements
        const portSelector = isOutput 
            ? `#node-${nodeId} .port-out[data-port="${portIndex}"]`
            : `#node-${nodeId} .port-in[data-port="${portIndex}"]`;
        const portEl = document.querySelector(portSelector);
        const nodeEl = document.getElementById(`node-${nodeId}`);
        if (!portEl || !nodeEl) return null;

        // Get screen positions
        const nodeRect = nodeEl.getBoundingClientRect();
        const portRect = portEl.getBoundingClientRect();

        // Check if elements are visible (have non-zero dimensions)
        // This prevents incorrect measurements when tab is hidden
        if (nodeRect.width === 0 || nodeRect.height === 0) return null;

        // Calculate port center relative to node's top-left (in screen pixels, already scaled)
        const screenOffsetX = portRect.left - nodeRect.left + portRect.width / 2;
        const screenOffsetY = portRect.top - nodeRect.top + portRect.height / 2;

        // Convert screen offset to canvas offset (unscale)
        const scale = NodeEditor.transform.scale;
        const canvasOffsetX = screenOffsetX / scale;
        const canvasOffsetY = screenOffsetY / scale;

        // Return absolute canvas position
        return {
            x: node.x + canvasOffsetX,
            y: node.y + canvasOffsetY
        };
    }

    /**
     * Create bezier curve path data
     */
    function createBezierPath(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Dynamic control point offset
        const offset = Math.max(80, Math.min(distance * 0.5, 200));
        
        const cp1x = x1 + offset;
        const cp1y = y1;
        const cp2x = x2 - offset;
        const cp2y = y2;
        
        return `M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`;
    }

    /**
     * Find input port element under mouse position
     */
    function findPortUnderMouse(e) {
        const elements = document.elementsFromPoint(e.clientX, e.clientY);
        for (const el of elements) {
            if (el.classList && el.classList.contains('port-in')) {
                return {
                    element: el,
                    nodeId: parseInt(el.dataset.node),
                    portIndex: parseInt(el.dataset.port),
                    portType: el.dataset.portType
                };
            }
        }
        return null;
    }

    // Create gradient definition for connections
    NEConnections.createConnectionGradient = function() {
        const { svg } = NEUtils.getElements();
        if (!svg) return;
        
        // Set SVG viewBox to match canvas size (5000x5000)
        svg.setAttribute('viewBox', '0 0 5000 5000');
        svg.setAttribute('preserveAspectRatio', 'none');
        
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        svg.appendChild(defs);
    };

    /**
     * Create a dynamic gradient for a specific connection
     */
    function createConnectionGradient(connId, fromColor, toColor, x1, y1, x2, y2) {
        const { svg } = NEUtils.getElements();
        if (!svg) return null;

        const defs = svg.querySelector('defs');
        if (!defs) return null;

        // Remove old gradient if exists
        const oldGradient = document.getElementById(`gradient-${connId}`);
        if (oldGradient) oldGradient.remove();

        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.id = `gradient-${connId}`;
        gradient.setAttribute('gradientUnits', 'userSpaceOnUse');
        gradient.setAttribute('x1', x1);
        gradient.setAttribute('y1', y1);
        gradient.setAttribute('x2', x2);
        gradient.setAttribute('y2', y2);

        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', fromColor);

        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', toColor);

        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        defs.appendChild(gradient);

        return `url(#gradient-${connId})`;
    }

    /**
     * Start connecting from an output port
     */
    NEConnections.startConnect = function(e, nodeId, portIndex) {
        e.stopPropagation();
        e.preventDefault();

        const { canvas, svg } = NEUtils.getElements();
        if (!canvas || !svg) return;
        
        // Get starting position
        const startPos = getPortPosition(nodeId, portIndex, true);
        if (!startPos) return;

        // Get port type and color
        const portEl = document.querySelector(`#node-${nodeId} .port-out[data-port="${portIndex}"]`);
        const portType = portEl.dataset.portType;
        const portColor = getPortColor(portType);

        // Create temporary connection path
        const tempPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        tempPath.id = 'temp-connection';
        tempPath.setAttribute('fill', 'none');
        tempPath.setAttribute('stroke', portColor);
        tempPath.setAttribute('stroke-width', '3');
        tempPath.setAttribute('stroke-linecap', 'round');
        tempPath.setAttribute('stroke-dasharray', '5,5');
        tempPath.style.opacity = '0.7';
        tempPath.style.pointerEvents = 'none';
        svg.appendChild(tempPath);

        // Set connection state
        canvas.classList.add('connecting');
        connectionState = {
            active: true,
            fromNode: nodeId,
            fromPort: portIndex,
            startX: startPos.x,
            startY: startPos.y,
            portColor: portColor,
            portType: portType,
            tempPath: tempPath,
            mouseMoveHandler: updateTempConnection,
            mouseUpHandler: endConnect
        };

        // Add event listeners
        document.addEventListener('mousemove', connectionState.mouseMoveHandler);
        document.addEventListener('mouseup', connectionState.mouseUpHandler);

        // Legacy support
        NodeEditor.connecting = {
            from: nodeId,
            port: portIndex,
            startX: startPos.x,
            startY: startPos.y,
            portColor: portColor
        };
    };

    /**
     * Start connecting from an input port (reconnection)
     */
    NEConnections.startConnectFromInput = function(e, nodeId, portIndex) {
        e.stopPropagation();
        e.preventDefault();

        // Find existing connection to this input
        const existingConnection = NodeEditor.connections.find(c =>
            c.to === nodeId && c.toPort === portIndex
        );

        if (!existingConnection) return;

        // Store source info before removing
        const sourceNodeId = existingConnection.from;
        const sourcePortIndex = existingConnection.fromPort;

        // Update conjunction if needed
        const targetNode = NodeEditor.nodes.find(n => n.id === nodeId);
        const needsConjunctionUpdate = targetNode && targetNode.type === 'conjunction';

        // Remove existing connection
        NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== existingConnection.id);
        const path = document.getElementById('conn-' + existingConnection.id);
        if (path) path.remove();

        const gradient = document.getElementById(`gradient-${existingConnection.id}`);
        if (gradient) gradient.remove();

        // Update conjunction
        if (needsConjunctionUpdate && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(nodeId);
        }

        // Start new connection from original source
        NEConnections.startConnect(e, sourceNodeId, sourcePortIndex);
    };

    /**
     * Update temporary connection during mouse move
     */
    function updateTempConnection(e) {
        if (!connectionState.active) return;

        const { canvas } = NEUtils.getElements();
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();

        // Get mouse position in canvas coordinates
        const mousePos = NEUtils.wrapperToCanvas(
            e.clientX - containerRect.left,
            e.clientY - containerRect.top
        );

        // Update path
        const pathData = createBezierPath(
            connectionState.startX, 
            connectionState.startY,
            mousePos.x, 
            mousePos.y
        );
        connectionState.tempPath.setAttribute('d', pathData);

        // Check if hovering over a valid input port
        const targetPort = findPortUnderMouse(e);
        
        if (targetPort && targetPort.nodeId !== connectionState.fromNode) {
            // Get source node info
            const fromNode = NodeEditor.nodes.find(n => n.id === connectionState.fromNode);
            const toNode = NodeEditor.nodes.find(n => n.id === targetPort.nodeId);
            
            if (fromNode && toNode) {
                const fromDef = NODES[fromNode.type];
                const toDef = NODES[toNode.type];
                const fromPortType = typeof NENodes.getOutputPortType === 'function'
                    ? NENodes.getOutputPortType(fromNode, connectionState.fromPort)
                    : fromDef.outputs[connectionState.fromPort];
                const toPortType = toDef.inputs[targetPort.portIndex];

                // Check for circular dependency
                const wouldCreateCycle = NodeEditor.connections.some(c =>
                    c.from === targetPort.nodeId && c.to === connectionState.fromNode
                );

                // Update visual feedback
                if (wouldCreateCycle || !arePortsCompatible(fromPortType, toPortType)) {
                    connectionState.tempPath.setAttribute('stroke', '#ef4444');
                    connectionState.tempPath.style.opacity = '0.7';
                } else {
                    connectionState.tempPath.setAttribute('stroke', connectionState.portColor);
                    connectionState.tempPath.style.opacity = '0.9';
                }
                return;
            }
        }

        // Default: reset to source color
        connectionState.tempPath.setAttribute('stroke', connectionState.portColor);
        connectionState.tempPath.style.opacity = '0.7';
    }

    /**
     * End connection attempt
     */
    function endConnect(e) {
        if (!connectionState.active) return;

        const { canvas } = NEUtils.getElements();
        if (canvas) canvas.classList.remove('connecting');

        // Remove temporary path
        if (connectionState.tempPath) {
            connectionState.tempPath.remove();
        }

        // Check if dropped on a valid input port
        const targetPort = findPortUnderMouse(e);
        if (targetPort && targetPort.nodeId !== connectionState.fromNode) {
            NEConnections.addConnection(
                connectionState.fromNode,
                connectionState.fromPort,
                targetPort.nodeId,
                targetPort.portIndex
            );
        }

        // Clean up event listeners
        document.removeEventListener('mousemove', connectionState.mouseMoveHandler);
        document.removeEventListener('mouseup', connectionState.mouseUpHandler);

        // Reset state
        connectionState = {
            active: false,
            fromNode: null,
            fromPort: null,
            startX: 0,
            startY: 0,
            portColor: null,
            tempPath: null,
            mouseMoveHandler: null,
            mouseUpHandler: null
        };

        // Legacy cleanup
        NodeEditor.connecting = null;
    }

    /**
     * Add a new connection between nodes
     */
    NEConnections.addConnection = function(fromNode, fromPort, toNode, toPort) {
        // Check if connection already exists
        const exists = NodeEditor.connections.some(c =>
            c.from === fromNode && c.fromPort === fromPort &&
            c.to === toNode && c.toPort === toPort
        );
        if (exists) return;

        // Get node definitions
        const fromNodeObj = NodeEditor.nodes.find(n => n.id === fromNode);
        const toNodeObj = NodeEditor.nodes.find(n => n.id === toNode);
        if (!fromNodeObj || !toNodeObj) return;

        // Get port types
        const fromDef = NODES[fromNodeObj.type];
        const toDef = NODES[toNodeObj.type];
        const fromPortType = typeof NENodes.getOutputPortType === 'function'
            ? NENodes.getOutputPortType(fromNodeObj, fromPort)
            : fromDef.outputs[fromPort];
        const toPortType = toDef.inputs[toPort];

        // Validate compatibility
        if (!arePortsCompatible(fromPortType, toPortType)) {
            console.warn(`Incompatible ports: ${fromPortType} → ${toPortType}`);
            return;
        }

        // Prevent circular dependencies
        const wouldCreateCycle = NodeEditor.connections.some(c =>
            c.from === toNode && c.to === fromNode
        );
        if (wouldCreateCycle) {
            console.warn(`Circular dependency blocked: ${toNode} → ${fromNode}`);
            return;
        }

        // Remove any existing connection to the same input port (single input rule)
        // Exception: Conjunction nodes can accept multiple connections on their captions input
        const allowMultiple = toNodeObj.type === 'conjunction' && toPortType === 'captions';
        if (!allowMultiple) {
            const existingToInput = NodeEditor.connections.find(c =>
                c.to === toNode && c.toPort === toPort
            );
            if (existingToInput) {
                NEConnections.deleteConnection(existingToInput.id);
            }
        }

        // Create and add connection
        const conn = { 
            id: NodeEditor.nextId++, 
            from: fromNode, 
            fromPort, 
            to: toNode, 
            toPort 
        };
        NodeEditor.connections.push(conn);
        renderConnection(conn);

        // Update conjunction node if needed
        if (toNodeObj.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(toNode);
        }
    };

    /**
     * Render a connection as SVG path
     */
    function renderConnection(conn) {
        const { svg } = NEUtils.getElements();
        if (!svg) return;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.id = 'conn-' + conn.id;
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', 'url(#connection-gradient)');
        path.setAttribute('stroke-width', '3');
        path.setAttribute('stroke-linecap', 'round');
        path.style.cursor = 'pointer';
        path.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))';
        path.style.transition = 'all 0.15s ease';

        // Hover effects
        path.onmouseenter = () => {
            path.setAttribute('stroke-width', '4');
            path.style.filter = 'drop-shadow(0 4px 12px rgba(99, 102, 241, 0.6))';
        };
        path.onmouseleave = () => {
            path.setAttribute('stroke-width', '3');
            path.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))';
        };
        
        // Click to delete
        path.onclick = () => NEConnections.deleteConnection(conn.id);

        svg.appendChild(path);
        
        // Defer initial position update to next frame to ensure DOM is ready
        requestAnimationFrame(() => {
            NEConnections.updateConnectionLine(conn.id);
        });
    }

    /**
     * Update a single connection line
     */
    NEConnections.updateConnectionLine = function(connId) {
        const conn = NodeEditor.connections.find(c => c.id === connId);
        if (!conn) return;

        const path = document.getElementById('conn-' + connId);
        if (!path) return;

        // Get port positions
        const pos1 = getPortPosition(conn.from, conn.fromPort, true);
        const pos2 = getPortPosition(conn.to, conn.toPort, false);
        if (!pos1 || !pos2) return;

        // Update path
        const pathData = createBezierPath(pos1.x, pos1.y, pos2.x, pos2.y);
        path.setAttribute('d', pathData);

        // Get port types and colors
        const fromEl = document.querySelector(`#node-${conn.from} .port-out[data-port="${conn.fromPort}"]`);
        const toEl = document.querySelector(`#node-${conn.to} .port-in[data-port="${conn.toPort}"]`);
        if (!fromEl || !toEl) return;

        const fromColor = getPortColor(fromEl.dataset.portType);
        const toColor = getPortColor(toEl.dataset.portType);

        // Update gradient
        const gradientUrl = createConnectionGradient(connId, fromColor, toColor, pos1.x, pos1.y, pos2.x, pos2.y);
        if (gradientUrl) {
            path.setAttribute('stroke', gradientUrl);
        }
    };

    /**
     * Update all connection lines
     */
    NEConnections.updateConnections = function() {
        NodeEditor.connections.forEach(c => NEConnections.updateConnectionLine(c.id));
    };

    /**
     * Delete a connection
     */
    NEConnections.deleteConnection = function(connId) {
        const conn = NodeEditor.connections.find(c => c.id === connId);
        const targetNode = conn ? NodeEditor.nodes.find(n => n.id === conn.to) : null;

        // Remove from array
        NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== connId);

        // Remove SVG elements
        const path = document.getElementById('conn-' + connId);
        if (path) path.remove();

        const gradient = document.getElementById(`gradient-${connId}`);
        if (gradient) gradient.remove();

        // Update conjunction if needed
        if (targetNode && targetNode.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(targetNode.id);
        }
    };

    // Export namespace and selected aliases for compatibility
    window.NEConnections = NEConnections;
})();

// Connections: creation, rendering, updates, and deletion
(function() {
    const NEConnections = {};

    // Port type to color mapping
    const PORT_COLORS = {
        'images': '#FF6B6B',
        'text': '#4ECDC4',
        'prompt': '#4ECDC4',
        'captions': '#4ECDC4',
        'data': '#FFD93D'
    };

    // Get color for a port type
    function getPortColor(portType) {
        return PORT_COLORS[portType] || '#888888';
    }

    // Create gradient definition for connections
    NEConnections.createConnectionGradient = function() {
        const { svg } = NEUtils.getElements();
        if (!svg) return;
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        svg.appendChild(defs);
    };

    // Create a dynamic gradient for a specific connection
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

    // Start connecting from an output port
    NEConnections.startConnect = function(e, nodeId, portIndex) {
        e.stopPropagation();
        e.preventDefault();

        const { canvas, svg } = NEUtils.getElements();
        if (!canvas || !svg) return;
        canvas.classList.add('connecting');

        // Get starting port position in canvas local coordinates
        const portEl = document.querySelector(`#node-${nodeId} .port-out[data-port="${portIndex}"]`);
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        const portRect = portEl.getBoundingClientRect();

        const startPos = NEUtils.wrapperToCanvas(
            portRect.left - containerRect.left + portRect.width / 2,
            portRect.top - containerRect.top + portRect.height / 2
        );

        // Get port type and color
        const portType = portEl.dataset.portType;
        const portColor = getPortColor(portType);

        // Create temporary connection path (bezier curve)
        const tempPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        tempPath.id = 'temp-connection';
        tempPath.setAttribute('fill', 'none');
        tempPath.setAttribute('stroke', portColor);
        tempPath.setAttribute('stroke-width', '3');
        tempPath.setAttribute('stroke-linecap', 'round');
        tempPath.setAttribute('stroke-dasharray', '5,5');
        tempPath.style.opacity = '0.6';
        svg.appendChild(tempPath);

        NodeEditor.connecting = {
            from: nodeId,
            port: portIndex,
            startX: startPos.x,
            startY: startPos.y,
            portColor: portColor
        };

        document.onmousemove = updateTempConnection;
        document.onmouseup = endConnect;
    };

    // Start connecting from an input port (ComfyUI-style reconnection)
    NEConnections.startConnectFromInput = function(e, nodeId, portIndex) {
        e.stopPropagation();
        e.preventDefault();

        // Find existing connection TO this input port
        const existingConnection = NodeEditor.connections.find(c =>
            c.to === nodeId && c.toPort === portIndex
        );

        if (!existingConnection) {
            // No connection exists, do nothing
            return;
        }

        // Store the source node/port before removing connection
        const sourceNodeId = existingConnection.from;
        const sourcePortIndex = existingConnection.fromPort;

        // Store affected conjunction nodes before deleting
        const targetNode = NodeEditor.nodes.find(n => n.id === nodeId);
        const affectedConjunctions = targetNode && targetNode.type === 'conjunction' ? [targetNode] : [];

        // Remove the existing connection
        NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== existingConnection.id);
        const path = document.getElementById('conn-' + existingConnection.id);
        if (path) path.remove();

        // Remove the gradient for this connection
        const gradient = document.getElementById(`gradient-${existingConnection.id}`);
        if (gradient) gradient.remove();

        // Update affected conjunction nodes
        affectedConjunctions.forEach(conjNode => {
            if (typeof updateConjunctionNode === 'function') {
                updateConjunctionNode(conjNode.id);
            }
        });

        // Now start a new connection FROM the original source
        NEConnections.startConnect(e, sourceNodeId, sourcePortIndex);
    };

    function updateTempConnection(e) {
        if (!NodeEditor.connecting) return;

        const { canvas } = NEUtils.getElements();
        const tempPath = document.getElementById('temp-connection');
        if (!canvas || !tempPath) return;

        // Convert mouse position to canvas local coordinates
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        const mousePos = NEUtils.wrapperToCanvas(
            e.clientX - containerRect.left,
            e.clientY - containerRect.top
        );

        // Calculate bezier curve for smooth flow
        const pos1 = { x: NodeEditor.connecting.startX, y: NodeEditor.connecting.startY };
        const pos2 = { x: mousePos.x, y: mousePos.y };

        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Control point offset
        const offset = Math.max(80, Math.min(distance * 0.5, 200));

        const cp1x = pos1.x + offset;
        const cp1y = pos1.y;
        const cp2x = pos2.x - offset;
        const cp2y = pos2.y;

        // Create bezier curve path
        const pathData = `M ${pos1.x} ${pos1.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${pos2.x} ${pos2.y}`;
        tempPath.setAttribute('d', pathData);

        // Visual feedback for port compatibility
        const target = e.target;
        if (target.classList && target.classList.contains('port-in')) {
            const toNode = parseInt(target.dataset.node);
            const toPort = parseInt(target.dataset.port);

            // Get port types
            const fromNodeObj = NodeEditor.nodes.find(n => n.id === NodeEditor.connecting.from);
            const toNodeObj = NodeEditor.nodes.find(n => n.id === toNode);

            if (fromNodeObj && toNodeObj && toNode !== NodeEditor.connecting.from) {
                const fromDef = NODES[fromNodeObj.type];
                const toDef = NODES[toNodeObj.type];
                const fromPortType = fromDef.outputs[NodeEditor.connecting.port];
                const toPortType = toDef.inputs[toPort];

                // Check for circular dependency
                const wouldCreateCycle = NodeEditor.connections.some(c =>
                    c.from === toNode && c.to === NodeEditor.connecting.from
                );

                // Update visual feedback
                if (wouldCreateCycle) {
                    // Red for circular dependency
                    tempPath.setAttribute('stroke', '#ef4444');
                    tempPath.style.opacity = '0.6';
                } else if (arePortsCompatible(fromPortType, toPortType)) {
                    // Use source port color for compatible connections
                    tempPath.setAttribute('stroke', NodeEditor.connecting.portColor);
                    tempPath.style.opacity = '0.8';
                } else {
                    // Red for incompatible types
                    tempPath.setAttribute('stroke', '#ef4444');
                    tempPath.style.opacity = '0.6';
                }
            }
        } else {
            // Reset to source port color when not hovering over a port
            tempPath.setAttribute('stroke', NodeEditor.connecting.portColor);
            tempPath.style.opacity = '0.6';
        }
    }

    function endConnect(e) {
        if (!NodeEditor.connecting) return;

        const { canvas } = NEUtils.getElements();
        if (canvas) canvas.classList.remove('connecting');

        // Remove temporary path
        const tempPath = document.getElementById('temp-connection');
        if (tempPath) tempPath.remove();

        const target = e.target;
        if (target.classList && target.classList.contains('port-in')) {
            const toNode = parseInt(target.dataset.node);
            const toPort = parseInt(target.dataset.port);

            if (toNode !== NodeEditor.connecting.from) {
                NEConnections.addConnection(NodeEditor.connecting.from, NodeEditor.connecting.port, toNode, toPort);
            }
        }

        NodeEditor.connecting = null;
        document.onmousemove = null;
        document.onmouseup = null;
    }

    // Check if two port types are compatible
    function arePortsCompatible(fromPortType, toPortType) {
        // Images can only connect to images
        if (fromPortType === 'images' || toPortType === 'images') {
            return fromPortType === 'images' && toPortType === 'images';
        }

        // Text-based ports (text, prompt, captions) can connect to each other
        const textTypes = ['text', 'prompt', 'captions'];
        if (textTypes.includes(fromPortType) && textTypes.includes(toPortType)) {
            return true;
        }

        // Data port accepts anything (generic output)
        if (toPortType === 'data') {
            return true;
        }

        return false;
    }

    // Add connection
    NEConnections.addConnection = function(fromNode, fromPort, toNode, toPort) {
        // Check if exists
        const exists = NodeEditor.connections.some(c =>
            c.from === fromNode && c.fromPort === fromPort &&
            c.to === toNode && c.toPort === toPort
        );
        if (exists) return;

        // Get port types for validation
        const fromNodeObj = NodeEditor.nodes.find(n => n.id === fromNode);
        const toNodeObj = NodeEditor.nodes.find(n => n.id === toNode);

        if (!fromNodeObj || !toNodeObj) return;

        const fromDef = NODES[fromNodeObj.type];
        const toDef = NODES[toNodeObj.type];

        const fromPortType = fromDef.outputs[fromPort];
        const toPortType = toDef.inputs[toPort];

        // Validate port type compatibility
        if (!arePortsCompatible(fromPortType, toPortType)) {
            console.warn(`Incompatible connection: ${fromPortType} cannot connect to ${toPortType}`);
            return;
        }

        // Prevent circular dependencies: if target node already connects to source node, block it
        // This prevents: Conjunction → AI Model → back to same Conjunction
        const wouldCreateCycle = NodeEditor.connections.some(c =>
            c.from === toNode && c.to === fromNode
        );
        if (wouldCreateCycle) {
            console.warn(`Circular connection blocked: ${toNodeObj.type} (${toNode}) already connects to ${fromNodeObj.type} (${fromNode})`);
            return;
        }

        const conn = { id: NodeEditor.nextId++, from: fromNode, fromPort, to: toNode, toPort };
        NodeEditor.connections.push(conn);
        renderConnection(conn);

        // Update conjunction node if the target is a conjunction
        const targetNode = NodeEditor.nodes.find(n => n.id === toNode);
        if (targetNode && targetNode.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(toNode);
        }
    };

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

        path.onmouseenter = () => {
            path.setAttribute('stroke-width', '4');
            path.style.filter = 'drop-shadow(0 4px 12px rgba(99, 102, 241, 0.6))';
        };
        path.onmouseleave = () => {
            path.setAttribute('stroke-width', '3');
            path.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))';
        };
        path.onclick = () => NEConnections.deleteConnection(conn.id);

        svg.appendChild(path);
        NEConnections.updateConnectionLine(conn.id);
    }

    // Update connection line
    NEConnections.updateConnectionLine = function(connId) {
        const conn = NodeEditor.connections.find(c => c.id === connId);
        if (!conn) return;

        const path = document.getElementById('conn-' + connId);
        if (!path) return;

        // Find ports using data attributes
        const fromEl = document.querySelector(`#node-${conn.from} .port-out[data-port="${conn.fromPort}"]`);
        const toEl = document.querySelector(`#node-${conn.to} .port-in[data-port="${conn.toPort}"]`);
        if (!fromEl || !toEl) return;

        // Get the correct wrapper (normal mode or fullscreen mode)
        const { canvas } = NEUtils.getElements();
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        const fromRect = fromEl.getBoundingClientRect();
        const toRect = toEl.getBoundingClientRect();

        // Calculate positions relative to container, then convert to canvas local coordinates
        const pos1 = NEUtils.wrapperToCanvas(
            fromRect.left - containerRect.left + fromRect.width / 2,
            fromRect.top - containerRect.top + fromRect.height / 2
        );
        const pos2 = NEUtils.wrapperToCanvas(
            toRect.left - containerRect.left + toRect.width / 2,
            toRect.top - containerRect.top + toRect.height / 2
        );

        // Calculate bezier curve control points for smooth horizontal flow
        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Control point offset based on distance (minimum 80px, scales with distance)
        const offset = Math.max(80, Math.min(distance * 0.5, 200));

        const cp1x = pos1.x + offset;
        const cp1y = pos1.y;
        const cp2x = pos2.x - offset;
        const cp2y = pos2.y;

        // Create smooth cubic bezier curve path
        const pathData = `M ${pos1.x} ${pos1.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${pos2.x} ${pos2.y}`;
        path.setAttribute('d', pathData);

        // Get port types and colors
        const fromPortType = fromEl.dataset.portType;
        const toPortType = toEl.dataset.portType;
        const fromColor = getPortColor(fromPortType);
        const toColor = getPortColor(toPortType);

        // Create/update gradient with port colors
        const gradientUrl = createConnectionGradient(connId, fromColor, toColor, pos1.x, pos1.y, pos2.x, pos2.y);
        if (gradientUrl) {
            path.setAttribute('stroke', gradientUrl);
        }
    };

    // Update all connections
    NEConnections.updateConnections = function() {
        NodeEditor.connections.forEach(c => NEConnections.updateConnectionLine(c.id));
    };

    // Delete connection
    NEConnections.deleteConnection = function(connId) {
        // Find the connection before deleting to check if it was connected to a conjunction
        const conn = NodeEditor.connections.find(c => c.id === connId);
        const targetNode = conn ? NodeEditor.nodes.find(n => n.id === conn.to) : null;

        NodeEditor.connections = NodeEditor.connections.filter(c => c.id !== connId);
        const path = document.getElementById('conn-' + connId);
        if (path) path.remove();

        // Remove the gradient for this connection
        const gradient = document.getElementById(`gradient-${connId}`);
        if (gradient) gradient.remove();

        // Update conjunction node if the deleted connection was connected to one
        if (targetNode && targetNode.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(targetNode.id);
        }
    };

    // Export namespace and selected aliases for compatibility
    window.NEConnections = NEConnections;
})();

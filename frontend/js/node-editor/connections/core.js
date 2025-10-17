// Connections: creation, rendering, updates, and deletion
(function() {
    const NEConnections = {};

    // Create gradient definition for connections
    NEConnections.createConnectionGradient = function() {
        const { svg } = NEUtils.getElements();
        if (!svg) return;
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
    };

    // Start connecting from an output port
    NEConnections.startConnect = function(e, nodeId, portIndex) {
        e.stopPropagation();
        e.preventDefault();

        const { canvas, svg } = NEUtils.getElements();
        if (!canvas || !svg) return;
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

        const startPos = NEUtils.wrapperToCanvas(
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
    };

    function updateTempConnection(e) {
        if (!NodeEditor.connecting) return;

        const { canvas } = NEUtils.getElements();
        const tempLine = document.getElementById('temp-connection');
        if (!canvas || !tempLine) return;

        // Convert mouse position to canvas local coordinates
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        const mousePos = NEUtils.wrapperToCanvas(
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

        const { canvas } = NEUtils.getElements();
        if (canvas) canvas.classList.remove('connecting');

        // Remove temporary line
        const tempLine = document.getElementById('temp-connection');
        if (tempLine) tempLine.remove();

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

    // Add connection
    NEConnections.addConnection = function(fromNode, fromPort, toNode, toPort) {
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
        if (targetNode && targetNode.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(toNode);
        }
    };

    function renderConnection(conn) {
        const { svg } = NEUtils.getElements();
        if (!svg) return;
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
        line.onclick = () => NEConnections.deleteConnection(conn.id);

        svg.appendChild(line);
        NEConnections.updateConnectionLine(conn.id);
    }

    // Update connection line
    NEConnections.updateConnectionLine = function(connId) {
        const conn = NodeEditor.connections.find(c => c.id === connId);
        if (!conn) return;

        const line = document.getElementById('conn-' + connId);
        if (!line) return;

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

        line.setAttribute('x1', pos1.x);
        line.setAttribute('y1', pos1.y);
        line.setAttribute('x2', pos2.x);
        line.setAttribute('y2', pos2.y);
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
        const line = document.getElementById('conn-' + connId);
        if (line) line.remove();

        // Update conjunction node if the deleted connection was connected to one
        if (targetNode && targetNode.type === 'conjunction' && typeof updateConjunctionNode === 'function') {
            updateConjunctionNode(targetNode.id);
        }
    };

    // Export namespace and selected aliases for compatibility
    window.NEConnections = NEConnections;
})();

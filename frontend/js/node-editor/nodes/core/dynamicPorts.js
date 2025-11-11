// Dynamic ports API split out from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

    // Add output port to a node (for nodes with allowDynamicOutputs)
    NENodes.addOutputPort = function(nodeId, portConfig) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return;

        const nodeDef = NODES[node.type];
        if (!nodeDef || !nodeDef.allowDynamicOutputs) {
            console.warn('Node type does not support dynamic outputs');
            return;
        }

        // Initialize ports array if it doesn't exist
        if (!node.data.ports) {
            node.data.ports = [];
        }

        // Generate port ID if not provided
        const portId = portConfig.id || `port_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const newPort = {
            id: portId,
            label: portConfig.label || `Port ${node.data.ports.length + 1}`,
            instruction: portConfig.instruction || '',
            isDefault: portConfig.isDefault || false
        };

        // If this is set as default, unset other defaults
        if (newPort.isDefault) {
            node.data.ports.forEach(p => p.isDefault = false);
        }

        node.data.ports.push(newPort);

        // Re-render ports section
        NENodes.updateNodePorts(nodeId);

        // Update connections
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

        return portId;
    };

    // Remove output port from a node
    NENodes.removeOutputPort = function(nodeId, portId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || !node.data.ports) return;

        const portIndex = node.data.ports.findIndex(p => p.id === portId);
        if (portIndex === -1) return;

        // Remove any connections to this port
        if (typeof NEConnections !== 'undefined') {
            const connectionsToRemove = NodeEditor.connections.filter(
                c => c.from === nodeId && c.fromPort === portIndex
            );
            connectionsToRemove.forEach(c => NEConnections.removeConnection(c.id));
        }

        // Remove port
        node.data.ports.splice(portIndex, 1);

        // Update port indices in remaining connections
        if (typeof NEConnections !== 'undefined') {
            NodeEditor.connections.forEach(c => {
                if (c.from === nodeId && c.fromPort > portIndex) {
                    c.fromPort--;
                }
            });
        }

        // Re-render ports section
        NENodes.updateNodePorts(nodeId);

        // Update connections
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Update node's ports section (re-render ports without re-rendering entire node)
    NENodes.updateNodePorts = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return;

        const nodeEl = document.getElementById(`node-${nodeId}`);
        if (!nodeEl) return;

        const portsSection = nodeEl.querySelector('.node-ports-section');
        if (!portsSection) return;

        const nodeDef = NODES[node.type];

        // Clear and rebuild ports section
        portsSection.innerHTML = '';

        // Input ports
        const inputsContainer = document.createElement('div');
        inputsContainer.className = 'node-ports-in';
        nodeDef.inputs.forEach((portName, i) => {
            inputsContainer.appendChild(NENodes.createPort(node, portName, i, false));
        });
        portsSection.appendChild(inputsContainer);

        // Output ports - handle dynamic outputs
        const outputsContainer = document.createElement('div');
        outputsContainer.className = 'node-ports-out';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            // Use dynamic ports from node data
            node.data.ports.forEach((portConfig, i) => {
                const portName = portConfig.label || `Port ${i + 1}`;
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        } else {
            // Use static ports from definition
            nodeDef.outputs.forEach((portName, i) => {
                outputsContainer.appendChild(NENodes.createPort(node, portName, i, true));
            });
        }

        portsSection.appendChild(outputsContainer);
    };

    // Get output port name for a node (handles both static and dynamic ports)
    NENodes.getOutputPortName = function(node, portIndex) {
        const nodeDef = NODES[node.type];
        if (!nodeDef) return '';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            const port = node.data.ports[portIndex];
            return port ? port.label : '';
        }

        return nodeDef.outputs[portIndex] || '';
    };

    try {
        window.addOutputPort = NENodes.addOutputPort;
        window.removeOutputPort = NENodes.removeOutputPort;
    } catch (e) {
        // ignore
    }

})();

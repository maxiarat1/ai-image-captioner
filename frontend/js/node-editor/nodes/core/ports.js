// Port creation and type helpers split out from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

    // Create port element
    NENodes.createPort = function(node, portName, portIndex, isOutput) {
        const portWrapper = document.createElement('div');
        portWrapper.className = 'port-wrapper';

        const port = document.createElement('div');
        port.className = isOutput ? 'port port-out' : 'port port-in';
        port.dataset.node = node.id;
        port.dataset.port = portIndex;

        // For dynamic output ports (curate node), get the proper port type
        // Normalize port type by stripping "(optional)" suffix for styling purposes
        const normalizePortType = (type) => type ? type.replace(/\s*\(optional\)\s*$/i, '').trim() : '';

        if (isOutput && typeof NENodes.getOutputPortType === 'function') {
            port.dataset.portType = normalizePortType(NENodes.getOutputPortType(node, portIndex));
        } else {
            port.dataset.portType = normalizePortType(portName);
        }

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

    // Get output port type for connections (returns 'route' for curate nodes)
    NENodes.getOutputPortType = function(node, portIndex) {
        if (node.type === 'curate') {
            return 'route';
        }

        const nodeDef = NODES[node.type];
        if (!nodeDef) return '';

        if (nodeDef.allowDynamicOutputs && node.data.ports) {
            return 'route';  // All dynamic outputs are route type
        }

        return nodeDef.outputs[portIndex] || '';
    };

    try {
        window.createPort = NENodes.createPort;
        window.getOutputPortType = NENodes.getOutputPortType;
    } catch (e) {
        // ignore
    }

})();

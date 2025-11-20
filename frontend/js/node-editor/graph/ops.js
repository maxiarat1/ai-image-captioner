// Graph operations: delete node and clear graph
(function() {
    const NEGraphOps = {};

    NEGraphOps.deleteNode = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node) return;

        // Remove associated connections first
        const toRemove = NodeEditor.connections.filter(c => c.from === nodeId || c.to === nodeId);
        toRemove.forEach(c => {
            if (typeof NEConnections !== 'undefined' && typeof NEConnections.deleteConnection === 'function') {
                NEConnections.deleteConnection(c.id);
            } else {
                // Fallback: remove line and update connections list
                const line = document.getElementById('conn-' + c.id);
                if (line) line.remove();
                NodeEditor.connections = NodeEditor.connections.filter(cc => cc.id !== c.id);
            }
        });

        // Remove node element
        const el = document.getElementById('node-' + nodeId);
        if (el) el.remove();

        // Remove from state
        NodeEditor.nodes = NodeEditor.nodes.filter(n => n.id !== nodeId);

        // Update visuals
        if (typeof NEConnections !== 'undefined' && typeof NEConnections.updateConnections === 'function') {
            NEConnections.updateConnections();
        }
        if (typeof NEMinimap !== 'undefined' && typeof NEMinimap.updateMinimap === 'function') {
            NEMinimap.updateMinimap();
        } else if (typeof updateMinimap === 'function') {
            updateMinimap();
        }

        // Schedule auto-save
        if (typeof NEPersistence !== 'undefined') NEPersistence.scheduleSave();
    };

    NEGraphOps.clearGraph = function() {
        if (!confirm('Clear all?')) return;

        // Delete all connections using the connection module to keep state consistent
        const allConns = [...NodeEditor.connections];
        allConns.forEach(c => {
            if (typeof NEConnections !== 'undefined' && typeof NEConnections.deleteConnection === 'function') {
                NEConnections.deleteConnection(c.id);
            } else {
                const line = document.getElementById('conn-' + c.id);
                if (line) line.remove();
            }
        });

        // Remove nodes from DOM
        NodeEditor.nodes.forEach(n => {
            const el = document.getElementById('node-' + n.id);
            if (el) el.remove();
        });

        // Reset state
        NodeEditor.connections = [];
        NodeEditor.nodes = [];

        // Update visuals
        if (typeof NEConnections !== 'undefined' && typeof NEConnections.updateConnections === 'function') {
            NEConnections.updateConnections();
        }
        if (typeof NEMinimap !== 'undefined' && typeof NEMinimap.updateMinimap === 'function') {
            NEMinimap.updateMinimap();
        } else if (typeof updateMinimap === 'function') {
            updateMinimap();
        }

        // Clear saved workflow (save empty state)
        if (typeof NEPersistence !== 'undefined') NEPersistence.scheduleSave();
    };

    // Expose
    window.NEGraphOps = NEGraphOps;
    // Backward-compat globals
    window.deleteNode = NEGraphOps.deleteNode;
    window.clearGraph = NEGraphOps.clearGraph;
})();

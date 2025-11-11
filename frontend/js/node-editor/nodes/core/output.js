// Output node helpers split out from core.js
(function(){
    const NENodes = (window.NENodes = window.NENodes || {});

    // Update output node statistics
    NENodes.updateOutputStats = function(nodeId, stats) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'output') return;

        // Update node data
        node.data.stats = { ...node.data.stats, ...stats };

        // Update the display
        const statsContainer = document.getElementById(`output-stats-${nodeId}`);
        if (statsContainer) {
            const body = document.getElementById(`node-${nodeId}`).querySelector('.node-body');
            if (body) {
                body.innerHTML = NENodes.getNodeContent(node);
            }
        }
    };

    // Reset output node statistics
    NENodes.resetOutputStats = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'output') return;

        node.data.stats = {};

        const body = document.getElementById(`node-${nodeId}`).querySelector('.node-body');
        if (body) {
            body.innerHTML = NENodes.getNodeContent(node);
        }
    };

    try {
        window.updateOutputStats = NENodes.updateOutputStats;
        window.resetOutputStats = NENodes.resetOutputStats;
    } catch (e) {
        // ignore
    }

})();

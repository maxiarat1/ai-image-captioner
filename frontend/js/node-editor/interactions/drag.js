// Node dragging interactions
(function() {
    const NEDrag = {};

    NEDrag.startDrag = function(e, node) {
        if (e.target.classList.contains('node-del')) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);
        const el = document.getElementById('node-' + node.id);
        if (el) el.classList.add('dragging');

        NodeEditor.dragging = {
            node: node,
            offsetX: canvasPos.x - node.x,
            offsetY: canvasPos.y - node.y
        };

        document.onmousemove = drag;
        document.onmouseup = stopDrag;
    };

    function drag(e) {
        if (!NodeEditor.dragging) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);
        const node = NodeEditor.dragging.node;
        node.x = canvasPos.x - NodeEditor.dragging.offsetX;
        node.y = canvasPos.y - NodeEditor.dragging.offsetY;

        const el = document.getElementById('node-' + node.id);
        if (el) {
            el.style.left = node.x + 'px';
            el.style.top = node.y + 'px';
        }

    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    }

    function stopDrag() {
        if (NodeEditor.dragging) {
            const el = document.getElementById('node-' + NodeEditor.dragging.node.id);
            if (el) el.classList.remove('dragging');
        }

        NodeEditor.dragging = null;
        document.onmousemove = null;
        document.onmouseup = null;
    }

    window.NEDrag = NEDrag;
})();

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

        // Prepare guides container
        if (NodeEditor.settings.showGuides) ensureGuides();

        document.onmousemove = (ev) => drag(ev);
        document.onmouseup = stopDrag;
    };

    function drag(e) {
        if (!NodeEditor.dragging) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);
        const node = NodeEditor.dragging.node;
        let newX = canvasPos.x - NodeEditor.dragging.offsetX;
        let newY = canvasPos.y - NodeEditor.dragging.offsetY;

        const GRID = NodeEditor.settings.gridSize || 20;
        const EPS = NodeEditor.settings.snapEpsilon ?? 4;
        const mode = NodeEditor.settings.snapMode || 'always';

        const wantSnap = (
            (mode === 'always') ||
            (mode === 'withShift' && e.shiftKey) ||
            (mode === 'disableWithAlt' && !e.altKey)
        ) && mode !== 'off';

        if (wantSnap) {
            const rx = Math.round(newX / GRID) * GRID;
            const ry = Math.round(newY / GRID) * GRID;
            if (Math.abs(rx - newX) <= EPS) newX = rx;
            if (Math.abs(ry - newY) <= EPS) newY = ry;
        }

        // Clamp within canvas bounds (account for node size if available)
        const elDom = document.getElementById('node-' + node.id);
        const maxX = NECanvas.SIZE - (elDom ? elDom.offsetWidth : 0);
        const maxY = NECanvas.SIZE - (elDom ? elDom.offsetHeight : 0);
        newX = Math.max(0, Math.min(maxX, newX));
        newY = Math.max(0, Math.min(maxY, newY));

        node.x = newX;
        node.y = newY;

        const el = document.getElementById('node-' + node.id);
        if (el) {
            el.style.left = node.x + 'px';
            el.style.top = node.y + 'px';
        }

    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

    // Update guides if enabled
    if (NodeEditor.settings.showGuides) updateGuides(node.x, node.y);
    }

    function stopDrag() {
        if (NodeEditor.dragging) {
            const dragged = NodeEditor.dragging.node;
            const GRID = NodeEditor.settings.gridSize || 20;
            const mode = NodeEditor.settings.snapMode || 'always';
            // Always do final snap unless snap mode is fully off
            if (mode !== 'off') {
                dragged.x = Math.round(dragged.x / GRID) * GRID;
                dragged.y = Math.round(dragged.y / GRID) * GRID;
            }

            // Final clamp to keep node entirely within canvas
            const elDom = document.getElementById('node-' + dragged.id);
            const maxX = NECanvas.SIZE - (elDom ? elDom.offsetWidth : 0);
            const maxY = NECanvas.SIZE - (elDom ? elDom.offsetHeight : 0);
            dragged.x = Math.max(0, Math.min(maxX, dragged.x));
            dragged.y = Math.max(0, Math.min(maxY, dragged.y));

            const el = document.getElementById('node-' + dragged.id);
            if (el) {
                el.classList.remove('dragging');
                el.style.left = dragged.x + 'px';
                el.style.top = dragged.y + 'px';
            }

            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

            removeGuides();

            // Schedule auto-save after drag completes
            if (typeof NEPersistence !== 'undefined') NEPersistence.scheduleSave();
        }

        NodeEditor.dragging = null;
        document.onmousemove = null;
        document.onmouseup = null;
    }

    // --- Guides helpers ---
    function ensureGuides() {
        const { canvas } = NEUtils.getElements();
        if (!canvas) return;
        if (document.getElementById('dragGuides')) return;
        const div = document.createElement('div');
        div.id = 'dragGuides';
        div.className = 'drag-guides';
        canvas.appendChild(div);
        // create two lines
        const h = document.createElement('div');
        h.className = 'drag-guide-line h';
        h.id = 'dragGuideH';
        const v = document.createElement('div');
        v.className = 'drag-guide-line v';
        v.id = 'dragGuideV';
        div.appendChild(h);
        div.appendChild(v);
    }

    function updateGuides(x, y) {
        const div = document.getElementById('dragGuides');
        if (!div) return;
        const h = document.getElementById('dragGuideH');
        const v = document.getElementById('dragGuideV');
        if (!h || !v) return;
        h.style.top = y + 'px';
        h.style.left = '0px';
        v.style.left = x + 'px';
        v.style.top = '0px';
    }

    function removeGuides() {
        const div = document.getElementById('dragGuides');
        if (div && div.parentElement) div.parentElement.removeChild(div);
    }

    window.NEDrag = NEDrag;
})();

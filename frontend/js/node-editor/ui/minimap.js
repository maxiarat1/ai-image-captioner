// Minimap rendering and updates for the Node Editor
(function() {
    const NEMinimap = {};

    // Create minimap container and initial content
    NEMinimap.createMinimap = function() {
        const { wrapper } = NEUtils.getElements();
        if (!wrapper) return;

        const minimap = document.createElement('div');
        minimap.className = 'node-minimap';
        minimap.id = 'nodeMinimap';

        const minimapCanvas = document.createElement('div');
        minimapCanvas.className = 'minimap-canvas';
        minimapCanvas.id = 'minimapCanvas';

        const viewport = document.createElement('div');
        viewport.className = 'minimap-viewport';
        viewport.id = 'minimapViewport';

        minimapCanvas.appendChild(viewport);
        minimap.appendChild(minimapCanvas);
        wrapper.appendChild(minimap);

        // --- Interaction: click/drag to move camera ---
        const state = { dragging: false };
        const scale = 200 / 5000; // minimap pixels per canvas unit

        function setCameraFromMinimap(mx, my, rectEl) {
            // mx,my are relative to minimapCanvas top-left in screen pixels
            const { wrapper } = NEUtils.getElements();
            if (!wrapper) return;

            const rect = wrapper.getBoundingClientRect();
            const s = NodeEditor.transform.scale || 1;
            const viewW = rect.width / s;
            const viewH = rect.height / s;

            // Account for CSS transform scaling on minimapCanvas (0.6 -> 1.0)
            const r = rectEl || minimapCanvas.getBoundingClientRect();
            const hoverScale = r.width / 200; // 200 is base minimap size

            // Convert to base minimap coordinates (0..200)
            const mxBase = mx / hoverScale;
            const myBase = my / hoverScale;

            // Target viewport top-left in canvas space centered on the cursor
            let vx = mxBase / scale - viewW / 2;
            let vy = myBase / scale - viewH / 2;

            // Clamp viewport within canvas bounds
            const maxVX = NECanvas.SIZE - viewW;
            const maxVY = NECanvas.SIZE - viewH;
            vx = Math.max(0, Math.min(maxVX, vx));
            vy = Math.max(0, Math.min(maxVY, vy));

            // Convert to transform (screen space)
            NodeEditor.transform.x = -vx * s;
            NodeEditor.transform.y = -vy * s;

            if (typeof NEViewport !== 'undefined' && typeof NEViewport.clampTransform === 'function') {
                NEViewport.clampTransform();
            }
            NEUtils.applyTransform();
            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            NEMinimap.updateMinimap();
        }

        minimapCanvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const r = minimapCanvas.getBoundingClientRect();
            const mx = e.clientX - r.left;
            const my = e.clientY - r.top;
            state.dragging = true;
            setCameraFromMinimap(mx, my, r);
        });

        document.addEventListener('mousemove', (e) => {
            if (!state.dragging) return;
            const r = minimapCanvas.getBoundingClientRect();
            const mx = Math.max(0, Math.min(r.width, e.clientX - r.left));
            const my = Math.max(0, Math.min(r.height, e.clientY - r.top));
            setCameraFromMinimap(mx, my, r);
        });

        document.addEventListener('mouseup', () => {
            state.dragging = false;
        });

        NEMinimap.updateMinimap();
    };

    // Update nodes and viewport rectangle in the minimap
    NEMinimap.updateMinimap = function() {
        const minimapCanvas = document.getElementById('minimapCanvas');
        const viewport = document.getElementById('minimapViewport');
        if (!minimapCanvas || !viewport) return;

        // Canvas is 5000x5000, minimap is 200x200, so scale is 0.04
        const scale = 200 / 5000;

        // Clear existing nodes
        minimapCanvas.querySelectorAll('.minimap-node').forEach(n => n.remove());

        // Draw nodes
        NodeEditor.nodes.forEach(node => {
            const dot = document.createElement('div');
            dot.className = 'minimap-node';
            dot.style.left = (node.x * scale) + 'px';
            dot.style.top = (node.y * scale) + 'px';
            dot.style.width = '6px';
            dot.style.height = '6px';
            minimapCanvas.appendChild(dot);
        });

        // Update viewport indicator
        const { canvas } = NEUtils.getElements();
        const wrapper = canvas ? canvas.parentElement : null;
        if (!wrapper) return;

        const rect = wrapper.getBoundingClientRect();
        const viewportWidth = rect.width / NodeEditor.transform.scale;
        const viewportHeight = rect.height / NodeEditor.transform.scale;
        const viewportX = -NodeEditor.transform.x / NodeEditor.transform.scale;
        const viewportY = -NodeEditor.transform.y / NodeEditor.transform.scale;

        viewport.style.left = (viewportX * scale) + 'px';
        viewport.style.top = (viewportY * scale) + 'px';
        viewport.style.width = (viewportWidth * scale) + 'px';
        viewport.style.height = (viewportHeight * scale) + 'px';
    };

    // Export namespace and global aliases for compatibility
    window.NEMinimap = NEMinimap;
    // Aliases removed after migration; use NEMinimap.* directly
})();

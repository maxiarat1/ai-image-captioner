// Viewport (panning and zoom) logic for the Node Editor
(function() {
    const NEViewport = {};

    NEViewport.initCanvasPanning = function() {
    const { wrapper, canvas } = NEUtils.getElements();
        if (!wrapper || !canvas) return;

        // Center the canvas initially (canvas is 5000x5000, center at 2500, 2500)
        const rect = wrapper.getBoundingClientRect();
        NodeEditor.transform.x = rect.width / 2 - 2500;
        NodeEditor.transform.y = rect.height / 2 - 2500;
        NEUtils.applyTransform();

        // Mouse down - start panning on canvas background only
        canvas.addEventListener('mousedown', (e) => {
            // Only pan on left click on canvas or SVG (not on nodes)
            const isNode = e.target.closest('.node');
            if (e.button === 0 && !isNode) {
                e.preventDefault();
                NodeEditor.panning = {
                    startX: e.clientX - NodeEditor.transform.x,
                    startY: e.clientY - NodeEditor.transform.y
                };
                wrapper.classList.add('panning');
            }
        });

        // Mouse move - do panning
        document.addEventListener('mousemove', (e) => {
            if (NodeEditor.panning) {
                NodeEditor.transform.x = e.clientX - NodeEditor.panning.startX;
                NodeEditor.transform.y = e.clientY - NodeEditor.panning.startY;
                NEUtils.applyTransform();
                if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
            }
        });

        // Mouse up - stop panning
        document.addEventListener('mouseup', () => {
            if (NodeEditor.panning) {
                wrapper.classList.remove('panning');
                NodeEditor.panning = null;
            }
        });

        // Mouse wheel - zoom
        wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            NEViewport.handleZoom(e);
        });
    };

    NEViewport.handleZoom = function(e) {
    const { wrapper } = NEUtils.getElements();
        const rect = wrapper.getBoundingClientRect();

        // Mouse position relative to wrapper
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Current mouse position in canvas space (before zoom)
        const canvasX = (mouseX - NodeEditor.transform.x) / NodeEditor.transform.scale;
        const canvasY = (mouseY - NodeEditor.transform.y) / NodeEditor.transform.scale;

        // Calculate zoom delta
        const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
        let newScale = NodeEditor.transform.scale * zoomDelta;

        // Limit zoom range
        newScale = Math.max(0.1, Math.min(3, newScale));

        // Adjust transform to zoom toward cursor
        NodeEditor.transform.scale = newScale;
        NodeEditor.transform.x = mouseX - canvasX * newScale;
        NodeEditor.transform.y = mouseY - canvasY * newScale;

    NEUtils.applyTransform();
    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Export namespace
    window.NEViewport = NEViewport;
    // Temporary global alias to preserve references until fullscreen is refactored
    // Removed temporary global alias: handleZoom
})();

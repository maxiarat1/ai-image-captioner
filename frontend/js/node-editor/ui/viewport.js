// Viewport (panning and zoom) logic for the Node Editor
(function() {
    const NEViewport = {};

    /**
     * Centers the camera on world origin (0,0).
     * World origin is at canvas center (2500, 2500).
     * @param {HTMLElement} container - The container element (wrapper or fullscreen)
     */
    NEViewport.centerCanvas = function(container) {
        if (!container) return;
        const rect = container.getBoundingClientRect();
        // Center viewport on canvas center (world origin)
        const s = NodeEditor.transform.scale || 1;
        NodeEditor.transform.x = rect.width / 2 - NECanvas.CENTER * s;
        NodeEditor.transform.y = rect.height / 2 - NECanvas.CENTER * s;
        // Ensure we don't center outside bounds on small zooms/sizes
        if (typeof NEViewport.clampTransform === 'function') NEViewport.clampTransform();
        NEUtils.applyTransform();
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Keep camera translation within canvas borders
    NEViewport.clampTransform = function() {
        const { wrapper } = NEUtils.getElements();
        if (!wrapper) return;
        const rect = wrapper.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const s = NodeEditor.transform.scale;
        const canvasW = NECanvas.SIZE * s;
        const canvasH = NECanvas.SIZE * s;

        // If canvas is smaller than viewport, keep it centered (no panning)
        if (canvasW <= w) {
            NodeEditor.transform.x = (w - canvasW) / 2;
        } else {
            const minX = w - canvasW; // when right edge aligns
            const maxX = 0;           // when left edge aligns
            NodeEditor.transform.x = Math.min(maxX, Math.max(minX, NodeEditor.transform.x));
        }

        if (canvasH <= h) {
            NodeEditor.transform.y = (h - canvasH) / 2;
        } else {
            const minY = h - canvasH; // when bottom edge aligns
            const maxY = 0;           // when top edge aligns
            NodeEditor.transform.y = Math.min(maxY, Math.max(minY, NodeEditor.transform.y));
        }
    };

    /**
     * Initialize canvas panning and interaction handlers
     */
    NEViewport.initCanvasPanning = function() {
        const { wrapper, canvas } = NEUtils.getElements();
        if (!wrapper || !canvas) return;

    // Zoom out to give more viewing distance, then center with current scale
    NodeEditor.transform.scale = 0.7;
    NEViewport.centerCanvas(wrapper);

        // Mouse down - start panning on canvas background only
        canvas.addEventListener('mousedown', (e) => {
            // Only pan on left click on canvas or SVG (not on nodes)
            if (e.button === 0 && !e.target.closest('.node')) {
                e.preventDefault();
                const { transform } = NodeEditor;
                NodeEditor.panning = {
                    startX: e.clientX - transform.x,
                    startY: e.clientY - transform.y
                };
                wrapper.classList.add('panning');
            }
        });

        // Mouse move - handle panning
        document.addEventListener('mousemove', (e) => {
            if (!NodeEditor.panning) return;
            
            const { transform } = NodeEditor;
            transform.x = e.clientX - NodeEditor.panning.startX;
            transform.y = e.clientY - NodeEditor.panning.startY;
            // Keep camera within canvas borders
            NEViewport.clampTransform();
            NEUtils.applyTransform();
            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
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

    /**
     * Handle zoom with mouse wheel
     * Zooms toward cursor position
     * @param {WheelEvent} e - The wheel event
     */
    NEViewport.handleZoom = function(e) {
        const { wrapper } = NEUtils.getElements();
        const rect = wrapper.getBoundingClientRect();
        const { transform } = NodeEditor;

        // Mouse position relative to wrapper
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Current mouse position in canvas space (before zoom)
        const canvasX = (mouseX - transform.x) / transform.scale;
        const canvasY = (mouseY - transform.y) / transform.scale;

        // Calculate new scale with limits
        const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(3, transform.scale * zoomDelta));

        // Only update if scale actually changed
        if (newScale === transform.scale) return;

        // Adjust transform to zoom toward cursor
        transform.scale = newScale;
        transform.x = mouseX - canvasX * newScale;
        transform.y = mouseY - canvasY * newScale;

    // Clamp after zoom to avoid exposing outside of canvas
    NEViewport.clampTransform();

        // Apply updates
        NEUtils.applyTransform();
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    // Export namespace
    window.NEViewport = NEViewport;
    // Temporary global alias to preserve references until fullscreen is refactored
    // Removed temporary global alias: handleZoom
})();

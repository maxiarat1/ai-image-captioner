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
        NodeEditor.transform.x = rect.width / 2 - NECanvas.CENTER;
        NodeEditor.transform.y = rect.height / 2 - NECanvas.CENTER;
        NEUtils.applyTransform();
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Initialize canvas panning and interaction handlers
     */
    NEViewport.initCanvasPanning = function() {
        const { wrapper, canvas } = NEUtils.getElements();
        if (!wrapper || !canvas) return;

        // Initial centering
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
            NEUtils.applyTransform();
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

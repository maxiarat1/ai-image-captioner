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

        // Mouse down - start panning or selection box on canvas background
        canvas.addEventListener('mousedown', (e) => {
            // Only handle clicks on canvas background (not on nodes)
            if (!e.target.closest('.node')) {
                e.preventDefault();

                // Ctrl/Cmd + left-click = selection box
                // Left-click or middle mouse = pan
                if (e.button === 0 && (e.ctrlKey || e.metaKey)) {
                    // Start selection box
                    if (typeof NESelection !== 'undefined') {
                        NESelection.startSelectionBox(e);
                        wrapper.classList.add('selecting');
                    }
                } else if (e.button === 0 || e.button === 1) {
                    // Clear selection when clicking on empty canvas
                    if (typeof NESelection !== 'undefined') {
                        NESelection.clearSelection();
                    }
                    // Start panning
                    const { transform } = NodeEditor;
                    NodeEditor.panning = {
                        startX: e.clientX - transform.x,
                        startY: e.clientY - transform.y
                    };
                    wrapper.classList.add('panning');
                }
            }
        });

        // Mouse move - handle panning and selection box
        document.addEventListener('mousemove', (e) => {
            // Handle panning
            if (NodeEditor.panning) {
                const { transform } = NodeEditor;
                transform.x = e.clientX - NodeEditor.panning.startX;
                transform.y = e.clientY - NodeEditor.panning.startY;
                // Keep camera within canvas borders
                NEViewport.clampTransform();
                NEUtils.applyTransform();
                if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
                return;
            }

            // Handle selection box
            if (typeof NESelection !== 'undefined' && NESelection.isSelecting()) {
                NESelection.updateSelectionBox(e);
                return;
            }

            // Handle multi-drag
            if (typeof NESelection !== 'undefined' && NESelection.isMultiDragging()) {
                NESelection.updateMultiDrag(e);
            }
        });

        // Mouse up - stop panning, selection box, or multi-drag
        document.addEventListener('mouseup', (e) => {
            // End panning
            if (NodeEditor.panning) {
                wrapper.classList.remove('panning');
                NodeEditor.panning = null;
            }

            // End selection box
            if (typeof NESelection !== 'undefined' && NESelection.isSelecting()) {
                NESelection.endSelectionBox(e);
                wrapper.classList.remove('selecting');
            }

            // End multi-drag
            if (typeof NESelection !== 'undefined' && NESelection.isMultiDragging()) {
                NESelection.endMultiDrag();
                wrapper.classList.remove('multi-dragging');
            }
        });

        // Mouse wheel - zoom
        wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            NEViewport.handleZoom(e);
        }, { passive: false });
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

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

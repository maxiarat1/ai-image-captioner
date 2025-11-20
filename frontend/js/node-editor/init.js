// Node editor initialization and default graph scaffolding

function initNodeEditor() {
    // Initialize grid size CSS variable
    const canvas = document.getElementById('nodeCanvas');
    if (canvas && NodeEditor.settings) {
        canvas.style.setProperty('--grid-size', `${NodeEditor.settings.gridSize}px`);
    }

    // Core UI setup
    NEToolbar.setupToolbar();
    if (typeof NEWorkflowControls !== 'undefined') NEWorkflowControls.setup();
    NEViewport.initCanvasPanning();
    NEConnections.createConnectionGradient();
    NEMinimap.createMinimap();

    // Button handlers
    const executeBtn = document.getElementById('executeGraphBtn');
    if (executeBtn) executeBtn.onclick = executeGraph;

    const clearBtn = document.getElementById('clearGraphBtn');
    if (clearBtn) clearBtn.onclick = clearGraph;

    const fullscreenBtn = document.getElementById('fullscreenBtn');
    if (fullscreenBtn) fullscreenBtn.onclick = NEFullscreen.openFullscreen;

    const closeFullscreenBtn = document.getElementById('closeNodeFullscreen');
    if (closeFullscreenBtn) closeFullscreenBtn.onclick = NEFullscreen.closeFullscreen;

    const fullscreenBackdrop = document.getElementById('nodeFullscreenBackdrop');
    if (fullscreenBackdrop) {
        fullscreenBackdrop.onclick = (e) => {
            if (e.target.id === 'nodeFullscreenBackdrop') {
                NEFullscreen.closeFullscreen();
            }
        };
    }

    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('nodeFullscreenModal');
        if (modal && e.key === 'Escape' && modal.classList.contains('active')) {
            NEFullscreen.closeFullscreen();
        }
    });

    // Context menu for grid settings
    const { wrapper } = NEUtils.getElements();
    if (wrapper) {
        wrapper.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (typeof AppContextMenu === 'undefined' || typeof GridSettings === 'undefined') return;
            AppContextMenu.open(e.pageX, e.pageY, [
                { label: 'Grid…', onClick: () => GridSettings.openAt(e.pageX + 8, e.pageY + 8) }
            ]);
        });
    }

    // Try to restore saved workflow from localStorage
    if (typeof NEPersistence !== 'undefined' && NEPersistence.hasSavedWorkflow()) {
        try {
            const saved = NEPersistence.load();
            if (saved && saved.nodes && saved.nodes.length > 0) {
                NEPersistence.restore(saved);
                // Update workflow controls display
                if (typeof NEWorkflowControls !== 'undefined') {
                    NEWorkflowControls.updateDisplay();
                }
                console.log('Workflow restored from localStorage');
                return; // Skip default graph scaffolding
            }
        } catch (e) {
            console.warn('Failed to restore workflow, loading default:', e);
        }
    }

    // Scaffold a default graph on first open
    try {
        if (NodeEditor.nodes.length === 0 && typeof addNode === 'function') {
            addNode('input');
            const inputNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

            addNode('prompt');
            const promptNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

            addNode('aimodel');
            const aiNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

            addNode('output');
            const outputNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

            const { wrapper: layoutWrapper } = NEUtils.getElements();
            if (layoutWrapper) {
                const rect = layoutWrapper.getBoundingClientRect();
                const center = NEUtils.wrapperToCanvas(rect.width / 2, rect.height / 2);
                const rowY = center.y - 20;

                const layout = [
                    { node: inputNode, x: center.x + 100, y: rowY + 250 },
                    { node: promptNode, x: center.x + 100, y: rowY + 550 },
                    { node: aiNode, x: center.x + 500, y: rowY + 400 },
                    { node: outputNode, x: center.x + 900, y: rowY + 407 }
                ];

                layout.forEach(({ node, x, y }) => {
                    node.x = x;
                    node.y = y;
                    const el = document.getElementById('node-' + node.id);
                    if (el) {
                        el.style.left = x + 'px';
                        el.style.top = y + 'px';
                    }
                });
            }

            // Default connections
            if (typeof NEConnections !== 'undefined' && typeof NEConnections.addConnection === 'function') {
                NEConnections.addConnection(inputNode.id, 0, aiNode.id, 0);
                NEConnections.addConnection(promptNode.id, 0, aiNode.id, 1);
                NEConnections.addConnection(aiNode.id, 0, outputNode.id, 0);
            }

            // Ensure minimap and connections reflect initial graph
            const updateGraphVisuals = () => {
                if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
                if (typeof NEConnections !== 'undefined' && typeof NEConnections.updateConnections === 'function') {
                    NEConnections.updateConnections();
                }
            };

            requestAnimationFrame(updateGraphVisuals);
            setTimeout(updateGraphVisuals, 50);
        }
    } catch (e) {
        console.warn('Failed to scaffold default graph:', e);
    }
}

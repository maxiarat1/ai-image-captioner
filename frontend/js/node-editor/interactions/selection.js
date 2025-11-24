// Selection interactions for Node Editor - marquee (rubber-band) selection and multi-select
(function() {
    const NESelection = {};

    // ─────────────────────────────────────────────────────────────────────────────
    // State Management
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Get array of selected node IDs
     * @returns {number[]}
     */
    NESelection.getSelected = function() {
        return NodeEditor.selectedNodes || [];
    };

    /**
     * Check if a node is selected
     * @param {number} nodeId
     * @returns {boolean}
     */
    NESelection.isSelected = function(nodeId) {
        return (NodeEditor.selectedNodes || []).includes(nodeId);
    };

    /**
     * Select a single node (clears previous selection)
     * @param {number} nodeId
     */
    NESelection.select = function(nodeId) {
        NESelection.clearSelection();
        NodeEditor.selectedNodes = [nodeId];
        updateNodeSelectionVisual(nodeId, true);
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Add a node to selection
     * @param {number} nodeId
     */
    NESelection.addToSelection = function(nodeId) {
        if (!NodeEditor.selectedNodes) NodeEditor.selectedNodes = [];
        if (!NodeEditor.selectedNodes.includes(nodeId)) {
            NodeEditor.selectedNodes.push(nodeId);
            updateNodeSelectionVisual(nodeId, true);
        }
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Remove a node from selection
     * @param {number} nodeId
     */
    NESelection.deselect = function(nodeId) {
        if (!NodeEditor.selectedNodes) return;
        const idx = NodeEditor.selectedNodes.indexOf(nodeId);
        if (idx !== -1) {
            NodeEditor.selectedNodes.splice(idx, 1);
            updateNodeSelectionVisual(nodeId, false);
        }
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Toggle node selection
     * @param {number} nodeId
     */
    NESelection.toggle = function(nodeId) {
        if (NESelection.isSelected(nodeId)) {
            NESelection.deselect(nodeId);
        } else {
            NESelection.addToSelection(nodeId);
        }
    };

    /**
     * Select all nodes
     */
    NESelection.selectAll = function() {
        NodeEditor.selectedNodes = NodeEditor.nodes.map(n => n.id);
        NodeEditor.nodes.forEach(n => updateNodeSelectionVisual(n.id, true));
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Clear all selections
     */
    NESelection.clearSelection = function() {
        if (!NodeEditor.selectedNodes) return;
        NodeEditor.selectedNodes.forEach(id => updateNodeSelectionVisual(id, false));
        NodeEditor.selectedNodes = [];
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Update DOM visual for node selection state
     * @param {number} nodeId
     * @param {boolean} selected
     */
    function updateNodeSelectionVisual(nodeId, selected) {
        const el = document.getElementById('node-' + nodeId);
        if (el) {
            if (selected) {
                el.classList.add('selected');
            } else {
                el.classList.remove('selected');
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Selection Box (Marquee Selection)
    // ─────────────────────────────────────────────────────────────────────────────

    let selectionBoxElement = null;

    /**
     * Start selection box drawing
     * @param {MouseEvent} e
     */
    NESelection.startSelectionBox = function(e) {
        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);

        NodeEditor.selectionBox = {
            startX: canvasPos.x,
            startY: canvasPos.y,
            endX: canvasPos.x,
            endY: canvasPos.y
        };

        // Create selection box element
        createSelectionBoxElement();
        updateSelectionBoxVisual();

        // If not holding Shift or Ctrl, clear previous selection
        if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
            NESelection.clearSelection();
        }
    };

    /**
     * Update selection box during drag
     * @param {MouseEvent} e
     */
    NESelection.updateSelectionBox = function(e) {
        if (!NodeEditor.selectionBox) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);
        NodeEditor.selectionBox.endX = canvasPos.x;
        NodeEditor.selectionBox.endY = canvasPos.y;

        updateSelectionBoxVisual();
    };

    /**
     * End selection box and select nodes within it
     * @param {MouseEvent} e
     */
    NESelection.endSelectionBox = function(e) {
        if (!NodeEditor.selectionBox) return;

        // Find nodes within selection box
        const box = NodeEditor.selectionBox;
        const rect = {
            left: Math.min(box.startX, box.endX),
            top: Math.min(box.startY, box.endY),
            right: Math.max(box.startX, box.endX),
            bottom: Math.max(box.startY, box.endY)
        };

        // Select nodes that intersect with the selection box
        const nodesInBox = findNodesInRect(rect);

        if (e.shiftKey || e.ctrlKey || e.metaKey) {
            // Add to existing selection
            nodesInBox.forEach(nodeId => NESelection.addToSelection(nodeId));
        } else {
            // Replace selection
            NodeEditor.selectedNodes = nodesInBox;
            nodesInBox.forEach(id => updateNodeSelectionVisual(id, true));
        }

        // Clean up
        removeSelectionBoxElement();
        NodeEditor.selectionBox = null;

        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
    };

    /**
     * Cancel selection box without selecting
     */
    NESelection.cancelSelectionBox = function() {
        removeSelectionBoxElement();
        NodeEditor.selectionBox = null;
    };

    /**
     * Check if currently drawing a selection box
     * @returns {boolean}
     */
    NESelection.isSelecting = function() {
        return NodeEditor.selectionBox !== null;
    };

    /**
     * Find nodes whose bounding boxes intersect with the given rect
     * @param {{left: number, top: number, right: number, bottom: number}} rect
     * @returns {number[]} Array of node IDs
     */
    function findNodesInRect(rect) {
        const result = [];

        NodeEditor.nodes.forEach(node => {
            const el = document.getElementById('node-' + node.id);
            if (!el) return;

            // Get node bounds in canvas coordinates
            const nodeRect = {
                left: node.x,
                top: node.y,
                right: node.x + el.offsetWidth,
                bottom: node.y + el.offsetHeight
            };

            // Check AABB intersection
            if (rectsIntersect(rect, nodeRect)) {
                result.push(node.id);
            }
        });

        return result;
    }

    /**
     * Check if two rectangles intersect (AABB)
     */
    function rectsIntersect(a, b) {
        return !(
            a.right < b.left ||
            a.left > b.right ||
            a.bottom < b.top ||
            a.top > b.bottom
        );
    }

    /**
     * Create the selection box DOM element
     */
    function createSelectionBoxElement() {
        if (selectionBoxElement) return;

        const { canvas } = NEUtils.getElements();
        if (!canvas) return;

        selectionBoxElement = document.createElement('div');
        selectionBoxElement.id = 'selectionBox';
        selectionBoxElement.className = 'selection-box';
        canvas.appendChild(selectionBoxElement);
    }

    /**
     * Update selection box visual position and size
     */
    function updateSelectionBoxVisual() {
        if (!selectionBoxElement || !NodeEditor.selectionBox) return;

        const box = NodeEditor.selectionBox;
        const left = Math.min(box.startX, box.endX);
        const top = Math.min(box.startY, box.endY);
        const width = Math.abs(box.endX - box.startX);
        const height = Math.abs(box.endY - box.startY);

        selectionBoxElement.style.left = left + 'px';
        selectionBoxElement.style.top = top + 'px';
        selectionBoxElement.style.width = width + 'px';
        selectionBoxElement.style.height = height + 'px';
    }

    /**
     * Remove selection box element from DOM
     */
    function removeSelectionBoxElement() {
        if (selectionBoxElement && selectionBoxElement.parentElement) {
            selectionBoxElement.parentElement.removeChild(selectionBoxElement);
        }
        selectionBoxElement = null;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Multi-Node Drag
    // ─────────────────────────────────────────────────────────────────────────────

    let multiDragState = null;

    /**
     * Start dragging multiple selected nodes
     * @param {MouseEvent} e
     * @param {object} primaryNode - The node that was clicked
     */
    NESelection.startMultiDrag = function(e, primaryNode) {
        if (!NodeEditor.selectedNodes || NodeEditor.selectedNodes.length === 0) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);

        // Calculate offset for each selected node relative to the primary node
        const nodeOffsets = {};
        NodeEditor.selectedNodes.forEach(nodeId => {
            const node = NodeEditor.nodes.find(n => n.id === nodeId);
            if (node) {
                nodeOffsets[nodeId] = {
                    offsetX: node.x - primaryNode.x,
                    offsetY: node.y - primaryNode.y
                };
                // Add dragging class
                const el = document.getElementById('node-' + nodeId);
                if (el) el.classList.add('dragging');
            }
        });

        multiDragState = {
            primaryNode: primaryNode,
            primaryOffsetX: canvasPos.x - primaryNode.x,
            primaryOffsetY: canvasPos.y - primaryNode.y,
            nodeOffsets: nodeOffsets
        };

        // Set up guides if enabled
        if (NodeEditor.settings.showGuides) {
            ensureMultiDragGuides();
        }
    };

    /**
     * Update position of all selected nodes during multi-drag
     * @param {MouseEvent} e
     */
    NESelection.updateMultiDrag = function(e) {
        if (!multiDragState) return;

        const canvasPos = NEUtils.screenToCanvas(e.clientX, e.clientY);

        // Calculate new primary node position
        let newPrimaryX = canvasPos.x - multiDragState.primaryOffsetX;
        let newPrimaryY = canvasPos.y - multiDragState.primaryOffsetY;

        // Apply grid snapping to primary node
        const GRID = NodeEditor.settings.gridSize || 20;
        const EPS = NodeEditor.settings.snapEpsilon ?? 4;
        const mode = NodeEditor.settings.snapMode || 'always';

        const wantSnap = (
            (mode === 'always') ||
            (mode === 'withShift' && e.shiftKey) ||
            (mode === 'disableWithAlt' && !e.altKey)
        ) && mode !== 'off';

        if (wantSnap) {
            const rx = Math.round(newPrimaryX / GRID) * GRID;
            const ry = Math.round(newPrimaryY / GRID) * GRID;
            if (Math.abs(rx - newPrimaryX) <= EPS) newPrimaryX = rx;
            if (Math.abs(ry - newPrimaryY) <= EPS) newPrimaryY = ry;
        }

        // Update all selected nodes
        NodeEditor.selectedNodes.forEach(nodeId => {
            const node = NodeEditor.nodes.find(n => n.id === nodeId);
            if (!node) return;

            const offset = multiDragState.nodeOffsets[nodeId];
            let newX = newPrimaryX + offset.offsetX;
            let newY = newPrimaryY + offset.offsetY;

            // Clamp within canvas bounds
            const el = document.getElementById('node-' + nodeId);
            const maxX = NECanvas.SIZE - (el ? el.offsetWidth : 0);
            const maxY = NECanvas.SIZE - (el ? el.offsetHeight : 0);
            newX = Math.max(0, Math.min(maxX, newX));
            newY = Math.max(0, Math.min(maxY, newY));

            node.x = newX;
            node.y = newY;

            if (el) {
                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';
            }
        });

        // Update connections and minimap
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

        // Update guides
        if (NodeEditor.settings.showGuides) {
            updateMultiDragGuides(newPrimaryX, newPrimaryY);
        }
    };

    /**
     * End multi-drag and finalize positions
     */
    NESelection.endMultiDrag = function() {
        if (!multiDragState) return;

        const GRID = NodeEditor.settings.gridSize || 20;
        const mode = NodeEditor.settings.snapMode || 'always';

        // Final snap for all selected nodes
        NodeEditor.selectedNodes.forEach(nodeId => {
            const node = NodeEditor.nodes.find(n => n.id === nodeId);
            if (!node) return;

            if (mode !== 'off') {
                node.x = Math.round(node.x / GRID) * GRID;
                node.y = Math.round(node.y / GRID) * GRID;
            }

            // Final clamp
            const el = document.getElementById('node-' + nodeId);
            const maxX = NECanvas.SIZE - (el ? el.offsetWidth : 0);
            const maxY = NECanvas.SIZE - (el ? el.offsetHeight : 0);
            node.x = Math.max(0, Math.min(maxX, node.x));
            node.y = Math.max(0, Math.min(maxY, node.y));

            if (el) {
                el.classList.remove('dragging');
                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';
            }
        });

        // Update connections and minimap
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

        // Clean up guides
        removeMultiDragGuides();

        // Schedule auto-save
        if (typeof NEPersistence !== 'undefined') NEPersistence.scheduleSave();

        multiDragState = null;
    };

    /**
     * Check if currently in multi-drag mode
     * @returns {boolean}
     */
    NESelection.isMultiDragging = function() {
        return multiDragState !== null;
    };

    // --- Multi-drag guides helpers ---
    function ensureMultiDragGuides() {
        const { canvas } = NEUtils.getElements();
        if (!canvas) return;
        if (document.getElementById('multiDragGuides')) return;

        const div = document.createElement('div');
        div.id = 'multiDragGuides';
        div.className = 'drag-guides';
        canvas.appendChild(div);

        const h = document.createElement('div');
        h.className = 'drag-guide-line h';
        h.id = 'multiDragGuideH';
        const v = document.createElement('div');
        v.className = 'drag-guide-line v';
        v.id = 'multiDragGuideV';
        div.appendChild(h);
        div.appendChild(v);
    }

    function updateMultiDragGuides(x, y) {
        const h = document.getElementById('multiDragGuideH');
        const v = document.getElementById('multiDragGuideV');
        if (!h || !v) return;
        h.style.top = y + 'px';
        h.style.left = '0px';
        v.style.left = x + 'px';
        v.style.top = '0px';
    }

    function removeMultiDragGuides() {
        const div = document.getElementById('multiDragGuides');
        if (div && div.parentElement) div.parentElement.removeChild(div);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Keyboard Shortcuts
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Initialize keyboard shortcuts for selection
     */
    NESelection.initKeyboardShortcuts = function() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }

            // Ctrl+A / Cmd+A: Select all
            if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
                e.preventDefault();
                NESelection.selectAll();
            }

            // Escape: Clear selection
            if (e.key === 'Escape') {
                if (NESelection.isSelecting()) {
                    NESelection.cancelSelectionBox();
                } else {
                    NESelection.clearSelection();
                }
            }

            // Delete/Backspace: Delete selected nodes
            if (e.key === 'Delete' || e.key === 'Backspace') {
                const selected = NESelection.getSelected();
                if (selected.length > 0) {
                    e.preventDefault();
                    // Delete in reverse order to avoid index issues
                    selected.slice().reverse().forEach(nodeId => {
                        if (typeof deleteNode === 'function') {
                            deleteNode(nodeId);
                        } else if (typeof NEGraphOps !== 'undefined' && NEGraphOps.deleteNode) {
                            NEGraphOps.deleteNode(nodeId);
                        }
                    });
                    NESelection.clearSelection();
                }
            }
        });
    };

    // Export namespace
    window.NESelection = NESelection;
})();

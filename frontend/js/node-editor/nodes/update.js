// Node update helpers (UI and data sync)
(function() {
    const NENodeUpdate = {};

    // Update all input nodes with current queue count
    NENodeUpdate.updateInputNodes = function() {
        NodeEditor.nodes.forEach(node => {
            if (node.type === 'input') {
                const el = document.getElementById('node-' + node.id);
                if (el) {
                    const body = el.querySelector('.node-body');
                    if (body && typeof getNodeContent === 'function') {
                        body.innerHTML = getNodeContent(node);
                    }
                }
            }
        });
    };

    // Update conjunction node with connected items
    NENodeUpdate.updateConjunctionNode = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'conjunction') return;

        // Find all incoming connections
        const incomingConnections = NodeEditor.connections.filter(c => c.to === nodeId);

        // Gather connected items with reference keys
        const connectedItems = [];
        const usedRefKeys = new Set();
        const labelCounts = {};

        incomingConnections.forEach(conn => {
            const sourceNode = NodeEditor.nodes.find(n => n.id === conn.from);
            if (!sourceNode) return;

            // Skip Curate nodes - they're routing nodes, not content generators
            if (sourceNode.type === 'curate') return;

            const sourceDef = NODES[sourceNode.type];
            const portType = sourceDef.outputs[conn.fromPort];
            const baseLabel = sourceDef.label.replace(/\s+/g, '_');

            // Generate reference key with priority: custom label > content preview > auto-numbered
            let refKey;
            let shouldSetLabel = false;

            if (sourceNode.label && sourceNode.label.trim()) {
                refKey = NEUtils.sanitizeLabel(sourceNode.label);
            } else if (sourceNode.type === 'prompt' && sourceNode.data.text && sourceNode.data.text.trim()) {
                const preview = sourceNode.data.text.trim().substring(0, 20);
                refKey = NEUtils.sanitizeLabel(preview);
            } else {
                labelCounts[baseLabel] = (labelCounts[baseLabel] || 0) + 1;
                refKey = labelCounts[baseLabel] === 1 ? baseLabel : `${baseLabel}${labelCounts[baseLabel]}`;
                shouldSetLabel = true;
            }

            // Handle duplicates
            const originalRefKey = refKey;
            let counter = 2;
            while (usedRefKeys.has(refKey)) {
                refKey = `${originalRefKey}_${counter}`;
                counter++;
                shouldSetLabel = true;
            }
            usedRefKeys.add(refKey);

            if (shouldSetLabel) {
                sourceNode.label = refKey;
                const labelInput = document.getElementById(`node-${sourceNode.id}-label`);
                if (labelInput) labelInput.value = refKey;
            }

            let content = '';
            let preview = '';
            if (sourceNode.type === 'prompt') {
                content = sourceNode.data.text || '';
                preview = content.substring(0, 60) + (content.length > 60 ? '...' : '');
            } else if (sourceNode.type === 'aimodel') {
                content = '[Generated Captions]';
                preview = 'Captions from AI Model (generated at runtime)';
            } else {
                content = `[${sourceDef.label} Output]`;
                preview = `Output from ${sourceDef.label}`;
            }

            connectedItems.push({
                sourceId: sourceNode.id,
                sourceLabel: sourceDef.label,
                customLabel: sourceNode.label || '',
                refKey: refKey,
                portType: portType,
                content: content,
                preview: preview
            });
        });

        // Update node data
        node.data.connectedItems = connectedItems;

        // Update node display - only the references section
        const el = document.getElementById('node-' + nodeId);
        if (el) {
            const refsSection = document.getElementById(`node-${nodeId}-refs-section`);
            if (refsSection && typeof getConjunctionReferencesHtml === 'function') {
                refsSection.innerHTML = getConjunctionReferencesHtml(node);

                // Re-attach click handlers to reference items to insert placeholders
                const refItems = refsSection.querySelectorAll('.conjunction-ref-item');
                refItems.forEach(refItem => {
                    refItem.onclick = (e) => {
                        e.stopPropagation();
                        const refKey = refItem.dataset.refKey;
                        const textarea = el.querySelector(`#node-${nodeId}-template`);
                        if (!textarea || !refKey) return;

                        // Get cursor position
                        const start = textarea.selectionStart;
                        const end = textarea.selectionEnd;
                        const text = textarea.value;

                        // Insert reference at cursor position
                        const placeholder = `{${refKey}}`;
                        const newText = text.substring(0, start) + placeholder + text.substring(end);
                        textarea.value = newText;

                        // Update node data
                        node.data.template = newText;

                        // Set cursor after inserted text
                        const newCursorPos = start + placeholder.length;
                        textarea.setSelectionRange(newCursorPos, newCursorPos);

                        // Focus textarea and update highlights
                        textarea.focus();
                        if (typeof highlightPlaceholders === 'function') highlightPlaceholders(nodeId);
                    };
                });

                // Update placeholder highlighting
                if (typeof highlightPlaceholders === 'function') highlightPlaceholders(nodeId);

                // Update preview if visible
                if (typeof NENodes !== 'undefined' && typeof NENodes.updateConjunctionPreview === 'function') {
                    NENodes.updateConjunctionPreview(nodeId);
                }
            }
        }
    };

    window.NENodeUpdate = NENodeUpdate;
    window.updateInputNodes = NENodeUpdate.updateInputNodes;
    window.updateConjunctionNode = NENodeUpdate.updateConjunctionNode;
})();

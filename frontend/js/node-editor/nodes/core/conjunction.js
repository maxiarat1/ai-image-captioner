// Conjunction helpers split out from core.js
(function(){
    // Ensure NENodes exists (core.js will create it); if not, create a shallow object to attach to
    const NENodes = (window.NENodes = window.NENodes || {});

    // Highlight placeholders in conjunction template
    NENodes.highlightPlaceholders = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'conjunction') return;

        const textarea = document.getElementById(`node-${nodeId}-template`);
        const highlightsDiv = document.getElementById(`node-${nodeId}-highlights`);
        if (!textarea || !highlightsDiv) return;

        const text = textarea.value;
        const validKeys = (node.data.connectedItems || []).map(item => item.refKey);

        // Find all placeholders and mark them as valid or invalid
        const regex = /\{([^}]+)\}/g;
        let highlightedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        highlightedText = highlightedText.replace(regex, (match, key) => {
            const isValid = validKeys.includes(key);
            const className = isValid ? 'placeholder-valid' : 'placeholder-invalid';
            return `<mark class="${className}">${match}</mark>`;
        });

        // Add line breaks for proper alignment
        highlightedText = highlightedText.replace(/\n/g, '<br>');

        highlightsDiv.innerHTML = highlightedText;

        // Sync scroll
        highlightsDiv.scrollTop = textarea.scrollTop;
        highlightsDiv.scrollLeft = textarea.scrollLeft;
    };

    // Resolve conjunction template with actual values
    NENodes.resolveConjunctionTemplate = function(node) {
        if (!node || node.type !== 'conjunction') return '';

        const template = node.data.template || '';
        if (!template) return '';

        const items = node.data.connectedItems || [];
        const refMap = {};

        // Build reference map
        items.forEach(item => {
            refMap[item.refKey] = item.content;
        });

        // Replace placeholders with actual content
        const resolved = template.replace(/\{([^}]+)\}/g, (match, key) => {
            return refMap[key] !== undefined ? refMap[key] : match;
        });

        // Escape HTML for display
        return resolved.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
    };

    // Get conjunction references HTML (separate from body)
    NENodes.getConjunctionReferencesHtml = function(node) {
        const items = node.data.connectedItems || [];

        if (items.length === 0) {
            return `
                <div class="conjunction-references-empty">
                    Connect prompts/captions to use as references
                </div>
            `;
        }

        const refsItems = items.map(item => {
            // Build tooltip with label info
            let tooltipText = item.sourceLabel;
            if (item.customLabel) {
                tooltipText += ` (${item.customLabel})`;
            }
            tooltipText += `: ${item.preview}`;

            return `
                <div class="conjunction-ref-item" data-ref-key="${item.refKey}" title="${tooltipText}">
                    <span class="conjunction-ref-key">{${item.refKey}}</span>
                </div>
            `;
        }).join('');

        return `
            <div class="conjunction-references">
                <div class="conjunction-references-title">Available References:</div>
                <div class="conjunction-ref-list">
                    ${refsItems}
                </div>
            </div>
        `;
    };

    // Update conjunction preview
    NENodes.updateConjunctionPreview = function(nodeId) {
        const node = NodeEditor.nodes.find(n => n.id === nodeId);
        if (!node || node.type !== 'conjunction') return;

        const previewContent = document.querySelector(`#preview-${nodeId} .conjunction-preview-content`);
        if (!previewContent) return;

        const preview = NENodes.resolveConjunctionTemplate(node);
        previewContent.innerHTML = preview || '<em style="color: var(--text-secondary);">Empty template</em>';

        // Update recent outputs history (last 5)
        const historyEl = document.getElementById(`preview-${nodeId}-history`);
        if (historyEl) {
            const hist = Array.isArray(node.data.history) ? node.data.history.slice(-5) : [];
            if (hist.length === 0) {
                historyEl.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem;">No recent outputs</div>';
            } else {
                const escape = (s) => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
                // Show newest first
                const items = hist.slice().reverse().map(t => `<li class="conjunction-history-item" style="margin: 4px 0; line-height: 1.2;">${escape(t)}</li>`).join('');
                historyEl.innerHTML = `<ul class="conjunction-history-list" style="padding-left: 16px; margin: 4px 0 0 0;">${items}</ul>`;
            }
        }
    };

    // Also expose legacy global aliases in case core.js already assigned them earlier
    try {
        window.highlightPlaceholders = NENodes.highlightPlaceholders;
        window.resolveConjunctionTemplate = NENodes.resolveConjunctionTemplate;
        window.getConjunctionReferencesHtml = NENodes.getConjunctionReferencesHtml;
        window.updateConjunctionPreview = NENodes.updateConjunctionPreview;
    } catch (e) {
        // ignore in environments where window is locked down
    }

})();

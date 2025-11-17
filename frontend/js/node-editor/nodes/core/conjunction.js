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

    // Also expose legacy global aliases in case core.js already assigned them earlier
    try {
        window.highlightPlaceholders = NENodes.highlightPlaceholders;
        window.getConjunctionReferencesHtml = NENodes.getConjunctionReferencesHtml;
    } catch (e) {
        // ignore in environments where window is locked down
    }

})();

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

    // Also expose legacy global aliases in case core.js already assigned them earlier
    try {
        window.highlightPlaceholders = NENodes.highlightPlaceholders;
        window.resolveConjunctionTemplate = NENodes.resolveConjunctionTemplate;
    } catch (e) {
        // ignore in environments where window is locked down
    }

})();

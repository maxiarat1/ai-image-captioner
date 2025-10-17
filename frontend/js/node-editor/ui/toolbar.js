// Toolbar rendering for adding nodes
(function() {
    const NEToolbar = {};

    NEToolbar.setupToolbar = function() {
        const toolbar = document.getElementById('nodeToolbar');
        if (!toolbar) return;
        toolbar.innerHTML = '';

        for (const [type, def] of Object.entries(NODES)) {
            const btn = document.createElement('button');
            btn.textContent = def.label;
            // Use a safe fallback color if not specified
            const color = def.color || 'var(--primary)';
            btn.style.borderLeft = `3px solid ${color}`;
            btn.onclick = () => addNode(type);
            toolbar.appendChild(btn);
        }
    };

    window.NEToolbar = NEToolbar;
    // Alias removed; use NEToolbar.setupToolbar directly
})();

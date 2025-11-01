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
            // Add a generic per-type class so CSS can theme via shared variables (no inline colors)
            btn.classList.add('node-toolbar-btn', `node-type-${type}`);
            btn.onclick = () => addNode(type);
            toolbar.appendChild(btn);
        }
    };

    window.NEToolbar = NEToolbar;
    // Alias removed; use NEToolbar.setupToolbar directly
})();

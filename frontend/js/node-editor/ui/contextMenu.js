(function() {
    const NEContextMenu = {};

    function createMenuElement() {
        const menu = document.createElement('div');
        menu.id = 'nodeContextMenu';
        menu.className = 'node-context-menu';
        menu.style.display = 'none';
        menu.innerHTML = `
            <div class="menu-section">
                <div class="menu-section-title">Grid</div>
                <div class="menu-item">
                    <label for="cm-grid-size">Grid size</label>
                    <input id="cm-grid-size" type="number" min="5" max="80" step="1" />
                </div>
                <div class="menu-item">
                    <label for="cm-snap-mode">Snap mode</label>
                    <select id="cm-snap-mode">
                        <option value="always">Always on</option>
                        <option value="withShift">Only with Shift</option>
                        <option value="disableWithAlt">On (hold Alt to disable)</option>
                        <option value="off">Off</option>
                    </select>
                </div>
                <div class="menu-item menu-item-inline">
                    <label for="cm-show-guides">Show guides</label>
                    <input id="cm-show-guides" type="checkbox" />
                </div>
            </div>
        `;
        document.body.appendChild(menu);
        return menu;
    }

    function applySettingsFromControls() {
        const gridSizeEl = document.getElementById('cm-grid-size');
        const snapModeEl = document.getElementById('cm-snap-mode');
        const guidesEl = document.getElementById('cm-show-guides');
        if (!gridSizeEl || !snapModeEl || !guidesEl) return;

        const s = NodeEditor.settings;
        const size = parseInt(gridSizeEl.value, 10);
        if (!Number.isNaN(size) && size >= 5 && size <= 80) s.gridSize = size;
        s.snapMode = snapModeEl.value;
        s.showGuides = !!guidesEl.checked;
    }

    function syncControlsWithSettings() {
        const s = NodeEditor.settings;
        const gridSizeEl = document.getElementById('cm-grid-size');
        const snapModeEl = document.getElementById('cm-snap-mode');
        const guidesEl = document.getElementById('cm-show-guides');
        if (!gridSizeEl || !snapModeEl || !guidesEl) return;
        gridSizeEl.value = s.gridSize;
        snapModeEl.value = s.snapMode;
        guidesEl.checked = !!s.showGuides;
    }

    function showMenu(x, y) {
        const menu = document.getElementById('nodeContextMenu') || createMenuElement();
        syncControlsWithSettings();
        menu.style.left = `${x}px`;
        menu.style.top = `${y}px`;
        menu.style.display = 'block';

        // Focus first input for quick changes
        const gridSizeEl = document.getElementById('cm-grid-size');
        if (gridSizeEl) gridSizeEl.focus();
    }

    function hideMenu() {
        const menu = document.getElementById('nodeContextMenu');
        if (menu) menu.style.display = 'none';
    }

    function onCanvasContextMenu(e) {
        const { wrapper } = NEUtils.getElements();
        if (!wrapper) return;
        if (wrapper.contains(e.target) || e.target.id === 'nodeCanvas') {
            e.preventDefault();
            showMenu(e.pageX, e.pageY);
        }
    }

    function attachEvents() {
        document.addEventListener('contextmenu', onCanvasContextMenu);
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('nodeContextMenu');
            if (!menu) return;
            if (menu.style.display === 'none') return;
            if (!menu.contains(e.target)) hideMenu();
        });

        document.addEventListener('input', (e) => {
            const menu = document.getElementById('nodeContextMenu');
            if (!menu || menu.style.display === 'none') return;
            if (menu.contains(e.target)) applySettingsFromControls();
        });
        document.addEventListener('change', (e) => {
            const menu = document.getElementById('nodeContextMenu');
            if (!menu || menu.style.display === 'none') return;
            if (menu.contains(e.target)) applySettingsFromControls();
        });
    }

    NEContextMenu.init = function() {
        attachEvents();
    };

    window.NEContextMenu = NEContextMenu;
})();
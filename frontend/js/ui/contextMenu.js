(function() {
    const AppContextMenu = {};

    function ensureMenu() {
        let menu = document.getElementById('appContextMenu');
        if (!menu) {
            menu = document.createElement('div');
            menu.id = 'appContextMenu';
            menu.className = 'app-context-menu';
            document.body.appendChild(menu);
        }
        return menu;
    }

    function hideMenu() {
        const menu = ensureMenu();
        menu.style.display = 'none';
    }

    function showMenu(x, y, items) {
        const menu = ensureMenu();
        menu.innerHTML = '';
        items.forEach((item, idx) => {
            if (item.type === 'separator') {
                const sep = document.createElement('div');
                sep.className = 'menu-separator';
                menu.appendChild(sep);
                return;
            }
            const el = document.createElement('div');
            el.className = 'menu-item';
            el.textContent = item.label;
            el.onclick = (e) => {
                e.stopPropagation();
                hideMenu();
                if (typeof item.onClick === 'function') item.onClick(e);
            };
            menu.appendChild(el);
        });
        menu.style.left = `${x}px`;
        menu.style.top = `${y}px`;
        menu.style.display = 'block';
    }

    // Public API
    AppContextMenu.open = showMenu;
    AppContextMenu.close = hideMenu;

    // Global click closes menu
    document.addEventListener('click', () => hideMenu());

    window.AppContextMenu = AppContextMenu;
})();

// Grid settings popover (reusable UI)
(function() {
    const GridSettings = {};

    function ensurePopover() {
        let pop = document.getElementById('gridSettingsPopover');
        if (!pop) {
            pop = document.createElement('div');
            pop.id = 'gridSettingsPopover';
            pop.className = 'settings-popover';
            pop.style.display = 'none';
            pop.innerHTML = `
                <div class="settings-title">Grid Settings</div>
                <div class="settings-row">
                    <label for="gs-grid-size">Grid size</label>
                    <input id="gs-grid-size" type="number" min="5" max="80" step="1" />
                </div>
                <div class="settings-row">
                    <label for="gs-snap-mode">Snap mode</label>
                    <select id="gs-snap-mode">
                        <option value="always">Always on</option>
                        <option value="withShift">Only with Shift</option>
                        <option value="disableWithAlt">On (hold Alt to disable)</option>
                        <option value="off">Off</option>
                    </select>
                </div>
                <div class="settings-row">
                    <label for="gs-show-guides">Show guides</label>
                    <input id="gs-show-guides" type="checkbox" />
                </div>
            `;
            document.body.appendChild(pop);
        }
        return pop;
    }

    function syncFromSettings() {
        const s = window.NodeEditor ? NodeEditor.settings : { gridSize: 20, snapMode: 'always', showGuides: true };
        const sizeEl = document.getElementById('gs-grid-size');
        const modeEl = document.getElementById('gs-snap-mode');
        const guidesEl = document.getElementById('gs-show-guides');
        if (!sizeEl || !modeEl || !guidesEl) return;
        sizeEl.value = s.gridSize;
        modeEl.value = s.snapMode;
        guidesEl.checked = !!s.showGuides;
    }

    function applyToSettings() {
        if (!window.NodeEditor) return;
        const s = NodeEditor.settings;
        const sizeEl = document.getElementById('gs-grid-size');
        const modeEl = document.getElementById('gs-snap-mode');
        const guidesEl = document.getElementById('gs-show-guides');
        const size = parseInt(sizeEl.value, 10);
        if (!Number.isNaN(size) && size >= 5 && size <= 80) {
            s.gridSize = size;
            // Update CSS variable for grid size
            const canvas = document.getElementById('nodeCanvas');
            if (canvas) {
                canvas.style.setProperty('--grid-size', `${size}px`);
            }
        }
        s.snapMode = modeEl.value;
        s.showGuides = !!guidesEl.checked;
    }

    function hide() {
        const pop = ensurePopover();
        pop.style.display = 'none';
    }

    function show(x, y) {
        const pop = ensurePopover();
        syncFromSettings();
        pop.style.left = `${x}px`;
        pop.style.top = `${y}px`;
        pop.style.display = 'block';
    }

    document.addEventListener('input', (e) => {
        const pop = document.getElementById('gridSettingsPopover');
        if (pop && pop.style.display !== 'none' && pop.contains(e.target)) {
            applyToSettings();
        }
    });
    document.addEventListener('change', (e) => {
        const pop = document.getElementById('gridSettingsPopover');
        if (pop && pop.style.display !== 'none' && pop.contains(e.target)) {
            applyToSettings();
        }
    });
    document.addEventListener('click', (e) => {
        const pop = document.getElementById('gridSettingsPopover');
        const menu = document.getElementById('appContextMenu');
        if (!pop) return;
        if (pop.style.display === 'none') return;
        if (pop.contains(e.target)) return; // keep open if clicking inside
        if (menu && menu.contains(e.target)) return; // clicking the menu shouldn't close popover immediately
        hide();
    });

    GridSettings.openAt = show;
    GridSettings.close = hide;

    window.GridSettings = GridSettings;
})();
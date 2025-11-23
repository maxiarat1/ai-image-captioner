// Core state for the Node Editor
// Exposes global NodeEditor and NODES used by other modules and legacy code
(function() {
    // Only define if not already present (defensive in case of multiple loads)
    if (!window.NodeEditor) {
        window.NodeEditor = {
            nodes: [],
            connections: [],
            nextId: 1,
            dragging: null,
            connecting: null,
            panning: null,
            selectedNodes: [],
            selectionBox: null,
            transform: {
                x: 0,
                y: 0,
                scale: 1
                },
                // Editor runtime settings
                settings: {
                    gridSize: 20,        // px grid size
                    snapEpsilon: 4,      // soft snap range in px
                    snapMode: 'always',  // 'always' | 'withShift' | 'disableWithAlt' | 'off'
                    showGuides: true     // show faint guides while dragging
                }
        };
    }

    if (!window.NODES) {
        window.NODES = {
            input: { label: 'Input', inputs: [], outputs: ['images'] },
            prompt: { label: 'Prompt', inputs: [], outputs: ['text'] },
            conjunction: { label: 'Conjunction', inputs: ['captions'], outputs: ['captions'] },
            aimodel: { label: 'AI Model', inputs: ['images', 'prompt'], outputs: ['captions'] },
            curate: {
                label: 'Curate',
                inputs: ['images (optional)', 'captions'],  // Images optional, captions from previous nodes
                outputs: [],  // Dynamic outputs managed at runtime
                allowDynamicOutputs: true
            },
            output: { label: 'Output', inputs: ['data'], outputs: [] }
        };
    }
})();

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
            transform: {
                x: 0,
                y: 0,
                scale: 1
            }
        };
    }

    if (!window.NODES) {
        window.NODES = {
            input: { label: 'Input', inputs: [], outputs: ['images'] },
            prompt: { label: 'Prompt', inputs: [], outputs: ['text'] },
            conjunction: { label: 'Conjunction', inputs: ['text', 'captions'], outputs: ['text'] },
            aimodel: { label: 'AI Model', inputs: ['images', 'prompt'], outputs: ['captions'] },
            output: { label: 'Output', inputs: ['data'], outputs: [] }
        };
    }
})();

// Core utilities for the Node Editor
// Provides helper functions and preserves global names for backward compatibility
(function() {
    const NEUtils = {};

    // Get frequently used DOM elements
    NEUtils.getElements = () => ({
        canvas: document.getElementById('nodeCanvas'),
        wrapper: document.querySelector('.node-canvas-wrapper'),
        svg: document.getElementById('connectionsSVG')
    });

    /**
     * Convert screen coordinates to canvas coordinates
     * @param {number} screenX - Screen X position
     * @param {number} screenY - Screen Y position
     * @returns {{x: number, y: number}} Canvas coordinates (0 to 5000)
     */
    NEUtils.screenToCanvas = function(screenX, screenY) {
        const { canvas } = NEUtils.getElements();
        const rect = canvas.getBoundingClientRect();
        return {
            x: (screenX - rect.left) / NodeEditor.transform.scale,
            y: (screenY - rect.top) / NodeEditor.transform.scale
        };
    };

    /**
     * Convert wrapper-relative position to canvas coordinates
     * @param {number} screenX - Wrapper-relative X position
     * @param {number} screenY - Wrapper-relative Y position
     * @returns {{x: number, y: number}} Canvas coordinates (0 to 5000)
     */
    NEUtils.wrapperToCanvas = function(screenX, screenY) {
        return {
            x: (screenX - NodeEditor.transform.x) / NodeEditor.transform.scale,
            y: (screenY - NodeEditor.transform.y) / NodeEditor.transform.scale
        };
    };

    // Sanitize label to create valid reference key
    NEUtils.sanitizeLabel = function(label) {
        if (!label || typeof label !== 'string') return '';

        // Replace spaces with underscores, remove special chars except underscores
        return label
            .trim()
            .replace(/\s+/g, '_')
            .replace(/[^a-zA-Z0-9_]/g, '')
            .substring(0, 30); // Limit length
    };

    // Apply transform to canvas
    NEUtils.applyTransform = function() {
        const canvas = document.getElementById('nodeCanvas');
        if (!canvas) return;

        const { x, y, scale } = NodeEditor.transform;
        canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    };

    // Export namespace and keep global aliases for transitional compatibility
    window.NEUtils = NEUtils;
    window.getElements = NEUtils.getElements;
    window.screenToCanvas = NEUtils.screenToCanvas;
    window.wrapperToCanvas = NEUtils.wrapperToCanvas;
    window.sanitizeLabel = NEUtils.sanitizeLabel;
    window.applyTransform = NEUtils.applyTransform;
})();
        // Aliases removed after migration; use NEUtils.* directly

// Fullscreen modal behavior for the Node Editor
(function() {
    const NEFullscreen = {};

    // Open fullscreen modal
    NEFullscreen.openFullscreen = function() {
        const modal = document.getElementById('nodeFullscreenModal');
        const container = document.getElementById('fullscreenCanvasContainer');
        const toolbar = document.getElementById('nodeToolbar');
        const wrapper = document.querySelector('.node-canvas-wrapper');
        if (!modal || !container || !toolbar || !wrapper) return;

        // Move toolbar and wrapper into modal
        container.appendChild(toolbar);
        container.appendChild(wrapper);

        // Show modal
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Update visuals to reflect new container, but keep current transform
        requestAnimationFrame(() => {
            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
        });

        // Add zoom support to fullscreen container
        const fullscreenZoomHandler = (e) => {
            e.preventDefault();
            NEViewport.handleZoom(e);
        };
        container.addEventListener('wheel', fullscreenZoomHandler);
        // Store handler for cleanup
        container._zoomHandler = fullscreenZoomHandler;
    };

    // Close fullscreen modal
    NEFullscreen.closeFullscreen = function() {
        const modal = document.getElementById('nodeFullscreenModal');
        const container = document.getElementById('fullscreenCanvasContainer');
        if (!modal || !container) return;

        // Remove zoom event listener from fullscreen container
        if (container._zoomHandler) {
            container.removeEventListener('wheel', container._zoomHandler);
            delete container._zoomHandler;
        }

        // Move elements back immediately for smooth transition
        const editorContainer = document.querySelector('.node-editor-container');
        const toolbar = document.getElementById('nodeToolbar');
        const wrapper = document.querySelector('.node-canvas-wrapper');
        
        editorContainer.appendChild(toolbar);
        editorContainer.appendChild(wrapper);
        document.body.style.overflow = '';

        // Update visuals immediately
        if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
        if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();

        // Then trigger the modal close animation
        modal.classList.add('closing');

        // Clean up after animation completes
        const handleAnimationEnd = (e) => {
            if (e.target === modal) {
                modal.removeEventListener('animationend', handleAnimationEnd);
                modal.classList.remove('active', 'closing');
            }
        };

        modal.addEventListener('animationend', handleAnimationEnd);
    };

    // Export
    window.NEFullscreen = NEFullscreen;
})();

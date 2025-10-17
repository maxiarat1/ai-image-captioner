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

        // Re-center canvas for fullscreen dimensions
        setTimeout(() => {
            const rect = container.getBoundingClientRect();
            NodeEditor.transform.x = rect.width / 2 - 2500;
            NodeEditor.transform.y = rect.height / 2 - 2500;
            NEUtils.applyTransform();
            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
        }, 50);

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

        // Add closing class to trigger animation
        modal.classList.add('closing');

        // Wait for animation to complete using animationend event
        const handleAnimationEnd = (e) => {
            // Only handle the modal's own animation, not child animations
            if (e.target === modal) {
                modal.removeEventListener('animationend', handleAnimationEnd);

                // Animation complete - now safe to move elements and cleanup
                const editorContainer = document.querySelector('.node-editor-container');
                const toolbar = document.getElementById('nodeToolbar');
                const wrapper = document.querySelector('.node-canvas-wrapper');

                modal.classList.remove('active', 'closing');
                // Move toolbar and wrapper back to editor container
                editorContainer.appendChild(toolbar);
                editorContainer.appendChild(wrapper);
                document.body.style.overflow = '';

                // Re-center canvas for normal wrapper dimensions
                setTimeout(() => {
                    const rect = wrapper.getBoundingClientRect();
                    NodeEditor.transform.x = rect.width / 2 - 2500;
                    NodeEditor.transform.y = rect.height / 2 - 2500;
                    NEUtils.applyTransform();
                    if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
                    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
                }, 50);
            }
        };

        modal.addEventListener('animationend', handleAnimationEnd);
    };

    // Export
    window.NEFullscreen = NEFullscreen;
})();

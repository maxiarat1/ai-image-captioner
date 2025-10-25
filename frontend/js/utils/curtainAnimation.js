// ============================================================================
// Curtain Animation Utility
// ============================================================================

function createCurtainTransition(callback) {
    const curtain = document.getElementById('curtainOverlay');
    
    curtain.classList.add('closing');
    
    setTimeout(() => {
        callback();
        
        setTimeout(() => {
            curtain.classList.remove('closing');
            curtain.classList.add('opening');
            
            setTimeout(() => curtain.classList.remove('opening'), 800);
        }, 300);
    }, 800);
}

// ============================================================================
// Page Load Transition
// ============================================================================

window.addEventListener('load', () => {
    const pageTransition = document.getElementById('pageTransition');

    setTimeout(() => {
        pageTransition.classList.add('fade-out');
        setTimeout(() => {
            pageTransition.style.display = 'none';
        }, 400);
    }, 800);
});

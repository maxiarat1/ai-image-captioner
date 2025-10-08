// ============================================================================
// Theme Toggle
// ============================================================================

function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const htmlElement = document.documentElement;
    const curtain = document.getElementById('curtainOverlay');

    const currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') {
        htmlElement.setAttribute('data-theme', 'light');
    }

    themeToggle.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        // Theatrical curtain animation
        curtain.classList.add('closing');
        
        setTimeout(() => {
            // Change theme while curtains are fully closed
            if (newTheme === 'light') {
                htmlElement.setAttribute('data-theme', 'light');
            } else {
                htmlElement.removeAttribute('data-theme');
            }
            localStorage.setItem('theme', newTheme);
            
            // Wait 0.5s with curtains closed, then open
            setTimeout(() => {
                curtain.classList.remove('closing');
                curtain.classList.add('opening');
                
                setTimeout(() => curtain.classList.remove('opening'), 800);
            }, 300);
        }, 800);
    });
}

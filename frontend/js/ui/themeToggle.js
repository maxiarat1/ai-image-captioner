// ============================================================================
// Theme Toggle
// ============================================================================

function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const htmlElement = document.documentElement;

    const currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') {
        htmlElement.setAttribute('data-theme', 'light');
    }

    themeToggle.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        createCurtainTransition(() => {
            if (newTheme === 'light') {
                htmlElement.setAttribute('data-theme', 'light');
            } else {
                htmlElement.removeAttribute('data-theme');
            }
            localStorage.setItem('theme', newTheme);
        });
    });
}

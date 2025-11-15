// Helper functions for displaying parameter conflict warnings

function showParameterWarning(container, message) {
    let warning = container.querySelector('.param-warning');
    if (!warning) {
        warning = document.createElement('div');
        warning.className = 'param-warning';
        container.insertBefore(warning, container.firstChild);
    }
    warning.textContent = '⚠️ ' + message;
    warning.style.display = 'block';
}

function clearParameterWarning(container) {
    const warning = container.querySelector('.param-warning');
    if (warning) {
        warning.style.display = 'none';
    }
}

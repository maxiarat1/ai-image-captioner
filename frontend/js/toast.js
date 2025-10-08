// ============================================================================
// Toast Notifications
// ============================================================================

let toastTimeout;
let isProcessing = false;
let isPaused = false;
let shouldStop = false;

function showToast(message, keepVisible = false, progress = null) {
    const statusToast = document.getElementById('statusToast');

    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }

    // If progress is provided, show progress bar
    if (progress !== null && progress !== undefined) {
        const percentage = Math.round(progress * 100);
        statusToast.innerHTML = `
            <div class="toast-message">${message}</div>
            <div class="toast-progress-bar">
                <div class="toast-progress-fill" style="width: ${percentage}%"></div>
            </div>
        `;
    } else {
        statusToast.innerHTML = `<div class="toast-message">${message}</div>`;
    }

    statusToast.classList.add('show');

    if (!keepVisible && !isProcessing) {
        toastTimeout = setTimeout(() => {
            statusToast.classList.remove('show');
        }, 2000);
    }
}

// Initialize toast hover behavior
function initToastHoverBehavior() {
    const statusToast = document.getElementById('statusToast');

    document.addEventListener('mousemove', (e) => {
        if (!statusToast.classList.contains('show')) return;

        // Get toast position and dimensions
        const rect = statusToast.getBoundingClientRect();

        // Calculate distance from cursor to toast (with 100px buffer zone)
        const bufferZone = 100;
        const distanceX = Math.max(0, Math.max(rect.left - e.clientX, e.clientX - rect.right));
        const distanceY = Math.max(0, Math.max(rect.top - e.clientY, e.clientY - rect.bottom));
        const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

        // Hide toast if cursor is within buffer zone
        if (distance < bufferZone) {
            statusToast.classList.add('toast-hidden');
        } else {
            statusToast.classList.remove('toast-hidden');
        }
    });
}

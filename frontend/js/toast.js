// ============================================================================
// Toast Notifications
// ============================================================================

let toastTimeout;
let isProcessing = false;
let isPaused = false;
let shouldStop = false;
let isHovering = false;
let currentToastType = 'info';

function showToast(message, keepVisible = false, progress = null, type = 'info') {
    const statusToast = document.getElementById('statusToast');

    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }

    // Store the current toast type
    currentToastType = type;

    // Add/remove error class based on type
    if (type === 'error') {
        statusToast.classList.add('toast-error');
    } else {
        statusToast.classList.remove('toast-error');
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
        // Start auto-hide timer (will be paused on hover if it's an error)
        startToastTimer();
    }
}

function startToastTimer() {
    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }

    // Don't start timer if user is hovering over an error toast
    if (isHovering && currentToastType === 'error') {
        return;
    }

    toastTimeout = setTimeout(() => {
        const statusToast = document.getElementById('statusToast');
        statusToast.classList.remove('show');
    }, 2000);
}

// Initialize toast hover behavior
function initToastHoverBehavior() {
    const statusToast = document.getElementById('statusToast');

    // Add hover listeners to pause error toasts
    statusToast.addEventListener('mouseenter', () => {
        isHovering = true;

        // If this is an error toast, pause the auto-hide timer
        if (currentToastType === 'error') {
            if (toastTimeout) {
                clearTimeout(toastTimeout);
                toastTimeout = null;
            }
            // Make toast easier to read/copy by preventing it from hiding
            statusToast.classList.remove('toast-hidden');
        }
    });

    statusToast.addEventListener('mouseleave', () => {
        isHovering = false;

        // If this was an error toast, resume the timer
        if (currentToastType === 'error') {
            startToastTimer();
        }
    });

    // Existing behavior: hide toast when cursor approaches (for non-error toasts)
    document.addEventListener('mousemove', (e) => {
        if (!statusToast.classList.contains('show')) return;

        // Skip this behavior for error toasts (let them stay visible on hover)
        if (currentToastType === 'error') return;

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

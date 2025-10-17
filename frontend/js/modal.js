// ============================================================================
// Image Preview Modal
// ============================================================================

function openImagePreview(imageSrc, captionText, fileName) {
    const modal = document.getElementById('imagePreviewModal');
    const modalImage = document.getElementById('modalImage');
    const modalInfo = document.getElementById('modalInfo');
    const backdrop = document.getElementById('modalBackdrop');
    const activeTab = document.querySelector('.tab-content.active');
    const currentTab = activeTab?.id;

    modalImage.src = imageSrc;
    modalInfo.textContent = captionText || fileName;
    backdrop.className = `modal-backdrop ${currentTab}`;
    modal.classList.add('active');

    // Prevent body scrolling when modal is open
    document.body.style.overflow = 'hidden';

    // Smoothly blur the active tab content behind the modal
    if (activeTab) activeTab.classList.add('blur-bg');
}

function closeImagePreview() {
    const modal = document.getElementById('imagePreviewModal');
    modal.classList.remove('active');

    // Re-enable body scrolling
    document.body.style.overflow = '';
    
    // Remove blur from whichever tab is currently active
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) activeTab.classList.remove('blur-bg');
}

// Initialize modal close handlers
function initModalHandlers() {
    const modal = document.getElementById('imagePreviewModal');
    const modalBackdrop = document.getElementById('modalBackdrop');
    const modalContent = document.querySelector('.modal-content');

    // Close on backdrop click (clicking outside modal content)
    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            closeImagePreview();
        }
    });

    // Prevent closing when clicking inside modal content
    if (modalContent) {
        modalContent.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    // Close on Esc key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeImagePreview();
        }
    });
}

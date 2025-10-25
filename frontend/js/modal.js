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

    document.body.style.overflow = 'hidden';

    if (activeTab) activeTab.classList.add('blur-bg');
}

function closeImagePreview() {
    const modal = document.getElementById('imagePreviewModal');
    modal.classList.remove('active');

    document.body.style.overflow = '';
    
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) activeTab.classList.remove('blur-bg');
}

function initModalHandlers() {
    const modal = document.getElementById('imagePreviewModal');
    const modalBackdrop = document.getElementById('modalBackdrop');
    const modalContent = document.querySelector('.modal-content');

    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            closeImagePreview();
        }
    });

    if (modalContent) {
        modalContent.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeImagePreview();
        }
    });
}

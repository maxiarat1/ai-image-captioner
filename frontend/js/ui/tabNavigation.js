// ============================================================================
// Tab Navigation
// ============================================================================

function initTabNavigation() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabNavigation = document.querySelector('.tab-navigation');

    tabNavigation.setAttribute('data-active', '0');

    tabBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            tabNavigation.setAttribute('data-active', index.toString());

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${targetTab}-content`).classList.add('active');
        });
    });
}

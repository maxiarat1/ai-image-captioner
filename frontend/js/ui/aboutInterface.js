// ============================================================================
// About Interface
// ============================================================================

function initAboutInterface() {
    const aboutOverlay = document.getElementById('aboutOverlay');
    const aboutCloseBtn = document.getElementById('aboutCloseBtn');
    const toolInterface = document.getElementById('tool-interface');

    // Check backend status (only when About page is opened)
    async function checkBackendStatus() {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.about-status span:last-child');

        try {
            const response = await fetch(`${AppState.apiBaseUrl}/health`, {
                signal: AbortSignal.timeout(2000)
            });
            if (response.ok) {
                indicator.style.background = '#22c55e';
                indicator.style.boxShadow = '0 0 12px #22c55e';
                statusText.textContent = 'System Operational';
            } else {
                indicator.style.background = '#ef4444';
                indicator.style.boxShadow = '0 0 12px #ef4444';
                statusText.textContent = 'Backend Offline';
            }
        } catch {
            // Silently handle offline state
            indicator.style.background = '#ef4444';
            indicator.style.boxShadow = '0 0 12px #ef4444';
            statusText.textContent = 'Backend Offline';
        }
    }

    // Load dynamic content from backend
    async function loadDynamicContent() {
        try {
            await checkBackendStatus();
            await updateStats();
            await updateTechStack();
        } catch (error) {
            console.error('Error loading dynamic about content:', error);
        }
    }

    // Update stats from backend
    async function updateStats() {
        const modelCount = await ModelsAPI.getModelCount();
        const vramRange = await ModelsAPI.getVRAMRange();
        const exportFormats = await ModelsAPI.getExportFormatsCount();

        const statModelCount = document.getElementById('statModels');
        const statVram = document.getElementById('statVram');
        const statExportFormats = document.getElementById('statExportFormats');

        if (statModelCount) statModelCount.textContent = modelCount;
        if (statVram) statVram.textContent = vramRange;
        if (statExportFormats) statExportFormats.textContent = exportFormats;
    }

    // Update tech stack from backend
    async function updateTechStack() {
        const techStack = await ModelsAPI.getTechStack();
        const techGrid = document.querySelector('.tech-grid');

        if (!techGrid || techStack.length === 0) return;

        // Clear existing content
        techGrid.innerHTML = '';

        // Generate tech cards
        techStack.forEach(tech => {
            const card = document.createElement('div');
            card.className = 'tech-card';
            card.innerHTML = `
                <div class="tech-name">${tech.name}</div>
                <div class="tech-desc">${tech.description}</div>
            `;
            techGrid.appendChild(card);
        });
    }

    // Open about
    function openAbout() {
        createCurtainTransition(() => {
            if (toolInterface) toolInterface.style.display = 'none';
            aboutOverlay.classList.add('active');
            animateStats();
        });
    }

    // Close about
    function closeAbout() {
        createCurtainTransition(() => {
            aboutOverlay.classList.remove('active');
            if (toolInterface) toolInterface.style.display = 'block';
        });
    }

    // Animate stats with counter effect
    function animateStats() {
        const statSpeed = document.getElementById('statSpeed');

        if (statSpeed) {
            // Simple counter animation for speed stat
            let count = 0;
            const target = 1.2;
            const increment = target / 30;
            const interval = setInterval(() => {
                count += increment;
                if (count >= target) {
                    statSpeed.textContent = target.toFixed(1);
                    clearInterval(interval);
                } else {
                    statSpeed.textContent = count.toFixed(1);
                }
            }, 30);
        }
    }

    // Close button handler
    if (aboutCloseBtn) {
        aboutCloseBtn.addEventListener('click', closeAbout);
        
        // Auto-hide close button after 5 seconds
        let hideTimeout;
        let isButtonHidden = false;

        // Function to show the button
        function showCloseBtn() {
            aboutCloseBtn.style.opacity = '1';
            aboutCloseBtn.style.pointerEvents = 'auto';
            isButtonHidden = false;
        }

        // Function to hide the button
        function hideCloseBtn() {
            aboutCloseBtn.style.opacity = '0';
            aboutCloseBtn.style.pointerEvents = 'none';
            isButtonHidden = true;
        }

        // Add smooth transition
        aboutCloseBtn.style.transition = 'opacity 0.3s ease-in-out';

        // Enhanced openAbout to handle button hiding
        window.aboutOpenHandler = function() {
            // Show button initially
            showCloseBtn();

            // Hide after 5 seconds
            clearTimeout(hideTimeout);
            hideTimeout = setTimeout(() => {
                hideCloseBtn();
            }, 5000);
        };
        
        // Track mouse movement to show button when near
        aboutOverlay.addEventListener('mousemove', (e) => {
            if (!aboutOverlay.classList.contains('active')) return;
            
            const rect = aboutCloseBtn.getBoundingClientRect();
            const distance = 150; // Distance threshold in pixels
            
            // Calculate distance from cursor to button
            const dx = Math.max(rect.left - e.clientX, 0, e.clientX - rect.right);
            const dy = Math.max(rect.top - e.clientY, 0, e.clientY - rect.bottom);
            const distanceToButton = Math.sqrt(dx * dx + dy * dy);
            
            if (distanceToButton < distance && isButtonHidden) {
                showCloseBtn();
            } else if (distanceToButton >= distance && !isButtonHidden) {
                // Only auto-hide if not hovering directly over button
                const isHoveringButton = e.clientX >= rect.left && e.clientX <= rect.right &&
                                        e.clientY >= rect.top && e.clientY <= rect.bottom;
                if (!isHoveringButton) {
                    hideCloseBtn();
                }
            }
        });
    }

    // ESC key handler
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && aboutOverlay.classList.contains('active')) {
            closeAbout();
        }
    });

    // Feature cards hover effect with parallax
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        // Add smooth transition for the tilt effect
        card.style.transition = 'transform 0.1s ease-out';
        
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });

    // Load dynamic content and handle close button when opening about
    const originalOpenAbout = openAbout;
    openAbout = async function() {
        originalOpenAbout();
        await loadDynamicContent();
        
        // Trigger close button handler if it exists
        if (window.aboutOpenHandler) {
            window.aboutOpenHandler();
        }
    };

    // Expose open function globally for navigation links
    window.openAbout = openAbout;
}

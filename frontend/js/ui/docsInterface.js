// ============================================================================
// Documentation Interface
// ============================================================================

function initDocsInterface() {
    const docsOverlay = document.getElementById('docsOverlay');
    const docsCloseBtn = document.getElementById('docsCloseBtn');
    const docsNavItems = document.querySelectorAll('.docs-nav-item');
    const docsSections = document.querySelectorAll('.docs-section');
    const toolInterface = document.getElementById('tool-interface');

    // Load dynamic content from backend
    async function loadDynamicContent() {
        try {
            await updateModelComparison();
        } catch (error) {
            console.error('Error loading dynamic docs content:', error);
        }
    }

    // Update model comparison matrix
    async function updateModelComparison() {
        const models = await ModelsAPI.getModels();
        const perfMatrix = document.querySelector('.performance-matrix');

        if (!perfMatrix || Object.keys(models).length === 0) return;

        // Clear existing content
        perfMatrix.innerHTML = '';

        // Show all available models dynamically from backend
        const allModelKeys = Object.keys(models);

        allModelKeys.forEach(modelKey => {
            const model = models[modelKey];
            if (!model) return;

            // For R-4B, show precision variants
            if (modelKey === 'r4b' && model.precision_variants) {
                model.precision_variants.forEach(variant => {
                    const card = createModelComparisonCard(
                        `${model.display_name} (${variant.name})`,
                        variant.speed_score,
                        `${variant.vram_gb}GB`
                    );
                    perfMatrix.appendChild(card);
                });
            } else {
                const card = createModelComparisonCard(
                    model.display_name,
                    model.speed_score,
                    model.vram_label
                );
                perfMatrix.appendChild(card);
            }
        });
    }

    // Create a model comparison card
    function createModelComparisonCard(name, speedScore, vramLabel) {
        const card = document.createElement('div');
        card.className = 'perf-model-card';

        const speedLabel = speedScore >= 70 ? 'Fast' : speedScore >= 50 ? 'Medium' : 'Slow';


        card.innerHTML = `
            <div class="perf-model-header">${name}</div>
            <div class="perf-metric">
                <span class="perf-label">Speed</span>
                <div class="perf-bar"><div class="perf-fill" style="width: ${speedScore}%"></div></div>
                <span class="perf-value">${speedLabel}</span>
            </div>
            <div class="perf-metric">
                <span class="perf-label">VRAM</span>
                <div class="perf-bar"><div class="perf-fill" style="width: ${Math.min(100, (parseFloat(vramLabel) / 22) * 100)}%"></div></div>
                <span class="perf-value">${vramLabel}</span>
            </div>
        `;

        return card;
    }

    // Open documentation
    function openDocs() {
        createCurtainTransition(() => {
            if (toolInterface) toolInterface.style.display = 'none';
            docsOverlay.classList.add('active');
        });
    }

    // Close documentation
    function closeDocs() {
        createCurtainTransition(() => {
            docsOverlay.classList.remove('active');
            if (toolInterface) toolInterface.style.display = 'block';
        });
    }

    // Close button handler
    if (docsCloseBtn) {
        docsCloseBtn.addEventListener('click', closeDocs);
    }

    // ESC key handler
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && docsOverlay.classList.contains('active')) {
            closeDocs();
        }
    });

    // Section navigation
    docsNavItems.forEach(navItem => {
        navItem.addEventListener('click', () => {
            const sectionId = navItem.getAttribute('data-section');

            // Update active nav item
            docsNavItems.forEach(item => item.classList.remove('active'));
            navItem.classList.add('active');

            // Show corresponding section
            docsSections.forEach(section => {
                if (section.id === `docs-${sectionId}`) {
                    section.classList.add('active');
                } else {
                    section.classList.remove('active');
                }
            });

            // Scroll to top of main content
            const docsMain = document.querySelector('.docs-main');
            if (docsMain) {
                docsMain.scrollTop = 0;
            }
        });
    });

    // Copy code functionality
    const copyButtons = document.querySelectorAll('.terminal-copy');
    copyButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const codeToCopy = button.getAttribute('data-copy');

            if (codeToCopy) {
                try {
                    await navigator.clipboard.writeText(codeToCopy);

                    // Visual feedback
                    const originalText = button.textContent;
                    button.textContent = 'Copied!';
                    button.style.background = 'var(--accent-primary)';
                    button.style.color = 'white';

                    setTimeout(() => {
                        button.textContent = originalText;
                        button.style.background = '';
                        button.style.color = '';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
            }
        });
    });

    // Load dynamic content when opening docs
    const originalOpenDocs = openDocs;
    openDocs = async function() {
        originalOpenDocs();
        await loadDynamicContent();
    };

    // Expose open function globally for navigation links
    window.openDocs = openDocs;
}

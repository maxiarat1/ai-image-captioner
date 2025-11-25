// Workflow Controls UI: save/load/export/import buttons and dropdown
(function() {
    const NEWorkflowControls = {};

    /**
     * Setup workflow control buttons and dropdown
     */
    NEWorkflowControls.setup = function() {
        const wrapper = document.querySelector('.workflow-select-wrapper');
        const selectBtn = document.getElementById('workflowSelectBtn');
        const dropdown = document.querySelector('.workflow-dropdown');
        const exportBtn = document.getElementById('exportWorkflowBtn');
        const importBtn = document.getElementById('importWorkflowBtn');
        const fileInput = document.getElementById('workflowFileInput');

        if (!selectBtn || !dropdown) {
            console.warn('Workflow controls not found in DOM');
            return;
        }

        // Toggle dropdown on button click
        selectBtn.onclick = (e) => {
            e.stopPropagation();
            const isOpen = wrapper.classList.contains('open');
            NEWorkflowControls.closeDropdown();
            if (!isOpen) {
                NEWorkflowControls.openDropdown();
            }
        };

        // Prevent dropdown from closing when clicking inside
        dropdown.onclick = (e) => {
            e.stopPropagation();
        };

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            NEWorkflowControls.closeDropdown();
        });

        // Export button
        if (exportBtn) {
            exportBtn.onclick = NEWorkflowControls.exportToFile;
        }

        // Import button
        if (importBtn && fileInput) {
            importBtn.onclick = () => fileInput.click();
            fileInput.onchange = NEWorkflowControls.importFromFile;
        }

        // Setup action buttons in dropdown
        const saveBtn = document.getElementById('saveWorkflowBtn');
        const revertBtn = document.getElementById('revertWorkflowBtn');
        const newBtn = document.getElementById('newWorkflowBtn');

        if (saveBtn) {
            saveBtn.onclick = () => {
                NEWorkflowControls.closeDropdown();
                NEWorkflowControls.saveWorkflow();
            };
        }

        if (revertBtn) {
            revertBtn.onclick = () => {
                NEWorkflowControls.closeDropdown();
                NEWorkflowControls.revertWorkflow();
            };
        }

        if (newBtn) {
            newBtn.onclick = () => {
                NEWorkflowControls.closeDropdown();
                NEWorkflowControls.newWorkflow();
            };
        }

        // Listen for workflow events to update UI
        window.addEventListener('workflowSaved', NEWorkflowControls.updateDisplay);
        window.addEventListener('workflowLoaded', NEWorkflowControls.updateDisplay);
        window.addEventListener('workflowDeleted', NEWorkflowControls.updateDisplay);
        window.addEventListener('workflowNew', NEWorkflowControls.updateDisplay);
        window.addEventListener('workflowChanged', NEWorkflowControls.updateUnsavedIndicator);

        // Initial display update
        NEWorkflowControls.updateDisplay();
    };

    /**
     * Open the workflow dropdown
     */
    NEWorkflowControls.openDropdown = function() {
        const wrapper = document.querySelector('.workflow-select-wrapper');
        if (wrapper) {
            wrapper.classList.add('open');
            NEWorkflowControls.populateWorkflowList();
        }
    };

    /**
     * Close the workflow dropdown
     */
    NEWorkflowControls.closeDropdown = function() {
        const wrapper = document.querySelector('.workflow-select-wrapper');
        if (wrapper) {
            wrapper.classList.remove('open');
        }
    };

    /**
     * Populate the workflow list in dropdown
     */
    NEWorkflowControls.populateWorkflowList = function() {
        const list = document.querySelector('.workflow-list');
        if (!list || typeof NEPersistence === 'undefined') return;

        const workflows = NEPersistence.getWorkflowList();
        const currentName = NEPersistence.getCurrentWorkflowName();

        if (workflows.length === 0) {
            list.innerHTML = '';
            return;
        }

        list.innerHTML = workflows.map(name => {
            const isActive = name === currentName;
            // Check if workflow has a draft (unsaved changes)
            // For current workflow, use hasUnsavedChanges; for others, check if draft exists
            const isUnsaved = isActive
                ? NEPersistence.hasUnsavedChanges()
                : NEPersistence.hasDraft(name);
            return `
                <div class="workflow-item ${isActive ? 'active' : ''}" data-name="${name}">
                    <span class="workflow-item-name ${isUnsaved ? 'unsaved' : ''}">${name}</span>
                    <button class="workflow-item-delete" data-name="${name}" title="Delete workflow">×</button>
                </div>
            `;
        }).join('');

        // Attach click handlers
        list.querySelectorAll('.workflow-item').forEach(item => {
            const name = item.dataset.name;

            // Load workflow on click
            item.onclick = (e) => {
                if (e.target.classList.contains('workflow-item-delete')) return;
                NEWorkflowControls.loadWorkflow(name);
            };
        });

        // Attach delete handlers
        list.querySelectorAll('.workflow-item-delete').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                NEWorkflowControls.deleteWorkflow(btn.dataset.name);
            };
        });
    };

    /**
     * Update the workflow name display
     */
    NEWorkflowControls.updateDisplay = function() {
        const nameSpan = document.querySelector('.workflow-select-name');
        if (!nameSpan || typeof NEPersistence === 'undefined') return;

        const currentName = NEPersistence.getCurrentWorkflowName();
        nameSpan.textContent = currentName;

        // Update unsaved indicator
        NEWorkflowControls.updateUnsavedIndicator();
    };

    /**
     * Update the unsaved changes indicator
     */
    NEWorkflowControls.updateUnsavedIndicator = function() {
        const nameSpan = document.querySelector('.workflow-select-name');
        if (!nameSpan || typeof NEPersistence === 'undefined') return;

        if (NEPersistence.hasUnsavedChanges()) {
            nameSpan.classList.add('unsaved');
        } else {
            nameSpan.classList.remove('unsaved');
        }
    };

    /**
     * Save current workflow
     */
    NEWorkflowControls.saveWorkflow = function() {
        if (typeof NEPersistence === 'undefined') return;

        const currentName = NEPersistence.getCurrentWorkflowName();

        // If untitled, prompt for name
        if (currentName === 'Untitled') {
            NEWorkflowControls.saveWorkflowAs();
            return;
        }

        NEPersistence.save();
    };

    /**
     * Save workflow with a new name
     */
    NEWorkflowControls.saveWorkflowAs = function() {
        if (typeof NEPersistence === 'undefined') return;

        const currentName = NEPersistence.getCurrentWorkflowName();
        const defaultName = currentName === 'Untitled' ? '' : currentName;

        const name = prompt('Enter workflow name:', defaultName);
        if (!name || name.trim() === '') return;

        // Check if name exists
        const workflows = NEPersistence.getWorkflowList();
        if (workflows.includes(name.trim()) && name.trim() !== currentName) {
            if (!confirm(`Workflow "${name}" already exists. Overwrite?`)) {
                return;
            }
        }

        NEPersistence.saveAs(name.trim());
    };

    /**
     * Revert current workflow to last saved state
     * Discards all unsaved changes
     */
    NEWorkflowControls.revertWorkflow = function() {
        if (typeof NEPersistence === 'undefined') return;

        // Check if there are unsaved changes
        if (!NEPersistence.hasUnsavedChanges()) {
            alert('No unsaved changes to revert.');
            return;
        }

        // Confirm with user before reverting
        if (!confirm('Revert to last saved state? All unsaved changes will be lost.')) {
            return;
        }

        const currentName = NEPersistence.getCurrentWorkflowName();

        // Clear any session draft for this workflow
        NEPersistence.clearDraft(currentName);

        // Reload the workflow from permanent storage
        NEPersistence.loadByName(currentName);

        console.log(`Workflow "${currentName}" reverted to last saved state`);
    };

    /**
     * Load a workflow by name
     * No warning needed - drafts are preserved when switching workflows
     */
    NEWorkflowControls.loadWorkflow = function(name) {
        if (typeof NEPersistence === 'undefined') return;

        NEWorkflowControls.closeDropdown();
        NEPersistence.loadByName(name);
    };

    /**
     * Delete a workflow
     */
    NEWorkflowControls.deleteWorkflow = function(name) {
        if (typeof NEPersistence === 'undefined') return;

        if (!confirm(`Delete workflow "${name}"?`)) {
            return;
        }

        NEPersistence.deleteByName(name);
        NEWorkflowControls.populateWorkflowList();
    };

    /**
     * Delete the current workflow
     */
    NEWorkflowControls.deleteCurrentWorkflow = function() {
        if (typeof NEPersistence === 'undefined') return;

        const currentName = NEPersistence.getCurrentWorkflowName();

        // Check if workflow is saved (exists in storage)
        const workflows = NEPersistence.getWorkflowList();
        if (!workflows.includes(currentName)) {
            alert('This workflow has not been saved yet. Nothing to delete.');
            return;
        }

        if (!confirm(`Delete workflow "${currentName}"? This cannot be undone.`)) {
            return;
        }

        NEPersistence.deleteByName(currentName);

        // Load another workflow or create new one
        const remainingWorkflows = NEPersistence.getWorkflowList();
        if (remainingWorkflows.length > 0) {
            NEPersistence.loadByName(remainingWorkflows[0]);
        } else {
            NEPersistence.newWorkflow('Untitled');
        }

        NEWorkflowControls.updateDisplay();
    };

    /**
     * Create a new workflow
     */
    NEWorkflowControls.newWorkflow = function() {
        if (typeof NEPersistence === 'undefined') return;

        // Warn about unsaved changes
        if (NEPersistence.hasUnsavedChanges()) {
            if (!confirm('You have unsaved changes. Create new workflow anyway?')) {
                return;
            }
        }

        const name = prompt('Enter name for new workflow:', 'Untitled');
        if (name === null) return; // Cancelled

        NEPersistence.newWorkflow(name || 'Untitled');
    };

    /**
     * Export current workflow to JSON file
     */
    NEWorkflowControls.exportToFile = function() {
        if (typeof NEPersistence === 'undefined') return;

        const json = NEPersistence.exportJSON();
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;

        // Use workflow name for filename
        const name = NEPersistence.getCurrentWorkflowName() || 'workflow';
        const safeName = name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        a.download = `${safeName}.json`;

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log('Workflow exported to file');
    };

    /**
     * Import workflow from JSON file
     */
    NEWorkflowControls.importFromFile = function(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Warn about unsaved changes
        if (typeof NEPersistence !== 'undefined' && NEPersistence.hasUnsavedChanges()) {
            if (!confirm('You have unsaved changes. Import anyway?')) {
                e.target.value = '';
                return;
            }
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                if (typeof NEPersistence !== 'undefined') {
                    const success = NEPersistence.importJSON(event.target.result);
                    if (success) {
                        // Use filename (without extension) as workflow name
                        const name = file.name.replace(/\.json$/i, '');
                        NEPersistence.setCurrentWorkflowName(name);
                        NEWorkflowControls.updateDisplay();
                        console.log(`Workflow imported from ${file.name}`);
                    } else {
                        alert('Failed to import workflow. Invalid format.');
                    }
                }
            } catch (err) {
                console.error('Failed to import workflow:', err);
                alert('Failed to import workflow. Please check the file format.');
            }
        };

        reader.onerror = () => {
            console.error('Failed to read file');
            alert('Failed to read file.');
        };

        reader.readAsText(file);

        // Reset file input so same file can be selected again
        e.target.value = '';
    };

    // Expose to window
    window.NEWorkflowControls = NEWorkflowControls;
})();

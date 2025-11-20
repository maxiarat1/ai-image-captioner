// Workflow persistence: save/load to localStorage
(function() {
    const NEPersistence = {};
    const STORAGE_KEY = 'nodeEditorWorkflow'; // Legacy key for migration
    const WORKFLOWS_KEY = 'nodeEditorWorkflows';
    const CURRENT_KEY = 'nodeEditorCurrentWorkflow';
    const SESSION_DRAFT_KEY = 'nodeEditorSessionDraft'; // sessionStorage key for drafts
    const SESSION_ID_KEY = 'nodeEditorSessionId'; // To detect fresh app launch
    const AUTOSAVE_DELAY = 1000; // ms
    let saveTimeout = null;
    let currentWorkflowName = 'Untitled';
    let hasUnsavedChanges = false;

    // Generate a unique session ID to detect fresh app launches
    const currentSessionId = Date.now().toString() + Math.random().toString(36).substr(2, 9);

    /**
     * Check if this is a fresh app launch (new browser session) or page refresh
     */
    function isFreshLaunch() {
        const storedSessionId = sessionStorage.getItem(SESSION_ID_KEY);
        if (!storedSessionId) {
            // First time in this session, mark the session
            sessionStorage.setItem(SESSION_ID_KEY, currentSessionId);
            return true;
        }
        return false;
    }

    // Check fresh launch on module load
    const freshLaunch = isFreshLaunch();

    /**
     * Migrate from old single-workflow storage to multi-workflow
     */
    function migrateOldStorage() {
        const oldData = localStorage.getItem(STORAGE_KEY);
        if (oldData && !localStorage.getItem(WORKFLOWS_KEY)) {
            try {
                const workflow = JSON.parse(oldData);
                const workflows = { 'Default Workflow': workflow };
                localStorage.setItem(WORKFLOWS_KEY, JSON.stringify(workflows));
                localStorage.setItem(CURRENT_KEY, 'Default Workflow');
                localStorage.removeItem(STORAGE_KEY);
                console.log('Migrated old workflow to new multi-workflow storage');
            } catch (e) {
                console.warn('Failed to migrate old workflow:', e);
            }
        }
    }

    // Run migration on load
    migrateOldStorage();

    /**
     * Serialize current workflow state to JSON-compatible object
     */
    NEPersistence.serialize = function() {
        return {
            version: 1,
            nodes: NodeEditor.nodes.map(n => ({
                id: n.id,
                type: n.type,
                x: n.x,
                y: n.y,
                label: n.label || '',
                data: JSON.parse(JSON.stringify(n.data || {}))
            })),
            connections: NodeEditor.connections.map(c => ({
                id: c.id,
                from: c.from,
                fromPort: c.fromPort,
                to: c.to,
                toPort: c.toPort
            })),
            nextId: NodeEditor.nextId,
            transform: {
                x: NodeEditor.transform.x,
                y: NodeEditor.transform.y,
                scale: NodeEditor.transform.scale
            },
            settings: { ...NodeEditor.settings }
        };
    };

    /**
     * Save workflow to localStorage (saves to current workflow name)
     */
    NEPersistence.save = function() {
        try {
            const workflow = NEPersistence.serialize();
            const workflows = NEPersistence.getAllWorkflows();
            workflows[currentWorkflowName] = workflow;
            localStorage.setItem(WORKFLOWS_KEY, JSON.stringify(workflows));
            localStorage.setItem(CURRENT_KEY, currentWorkflowName);
            hasUnsavedChanges = false;

            // Clear the session draft for this workflow since we've saved permanently
            NEPersistence.clearDraft(currentWorkflowName);

            console.log(`Workflow "${currentWorkflowName}" saved to localStorage`);

            // Dispatch event for UI updates
            window.dispatchEvent(new CustomEvent('workflowSaved', {
                detail: { name: currentWorkflowName }
            }));

            return true;
        } catch (e) {
            console.error('Failed to save workflow:', e);
            return false;
        }
    };

    /**
     * Save current state as a session draft (to sessionStorage)
     * This persists during the session but not across app relaunches
     * Each workflow has its own draft slot
     */
    NEPersistence.saveDraft = function() {
        try {
            const drafts = NEPersistence.getAllDrafts();
            drafts[currentWorkflowName] = NEPersistence.serialize();
            sessionStorage.setItem(SESSION_DRAFT_KEY, JSON.stringify(drafts));
            hasUnsavedChanges = true;
            console.log(`Draft saved for "${currentWorkflowName}"`);
            return true;
        } catch (e) {
            console.error('Failed to save draft:', e);
            return false;
        }
    };

    /**
     * Get all session drafts from sessionStorage
     */
    NEPersistence.getAllDrafts = function() {
        try {
            const draftData = sessionStorage.getItem(SESSION_DRAFT_KEY);
            return draftData ? JSON.parse(draftData) : {};
        } catch (e) {
            console.error('Failed to get drafts:', e);
            return {};
        }
    };

    /**
     * Load session draft for a specific workflow
     * @param {string} name - Workflow name to load draft for
     */
    NEPersistence.loadDraft = function(name) {
        try {
            const drafts = NEPersistence.getAllDrafts();
            return drafts[name] || null;
        } catch (e) {
            console.error('Failed to load draft:', e);
            return null;
        }
    };

    /**
     * Check if a workflow has a draft
     * @param {string} name - Workflow name to check
     */
    NEPersistence.hasDraft = function(name) {
        const drafts = NEPersistence.getAllDrafts();
        return !!drafts[name];
    };

    /**
     * Clear session draft for a specific workflow (or all if no name provided)
     * @param {string} [name] - Workflow name to clear draft for, or all if omitted
     */
    NEPersistence.clearDraft = function(name) {
        if (!name) {
            sessionStorage.removeItem(SESSION_DRAFT_KEY);
            return;
        }

        const drafts = NEPersistence.getAllDrafts();
        if (drafts[name]) {
            delete drafts[name];
            sessionStorage.setItem(SESSION_DRAFT_KEY, JSON.stringify(drafts));
        }
    };

    /**
     * Check if this was a fresh app launch
     */
    NEPersistence.isFreshLaunch = function() {
        return freshLaunch;
    };

    /**
     * Get all saved workflows
     */
    NEPersistence.getAllWorkflows = function() {
        try {
            const data = localStorage.getItem(WORKFLOWS_KEY);
            return data ? JSON.parse(data) : {};
        } catch (e) {
            console.error('Failed to get workflows:', e);
            return {};
        }
    };

    /**
     * Get list of workflow names
     */
    NEPersistence.getWorkflowList = function() {
        return Object.keys(NEPersistence.getAllWorkflows()).sort();
    };

    /**
     * Get current workflow name
     */
    NEPersistence.getCurrentWorkflowName = function() {
        return currentWorkflowName;
    };

    /**
     * Set current workflow name
     */
    NEPersistence.setCurrentWorkflowName = function(name) {
        currentWorkflowName = name;
        localStorage.setItem(CURRENT_KEY, name);
    };

    /**
     * Check if there are unsaved changes
     */
    NEPersistence.hasUnsavedChanges = function() {
        return hasUnsavedChanges;
    };

    /**
     * Mark as having unsaved changes
     */
    NEPersistence.markUnsaved = function() {
        hasUnsavedChanges = true;
        window.dispatchEvent(new CustomEvent('workflowChanged'));
    };

    /**
     * Save workflow with a specific name
     */
    NEPersistence.saveAs = function(name) {
        if (!name || name.trim() === '') {
            console.error('Invalid workflow name');
            return false;
        }

        const oldName = currentWorkflowName;
        currentWorkflowName = name.trim();
        const result = NEPersistence.save();

        if (!result) {
            currentWorkflowName = oldName;
        }

        return result;
    };

    /**
     * Load workflow by name
     * Loads from session draft if exists, otherwise from permanent save
     */
    NEPersistence.loadByName = function(name) {
        // Check for a session draft first
        const draft = NEPersistence.loadDraft(name);
        if (draft) {
            console.log(`Loading from session draft for "${name}"`);
            const result = NEPersistence.restore(draft);
            if (result) {
                currentWorkflowName = name;
                localStorage.setItem(CURRENT_KEY, name);
                hasUnsavedChanges = true; // Draft means unsaved changes

                window.dispatchEvent(new CustomEvent('workflowLoaded', {
                    detail: { name: name }
                }));
            }
            return result;
        }

        // No draft, load from permanent storage
        const workflows = NEPersistence.getAllWorkflows();
        const workflow = workflows[name];

        if (!workflow) {
            console.error(`Workflow "${name}" not found`);
            return false;
        }

        const result = NEPersistence.restore(workflow);
        if (result) {
            currentWorkflowName = name;
            localStorage.setItem(CURRENT_KEY, name);
            hasUnsavedChanges = false;

            window.dispatchEvent(new CustomEvent('workflowLoaded', {
                detail: { name: name }
            }));
        }

        return result;
    };

    /**
     * Delete workflow by name
     */
    NEPersistence.deleteByName = function(name) {
        const workflows = NEPersistence.getAllWorkflows();

        if (!workflows[name]) {
            console.error(`Workflow "${name}" not found`);
            return false;
        }

        delete workflows[name];
        localStorage.setItem(WORKFLOWS_KEY, JSON.stringify(workflows));

        // Also delete any session draft for this workflow
        NEPersistence.clearDraft(name);

        // If we deleted the current workflow, reset to Untitled
        if (name === currentWorkflowName) {
            currentWorkflowName = 'Untitled';
            localStorage.setItem(CURRENT_KEY, 'Untitled');
        }

        window.dispatchEvent(new CustomEvent('workflowDeleted', {
            detail: { name: name }
        }));

        console.log(`Workflow "${name}" deleted`);
        return true;
    };

    /**
     * Rename workflow
     */
    NEPersistence.rename = function(oldName, newName) {
        if (!newName || newName.trim() === '') {
            console.error('Invalid new workflow name');
            return false;
        }

        const workflows = NEPersistence.getAllWorkflows();

        if (!workflows[oldName]) {
            console.error(`Workflow "${oldName}" not found`);
            return false;
        }

        if (workflows[newName.trim()]) {
            console.error(`Workflow "${newName}" already exists`);
            return false;
        }

        workflows[newName.trim()] = workflows[oldName];
        delete workflows[oldName];
        localStorage.setItem(WORKFLOWS_KEY, JSON.stringify(workflows));

        // Update current name if we renamed the active workflow
        if (oldName === currentWorkflowName) {
            currentWorkflowName = newName.trim();
            localStorage.setItem(CURRENT_KEY, currentWorkflowName);
        }

        console.log(`Workflow renamed from "${oldName}" to "${newName}"`);
        return true;
    };

    /**
     * Create new empty workflow
     */
    NEPersistence.newWorkflow = function(name) {
        currentWorkflowName = name || 'Untitled';
        hasUnsavedChanges = false;

        // Clear the canvas
        if (typeof NEGraphOps !== 'undefined' && typeof NEGraphOps.clearGraph === 'function') {
            // Bypass confirmation for programmatic clear
            NodeEditor.nodes.forEach(n => {
                const el = document.getElementById('node-' + n.id);
                if (el) el.remove();
            });
            NodeEditor.connections.forEach(c => {
                const path = document.getElementById('conn-' + c.id);
                if (path) path.remove();
            });
            NodeEditor.nodes = [];
            NodeEditor.connections = [];
            if (typeof NEConnections !== 'undefined') NEConnections.updateConnections();
            if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
        }

        // Save the new workflow to localStorage so it appears in the list
        NEPersistence.save();

        window.dispatchEvent(new CustomEvent('workflowNew', {
            detail: { name: currentWorkflowName }
        }));
    };

    /**
     * Schedule auto-save of draft with debounce
     * This saves to sessionStorage (session draft) not localStorage (permanent save)
     */
    NEPersistence.scheduleSave = function() {
        if (saveTimeout) clearTimeout(saveTimeout);
        hasUnsavedChanges = true;
        window.dispatchEvent(new CustomEvent('workflowChanged'));
        // Save to session draft, not permanent storage
        saveTimeout = setTimeout(NEPersistence.saveDraft, AUTOSAVE_DELAY);
    };

    /**
     * Load workflow from storage
     * - On fresh app launch: loads from localStorage (permanent save), clears all drafts
     * - On page refresh: loads from sessionStorage draft if exists for that workflow
     * @returns {Object|null} Workflow object or null if not found/invalid
     */
    NEPersistence.load = function() {
        try {
            // Get the last used workflow name
            const savedName = localStorage.getItem(CURRENT_KEY);
            if (savedName) {
                currentWorkflowName = savedName;
            }

            // On fresh launch, clear all drafts
            if (freshLaunch) {
                NEPersistence.clearDraft(); // Clear all drafts
                console.log('Fresh app launch - loading from permanent storage');
            } else {
                // Page refresh - check for draft of current workflow
                const draft = NEPersistence.loadDraft(currentWorkflowName);
                if (draft && Array.isArray(draft.nodes)) {
                    console.log(`Loading from session draft for "${currentWorkflowName}" (page refresh)`);
                    hasUnsavedChanges = true; // Draft means unsaved changes
                    return draft;
                }
            }

            // Load from localStorage (permanent save)
            const workflows = NEPersistence.getAllWorkflows();
            const workflow = workflows[currentWorkflowName];

            if (!workflow) {
                // Try to load any available workflow
                const names = Object.keys(workflows);
                if (names.length > 0) {
                    currentWorkflowName = names[0];
                    hasUnsavedChanges = false;
                    return workflows[currentWorkflowName];
                }
                return null;
            }

            // Basic validation
            if (!workflow || !Array.isArray(workflow.nodes)) {
                console.warn('Invalid workflow data in localStorage');
                return null;
            }

            hasUnsavedChanges = false;
            return workflow;
        } catch (e) {
            console.error('Failed to load workflow:', e);
            return null;
        }
    };

    /**
     * Check if any saved workflow exists
     */
    NEPersistence.hasSavedWorkflow = function() {
        const workflows = NEPersistence.getAllWorkflows();
        return Object.keys(workflows).length > 0;
    };

    /**
     * Clear saved workflow from localStorage
     */
    NEPersistence.clear = function() {
        localStorage.removeItem(STORAGE_KEY);
        console.log('Workflow cleared from localStorage');
    };

    /**
     * Restore workflow from saved state
     * @param {Object} workflow - The workflow object to restore
     */
    NEPersistence.restore = function(workflow) {
        if (!workflow || !Array.isArray(workflow.nodes)) {
            console.error('Invalid workflow object');
            return false;
        }

        try {
            // Clear existing nodes from DOM
            NodeEditor.nodes.forEach(n => {
                const el = document.getElementById('node-' + n.id);
                if (el) el.remove();
            });

            // Clear existing connections from DOM
            NodeEditor.connections.forEach(c => {
                const path = document.getElementById('conn-' + c.id);
                if (path) path.remove();
                const gradient = document.getElementById('gradient-' + c.id);
                if (gradient) gradient.remove();
            });

            // Reset state arrays
            NodeEditor.nodes = [];
            NodeEditor.connections = [];

            // Restore nextId (ensure it's at least as high as saved value)
            NodeEditor.nextId = workflow.nextId || 1;

            // Restore transform
            if (workflow.transform) {
                NodeEditor.transform.x = workflow.transform.x || 0;
                NodeEditor.transform.y = workflow.transform.y || 0;
                NodeEditor.transform.scale = workflow.transform.scale || 1;
            }

            // Restore settings (merge with defaults)
            if (workflow.settings) {
                NodeEditor.settings = {
                    ...NodeEditor.settings,
                    ...workflow.settings
                };
            }

            // Restore nodes
            workflow.nodes.forEach(nodeData => {
                const node = {
                    id: nodeData.id,
                    type: nodeData.type,
                    x: nodeData.x,
                    y: nodeData.y,
                    label: nodeData.label || '',
                    data: JSON.parse(JSON.stringify(nodeData.data || {}))
                };
                NodeEditor.nodes.push(node);

                // Render node if NENodes is available
                if (typeof NENodes !== 'undefined' && typeof NENodes.renderNode === 'function') {
                    NENodes.renderNode(node);
                }
            });

            // Restore connections
            workflow.connections.forEach(connData => {
                // Use NEConnections.addConnection which handles validation and rendering
                if (typeof NEConnections !== 'undefined' && typeof NEConnections.addConnection === 'function') {
                    NEConnections.addConnection(
                        connData.from,
                        connData.fromPort,
                        connData.to,
                        connData.toPort
                    );
                }
            });

            // Ensure nextId is high enough after adding connections
            // (addConnection increments nextId)
            if (workflow.nextId && NodeEditor.nextId < workflow.nextId) {
                NodeEditor.nextId = workflow.nextId;
            }

            // Update visuals
            if (typeof NEConnections !== 'undefined' && typeof NEConnections.updateConnections === 'function') {
                NEConnections.updateConnections();
            }
            if (typeof NEMinimap !== 'undefined' && typeof NEMinimap.updateMinimap === 'function') {
                NEMinimap.updateMinimap();
            }

            // Apply transform to canvas
            if (typeof NEViewport !== 'undefined' && typeof NEViewport.setTransform === 'function') {
                NEViewport.setTransform(
                    NodeEditor.transform.x,
                    NodeEditor.transform.y,
                    NodeEditor.transform.scale
                );
            } else {
                // Fallback: manually apply transform
                const canvas = document.getElementById('nodeCanvas');
                if (canvas) {
                    const { x, y, scale } = NodeEditor.transform;
                    canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
                }
            }

            console.log('Workflow restored from localStorage');
            return true;
        } catch (e) {
            console.error('Failed to restore workflow:', e);
            return false;
        }
    };

    /**
     * Export workflow as JSON string (for download/copy)
     */
    NEPersistence.exportJSON = function() {
        return JSON.stringify(NEPersistence.serialize(), null, 2);
    };

    /**
     * Import workflow from JSON string
     * @param {string} jsonString - The JSON string to import
     */
    NEPersistence.importJSON = function(jsonString) {
        try {
            const workflow = JSON.parse(jsonString);
            return NEPersistence.restore(workflow);
        } catch (e) {
            console.error('Failed to import workflow:', e);
            return false;
        }
    };

    // Expose to window
    window.NEPersistence = NEPersistence;
})();

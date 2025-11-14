// Graph execution: Submit to backend and listen for updates via SSE
(function() {
    const NEExec = {};

    let currentJobId = null;
    let eventSource = null;
    let activeChain = [];

    // Execute the graph: validate and submit to backend
    NEExec.executeGraph = async function() {
        // Validate graph
        const inputNode = NodeEditor.nodes.find(n => n.type === 'input');
        const outputNode = NodeEditor.nodes.find(n => n.type === 'output');

        if (!inputNode || !outputNode) {
            showToast('Need Input and Output nodes');
            return;
        }

        if (AppState.uploadQueue.length === 0) {
            showToast('Upload images first');
            return;
        }

        // Validate AI model chain
        const chain = NEExec._buildAiModelChain(inputNode, outputNode);
        if (chain.length === 0) {
            showToast('Add an AI Model or Curate node and connect it to Input');
            return;
        }

        // Warn if no caption-generating nodes (only Curate nodes without VLM)
        const hasAiModel = chain.some(n => {
            if (n.type === 'aimodel') return true;
            // Curate with VLM model can generate routing analysis
            if (n.type === 'curate' && n.data.modelType === 'vlm' && n.data.model) return true;
            return false;
        });
        if (!hasAiModel) {
            showToast('No AI Model in workflow - no captions will be generated (routing only)', 'warning', 4000);
            // Don't return - allow execution, just warn
        }

        // Prepare graph definition
        const graph = {
            nodes: NodeEditor.nodes,
            connections: NodeEditor.connections
        };

        const image_ids = AppState.uploadQueue.map(item => item.image_id).filter(id => id);
        if (image_ids.length === 0) {
            showToast('No valid images to process');
            return;
        }

        try {
            // Clear UI immediately
            NEExec._clearResultsUI();

            // Submit to backend (clears database captions automatically)
            const response = await fetch(`${AppState.apiBaseUrl}/graph/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ graph, image_ids, clear_previous: true })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Execution failed');
            }

            const data = await response.json();
            currentJobId = data.job_id;

            // Store job_id in session storage for resume on refresh
            sessionStorage.setItem('currentJobId', currentJobId);

            // Prepare UI
            NEExec._prepareUI();

            // Build the processing chain to map stages to nodes
            activeChain = NEExec._buildProcessingChain();

            // Start listening to SSE
            NEExec._listenToJobUpdates(currentJobId);

            showToast('Graph execution started');

        } catch (error) {
            console.error('Error starting execution:', error);
            showToast(error.message || 'Failed to start execution');
        }
    };

    // Build AI model chain (same validation logic as before)
    NEExec._buildAiModelChain = function(inputNode, outputNode) {
        const aimodels = NodeEditor.nodes.filter(n => n.type === 'aimodel' || n.type === 'curate');
        if (aimodels.length === 0) return [];

        const hasConn = (fromId, fromPort, toId, toPort) =>
            NodeEditor.connections.some(c =>
                c.from === fromId && c.fromPort === fromPort &&
                c.to === toId && c.toPort === toPort
            );

        // Find first AI: receives images from Input
        const starts = aimodels.filter(n => hasConn(inputNode.id, 0, n.id, 0));
        if (starts.length === 0) return [];

        const isFedByPrevAi = (node) => NodeEditor.connections.some(c => {
            const fromNode = NodeEditor.nodes.find(nn => nn.id === c.from);
            return c.to === node.id && c.toPort === 1 && fromNode && (fromNode.type === 'aimodel' || fromNode.type === 'curate');
        });

        const start = starts.find(n => !isFedByPrevAi(n)) || starts[0];
        const chain = [start];
        const visited = new Set([start.id]);
        let current = start;

        while (true) {
            const outgoing = NodeEditor.connections.filter(c => c.from === current.id && c.fromPort === 0);
            if (!outgoing || outgoing.length === 0) break;

            let nextConn = outgoing.find(c => {
                const toNode = NodeEditor.nodes.find(n => n.id === c.to);
                return toNode && (toNode.type === 'aimodel' || toNode.type === 'curate') && c.toPort === 1;
            });

            if (!nextConn) break;

            const next = NodeEditor.nodes.find(n => n.id === nextConn.to && (n.type === 'aimodel' || n.type === 'curate'));
            if (!next || visited.has(next.id)) break;

            chain.push(next);
            visited.add(next.id);
            current = next;
        }

        return chain;
    };

    // Clear results UI completely
    NEExec._clearResultsUI = function() {
        const resultsGrid = document.getElementById('resultsGrid');
        const downloadBtn = document.getElementById('downloadAllBtn');
        const paginationControls = document.getElementById('paginationControls');

        if (resultsGrid) resultsGrid.innerHTML = '';
        if (paginationControls) paginationControls.style.display = 'none';
        if (downloadBtn) downloadBtn.style.display = 'none';

        AppState.processedResults = [];
        AppState.allResults = [];
        AppState.currentPage = 1;
    };

    // Prepare UI for execution
    NEExec._prepareUI = function() {
        const processingControls = document.getElementById('processingControls');
        if (processingControls) processingControls.style.display = 'flex';
    };

    // Build ordered chain including AI Model and Curate nodes to mirror backend
    NEExec._buildProcessingChain = function() {
        const nodes = NodeEditor.nodes || [];
        const connections = NodeEditor.connections || [];

        // Aim for the same rules used in backend GraphExecutor._build_ai_chain
        const aiNodes = nodes.filter(n => n.type === 'aimodel' || n.type === 'curate');
        const inputNode = nodes.find(n => n.type === 'input');
        if (!inputNode || aiNodes.length === 0) return [];

        const hasConn = (fromId, fromPort, toId, toPort) =>
            connections.some(c => c.from === fromId && c.fromPort === fromPort && c.to === toId && c.toPort === toPort);

        // First AI receives images from Input (port 0 -> port 0)
        const candidates = aiNodes.filter(n => hasConn(inputNode.id, 0, n.id, 0));
        if (candidates.length === 0) return [];

        // Choose a start that's not fed by another AI on prompt port (port 1)
        const isFedByAi = (node) => connections.some(c => {
            if (c.to !== node.id) return false;
            const fromNode = nodes.find(nn => nn.id === c.from);
            return !!fromNode && (fromNode.type === 'aimodel' || fromNode.type === 'curate') && c.toPort === 1;
        });

        const start = candidates.find(n => !isFedByAi(n)) || candidates[0];
        const chain = [start];
        const visited = new Set([start.id]);
        let current = start;

        // Follow chain forward; for curate nodes accept toPort 0 as well
        while (true) {
            const nextConn = connections.find(c => {
                if (c.from !== current.id) return false;
                const toNode = nodes.find(n => n.id === c.to);
                if (!toNode) return false;
                const isAi = toNode.type === 'aimodel' || toNode.type === 'curate';
                const aiPortMatch = (c.toPort === 1 || c.toPort === 0);
                return isAi && aiPortMatch;
            });

            if (!nextConn) break;
            const nextNode = nodes.find(n => n.id === nextConn.to);
            if (!nextNode || visited.has(nextNode.id)) break;

            chain.push(nextNode);
            visited.add(nextNode.id);
            current = nextNode;
        }

        // Return node IDs for quick DOM lookup
        return chain.map(n => n.id);
    };

    // Highlight the node corresponding to the current stage (1-based)
    NEExec._highlightProcessingNode = function(stageIndex) {
        // Clear previous highlights first (cheap and safe)
        document.querySelectorAll('.node.processing').forEach(el => el.classList.remove('processing'));

        if (!activeChain || activeChain.length === 0) return;
        const idx = Math.max(0, Math.min((stageIndex || 1) - 1, activeChain.length - 1));
        const nodeId = activeChain[idx];
        const el = document.getElementById(`node-${nodeId}`);
        if (el) el.classList.add('processing');
    };

    // Clear all processing highlights
    NEExec._clearProcessingHighlights = function() {
        document.querySelectorAll('.node.processing').forEach(el => el.classList.remove('processing'));
    };

    // Listen to job updates via SSE
    NEExec._listenToJobUpdates = function(jobId) {
        // Close existing connection if any
        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource(`${AppState.apiBaseUrl}/graph/status/${jobId}`);

        eventSource.onmessage = (event) => {
            try {
                const status = JSON.parse(event.data);
                NEExec._handleStatusUpdate(status);
            } catch (e) {
                console.error('Error parsing status update:', e);
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            eventSource.close();
            eventSource = null;

            // Check if job completed by fetching final status
            NEExec._checkFinalStatus(jobId);
        };

        // Start polling for new results every 2 seconds
        NEExec._startResultsPolling();
    };

    // Polling interval for loading results during execution
    let resultsPollingInterval = null;

    // Start polling for new results while job is running
    NEExec._startResultsPolling = function() {
        // Clear any existing interval
        if (resultsPollingInterval) {
            clearInterval(resultsPollingInterval);
        }

        // Load results immediately
        NEExec._loadResultsFromDatabase();

        // Poll every 2 seconds
        resultsPollingInterval = setInterval(() => {
            NEExec._loadResultsFromDatabase();
        }, 2000);
    };

    // Stop polling for results
    NEExec._stopResultsPolling = function() {
        if (resultsPollingInterval) {
            clearInterval(resultsPollingInterval);
            resultsPollingInterval = null;
        }
    };

    // Handle status update from SSE
    NEExec._handleStatusUpdate = function(status) {
        const { job_id, status: jobStatus, current_stage, total_stages,
                processed, total, success, failed, progress, error } = status;

        // Update toast with progress
        if (jobStatus === 'running') {
            const stageText = total_stages > 1 ? `Stage ${current_stage}/${total_stages}: ` : '';
            const progressText = `${stageText}${processed}/${total}`;
            const progressPct = processed / total;

            showToast(progressText, true, progressPct);

            // Update output node stats if available
            const outputNode = NodeEditor.nodes.find(n => n.type === 'output');
            if (outputNode && typeof updateOutputStats === 'function') {
                updateOutputStats(outputNode.id, {
                    total,
                    processed,
                    success,
                    failed,
                    stage: total_stages > 1 ? `${current_stage}/${total_stages}` : '',
                    speed: progress?.speed || '',
                    eta: progress?.eta || '',
                    resultsReady: 0
                });
            }

            // Update per-node processing highlight
            NEExec._highlightProcessingNode(current_stage || 1);
        }

        // Handle terminal states
        if (jobStatus === 'completed') {
            NEExec._handleCompletion(job_id);
        } else if (jobStatus === 'failed') {
            NEExec._handleFailure(error);
        } else if (jobStatus === 'cancelled') {
            NEExec._handleCancellation();
        }
    };

    // Handle job completion
    NEExec._handleCompletion = async function(jobId) {
        // Stop polling for results
        NEExec._stopResultsPolling();

        // Close SSE
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        // Hide processing controls
        const processingControls = document.getElementById('processingControls');
        if (processingControls) processingControls.style.display = 'none';

    // Clear any node highlights
    NEExec._clearProcessingHighlights();

        // Final load of results from database
        await NEExec._loadResultsFromDatabase();

        // Update output node to reflect completion and number of results
        try {
            const outputNode = NodeEditor.nodes.find(n => n.type === 'output');
            if (outputNode && typeof updateOutputStats === 'function') {
                const currentStats = outputNode.data.stats || {};
                const resultsCount = Array.isArray(AppState.allResults) ? AppState.allResults.length : 0;
                updateOutputStats(outputNode.id, {
                    // Preserve existing counters where possible
                    total: currentStats.total || 0,
                    processed: currentStats.processed || 0,
                    success: currentStats.success || 0,
                    failed: currentStats.failed || 0,
                    // Clear transient fields
                    stage: '',
                    speed: '',
                    eta: '',
                    // Signal completion via results count so UI hides idle
                    resultsReady: resultsCount
                });
            }
        } catch (e) {
            console.warn('Could not update output stats on completion:', e);
        }

        // Show download button
        const downloadBtn = document.getElementById('downloadAllBtn');
        if (downloadBtn && AppState.allResults.length > 0) {
            downloadBtn.style.display = 'inline-flex';
        }

        // Clear stored job ID
        sessionStorage.removeItem('currentJobId');
        currentJobId = null;

        showToast('Execution completed!');
    };

    // Handle job failure
    NEExec._handleFailure = function(error) {
        // Stop polling for results
        NEExec._stopResultsPolling();

        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        const processingControls = document.getElementById('processingControls');
        if (processingControls) processingControls.style.display = 'none';

    // Clear any node highlights
    NEExec._clearProcessingHighlights();

        sessionStorage.removeItem('currentJobId');
        currentJobId = null;

        showToast(`Execution failed: ${error || 'Unknown error'}`);
    };

    // Handle job cancellation
    NEExec._handleCancellation = function() {
        // Stop polling for results
        NEExec._stopResultsPolling();

        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        const processingControls = document.getElementById('processingControls');
        if (processingControls) processingControls.style.display = 'none';

    // Clear any node highlights
    NEExec._clearProcessingHighlights();

        sessionStorage.removeItem('currentJobId');
        currentJobId = null;

        showToast('Execution cancelled');
    };

    // Check final status (called on SSE error)
    NEExec._checkFinalStatus = async function(jobId) {
        try {
            const response = await fetch(`${AppState.apiBaseUrl}/graph/status/${jobId}`);
            if (response.ok) {
                const status = await response.json();
                if (status.status === 'completed') {
                    NEExec._handleCompletion(jobId);
                }
            }
        } catch (e) {
            console.error('Error checking final status:', e);
        }
    };

    // Load results from database (called during execution and after completion)
    NEExec._loadResultsFromDatabase = async function() {
        try {
            const response = await fetch(`${AppState.apiBaseUrl}/images?page=1&per_page=1000`);
            if (!response.ok) return;

            const data = await response.json();
            const imagesWithCaptions = data.images.filter(img => img.caption);

            // Only update if we have new results
            if (imagesWithCaptions.length === 0) return;

            AppState.allResults = imagesWithCaptions.map(img => ({
                queueItem: img,
                data: { caption: img.caption }
            }));

            AppState.processedResults = imagesWithCaptions.map(img => ({
                filename: img.filename,
                caption: img.caption,
                path: img.filename
            }));

            // Render results
            if (typeof renderCurrentPage === 'function') {
                renderCurrentPage();
            }

            // Show download button if we have results
            const downloadBtn = document.getElementById('downloadAllBtn');
            if (downloadBtn && AppState.allResults.length > 0) {
                downloadBtn.style.display = 'inline-flex';
            }

        } catch (error) {
            console.error('Error loading results:', error);
        }
    };

    // Resume execution on page load if job exists
    NEExec.resumeExecution = function() {
        const jobId = sessionStorage.getItem('currentJobId');
        if (jobId) {
            currentJobId = jobId;
            NEExec._prepareUI();
            // Rebuild chain in case of refresh/resume and let updates drive highlights
            activeChain = NEExec._buildProcessingChain();
            NEExec._listenToJobUpdates(jobId);
            showToast('Resumed execution monitoring', false);
        }
    };

    window.NEExec = NEExec;
    window.executeGraph = NEExec.executeGraph;
})();

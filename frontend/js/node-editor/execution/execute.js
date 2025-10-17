// Graph execution: running the pipeline and processing results
(function() {
    const NEExec = {};

    // Execute the graph: validate nodes/connections and start processing
    NEExec.executeGraph = async function() {
        // Find nodes
        const inputNode = NodeEditor.nodes.find(n => n.type === 'input');
        const aiNode = NodeEditor.nodes.find(n => n.type === 'aimodel');
        const outputNode = NodeEditor.nodes.find(n => n.type === 'output');

        if (!inputNode || !aiNode || !outputNode) {
            showToast('Need Input, AI Model, and Output nodes');
            return;
        }

        // Check connections
        const hasInput = NodeEditor.connections.some(c => c.to === aiNode.id);
        const hasOutput = NodeEditor.connections.some(c => c.from === aiNode.id);

        if (!hasInput || !hasOutput) {
            showToast('AI Model must be connected');
            return;
        }

        if (AppState.uploadQueue.length === 0) {
            showToast('Upload images first');
            return;
        }

        // Switch to results
        const resultsTabBtn = document.querySelector('.tab-btn[data-tab="results"]');
        if (resultsTabBtn) resultsTabBtn.click();

        // Process
        await NEExec.processGraph(aiNode);
    };

    // Process the graph given the AI model node
    NEExec.processGraph = async function(aiNode) {
        const resultsGrid = document.getElementById('resultsGrid');
        const downloadBtn = document.getElementById('downloadAllBtn');
        const processingControls = document.getElementById('processingControls');
        const paginationControls = document.getElementById('paginationControls');

        if (resultsGrid) resultsGrid.innerHTML = '';
        if (paginationControls) paginationControls.style.display = 'none';
        if (downloadBtn) downloadBtn.style.display = 'none';
        if (processingControls) processingControls.style.display = 'flex';

        AppState.processedResults = [];
        AppState.allResults = [];
        AppState.currentPage = 1;
        isProcessing = true;

        const total = AppState.uploadQueue.length;
        let count = 0;

        // Build prompt if connected (from prompt or conjunction node)
        let prompt = '';

        // Check for direct prompt connection
        const promptNode = NodeEditor.nodes.find(n => n.type === 'prompt');
        const hasPromptConn = promptNode && NodeEditor.connections.some(c =>
            c.from === promptNode.id && c.to === aiNode.id
        );
        if (hasPromptConn) {
            prompt = promptNode.data.text || '';
        }

        // Check for conjunction node connection (takes precedence)
        const conjunctionNode = NodeEditor.nodes.find(n => n.type === 'conjunction');
        const hasConjunctionConn = conjunctionNode && NodeEditor.connections.some(c =>
            c.from === conjunctionNode.id && c.to === aiNode.id
        );
        if (hasConjunctionConn) {
            // Use the template and resolve placeholders
            let template = conjunctionNode.data.template || '';
            const items = conjunctionNode.data.connectedItems || [];

            // Create a map of reference keys to content
            const refMap = {};
            items.forEach(item => {
                refMap[item.refKey] = item.content;
            });

            // Replace all placeholders with actual content
            prompt = template.replace(/\{([^}]+)\}/g, (match, key) => {
                return refMap[key] !== undefined ? refMap[key] : match;
            });
        }

        // Process images
        for (const item of AppState.uploadQueue) {
            if (shouldStop) break;
            while (isPaused && !shouldStop) {
                await new Promise(r => setTimeout(r, 100));
            }

            const formData = new FormData();
            if (item.file) formData.append('image', item.file);
            else if (item.path) formData.append('image_path', item.path);

            formData.append('model', aiNode.data.model);
            formData.append('parameters', JSON.stringify(aiNode.data.parameters || {}));
            if (prompt) formData.append('prompt', prompt);

            try {
                const res = await fetch(`${AppState.apiBaseUrl}/generate`, {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) throw new Error(res.statusText);

                const data = await res.json();
                AppState.allResults.push({ queueItem: item, data });
                AppState.processedResults.push({
                    filename: item.filename,
                    caption: data.caption,
                    path: item.path || item.filename
                });

                await addResultItemToCurrentPage(item, data);
            } catch (err) {
                console.error(err);
            }

            count++;
            showToast(`Processed ${count}/${total}`, true, count / total);
        }

        shouldStop = false;
        isPaused = false;
        isProcessing = false;
        if (processingControls) processingControls.style.display = 'none';

        if (AppState.processedResults.length > 0) {
            if (downloadBtn) downloadBtn.style.display = 'inline-flex';
            showToast('Done!');
        }
    };

    window.NEExec = NEExec;
    // Back-compat globals
    window.executeGraph = NEExec.executeGraph;
    window.processGraph = NEExec.processGraph;
})();

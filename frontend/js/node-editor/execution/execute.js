// Graph execution: running the pipeline and processing results
(function() {
    const NEExec = {};

    // Execute the graph: validate nodes/connections and start processing
    NEExec.executeGraph = async function() {
        // Find core nodes
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

        // Build AI model chain (aimodel -> aimodel ...)
        const chain = NEExec._buildAiModelChain(inputNode, outputNode);
        if (chain.length === 0) {
            showToast('Add an AI Model node and connect it to Input');
            return;
        }

        // Optional: ensure last AI model connects to output (best-effort, not hard-fail)
        const last = chain[chain.length - 1];
        const connectedToOutput = NodeEditor.connections.some(c => c.from === last.id && c.to === outputNode.id);
        if (!connectedToOutput) {
            // Non-blocking warning to keep UX simple
            console.warn('Last AI Model is not connected to Output node');
        }

        // Process sequentially across the chain
        await NEExec.processGraph(chain);
    };

    // Build a sequential chain of AI Model nodes starting from the one fed by Input images
    // Rules:
    // - First node has an incoming connection from Input node to its 'images' input (port 0)
    // - Next nodes are connected via previous 'captions' output (port 0) -> next 'prompt' input (port 1)
    NEExec._buildAiModelChain = function(inputNode, outputNode) {
        const aimodels = NodeEditor.nodes.filter(n => n.type === 'aimodel');
        if (aimodels.length === 0) return [];

        // Helper to test a connection
        const hasConn = (fromId, fromPort, toId, toPort) =>
            NodeEditor.connections.some(c => c.from === fromId && c.fromPort === fromPort && c.to === toId && c.toPort === toPort);

        // Find candidate starts: aimodels that receive images from Input on port 0
        const starts = aimodels.filter(n => hasConn(inputNode.id, 0, n.id, 0));
        if (starts.length === 0) return [];

        // If multiple, pick the one that is NOT fed by another aimodel on its prompt
        const isFedByPrevAi = (node) => NodeEditor.connections.some(c => {
            const fromNode = NodeEditor.nodes.find(nn => nn.id === c.from);
            return c.to === node.id && c.toPort === 1 && fromNode && fromNode.type === 'aimodel';
        });
        const start = starts.find(n => !isFedByPrevAi(n)) || starts[0];

        // Follow chain forward via captions (0) -> prompt (1)
        const chain = [start];
        let current = start;
        while (true) {
            // Find connection from current captions to another aimodel prompt
            const link = NodeEditor.connections.find(c => c.from === current.id && c.fromPort === 0);
            if (!link) break;
            const next = NodeEditor.nodes.find(n => n.id === link.to && n.type === 'aimodel' && link.toPort === 1);
            if (!next) break;
            chain.push(next);
            current = next;
        }
        return chain;
    };

    // Process the graph given a chain of AI model nodes (sequential by stage)
    // Back-compat: if a single node is passed, wrap into an array
    NEExec.processGraph = async function(aiChainOrNode) {
        const aiChain = Array.isArray(aiChainOrNode) ? aiChainOrNode : [aiChainOrNode];
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

        const totalImages = AppState.uploadQueue.length;
        let stageIndex = 0;
        // Holds per-image prompt for next stage (captions from previous stage)
        let prevCaptions = new Array(totalImages).fill('');

        // Find output node and reset its stats
        const outputNode = NodeEditor.nodes.find(n => n.type === 'output');
        if (outputNode && typeof resetOutputStats === 'function') {
            resetOutputStats(outputNode.id);
        }

        // Track overall statistics
        let totalSuccess = 0;
        let totalFailed = 0;
        let totalProcessed = 0;
        const startTime = Date.now();

        // Stage-by-stage processing: finish all images for stage 0, then stage 1, etc.
        for (const aiNode of aiChain) {
            // Highlight current node as processing
            NEExec._setNodeProcessing(aiNode.id, true);
            let processedInStage = 0;

            // Compute any static prompt connected directly into this AI node
            const basePrompt = NEExec._buildPromptForNode(aiNode);

            for (let idx = 0; idx < AppState.uploadQueue.length; idx++) {
                if (shouldStop) break;
                while (isPaused && !shouldStop) {
                    await new Promise(r => setTimeout(r, 100));
                }

                const item = AppState.uploadQueue[idx];

                const formData = new FormData();
                if (item.file) formData.append('image', item.file);
                else if (item.path) formData.append('image_path', item.path);

                formData.append('model', aiNode.data.model);
                formData.append('parameters', JSON.stringify(aiNode.data.parameters || {}));

                // Determine prompt for this request:
                // - If this node is fed by a previous AI node, use previous caption as prompt
                // - Otherwise, use basePrompt (from prompt/conjunction)
                const fedByPrevAi = NodeEditor.connections.some(c => {
                    const fromNode = NodeEditor.nodes.find(nn => nn.id === c.from);
                    return c.to === aiNode.id && c.toPort === 1 && fromNode && fromNode.type === 'aimodel';
                });
                let promptToUse = '';
                if (fedByPrevAi) {
                    promptToUse = prevCaptions[idx] || '';
                } else {
                    promptToUse = basePrompt || '';
                }
                if (promptToUse) formData.append('prompt', promptToUse);

                let requestSuccess = false;
                try {
                    const res = await fetch(`${AppState.apiBaseUrl}/generate`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!res.ok) throw new Error(res.statusText);

                    const data = await res.json();
                    requestSuccess = true;

                    // Store caption for next stage or final results
                    prevCaptions[idx] = data.caption || '';

                    // If last stage, record/show results
                    const isLastStage = (stageIndex === aiChain.length - 1);
                    if (isLastStage) {
                        // Check if there's a Conjunction node between last AI model and Output
                        const conjunctionNode = NEExec._findOutputConjunction(aiNode, outputNode);
                        let finalCaption = data.caption;

                        if (conjunctionNode) {
                            // Resolve conjunction template with AI model outputs
                            finalCaption = NEExec._resolveConjunctionForImage(conjunctionNode, prevCaptions, idx);
                        }

                        AppState.allResults.push({ queueItem: item, data: { ...data, caption: finalCaption } });
                        AppState.processedResults.push({
                            filename: item.filename,
                            caption: finalCaption,
                            path: item.path || item.filename
                        });
                        await addResultItemToCurrentPage(item, { ...data, caption: finalCaption });
                        totalSuccess++;
                    }
                } catch (err) {
                    console.error(err);
                    totalFailed++;
                }

                processedInStage++;
                totalProcessed++;
                const progress = processedInStage / totalImages;
                showToast(`Stage ${stageIndex + 1}/${aiChain.length}: ${processedInStage}/${totalImages}`, true, progress);

                // Update Output node stats
                if (outputNode && typeof updateOutputStats === 'function') {
                    const elapsed = (Date.now() - startTime) / 1000; // seconds
                    const avgSpeed = totalProcessed > 0 ? elapsed / totalProcessed : 0;
                    const remaining = (totalImages * aiChain.length) - totalProcessed;
                    const eta = remaining > 0 ? avgSpeed * remaining : 0;

                    // Format speed and ETA
                    const speedText = avgSpeed > 0 ? `~${avgSpeed.toFixed(1)}s/img` : '';
                    const etaText = eta > 0 ? NEExec._formatTime(eta) : '';

                    // Calculate stage display
                    const stageText = aiChain.length > 1 ? `${stageIndex + 1}/${aiChain.length}` : '';

                    updateOutputStats(outputNode.id, {
                        total: totalImages,
                        processed: processedInStage,
                        success: totalSuccess,
                        failed: totalFailed,
                        stage: stageText,
                        speed: speedText,
                        eta: etaText,
                        resultsReady: 0
                    });
                }
            }

            // Prepare for next stage: ensure we keep captions from this stage
            if (stageIndex < aiChain.length - 1) {
                // If some images failed, ensure array length remains consistent
                prevCaptions.length = totalImages;
            }
            // Remove highlight from this node before moving to the next stage
            NEExec._setNodeProcessing(aiNode.id, false);
            stageIndex++;
        }

        shouldStop = false;
        isPaused = false;
        isProcessing = false;
        if (processingControls) processingControls.style.display = 'none';

        // Calculate final statistics
        const totalTime = (Date.now() - startTime) / 1000; // seconds
        const iterationsPerSecond = totalProcessed > 0 ? totalProcessed / totalTime : 0;

        // Update Output node with final results
        if (outputNode && typeof updateOutputStats === 'function') {
            updateOutputStats(outputNode.id, {
                resultsReady: AppState.processedResults.length,
                eta: '',
                speed: '',
                totalTime: NEExec._formatTime(totalTime),
                iterationsPerSecond: iterationsPerSecond.toFixed(2)
            });
        }

        if (AppState.processedResults.length > 0) {
            if (downloadBtn) downloadBtn.style.display = 'inline-flex';
            showToast('Done!');
        }
    };

    // Format time in seconds to human-readable string
    NEExec._formatTime = function(seconds) {
        if (seconds < 60) {
            return `~${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `~${mins}m ${secs}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            return `~${hours}h ${mins}m`;
        }
    };

    // Find if there's a Conjunction node between last AI model and Output
    NEExec._findOutputConjunction = function(lastAiNode, outputNode) {
        // Look for conjunction nodes
        const conjunctionNodes = NodeEditor.nodes.filter(n => n.type === 'conjunction');

        for (const conjNode of conjunctionNodes) {
            // Check if this conjunction receives from the last AI model
            const hasAiInput = NodeEditor.connections.some(c =>
                c.from === lastAiNode.id && c.to === conjNode.id
            );

            // Check if this conjunction connects to the output
            const hasOutputConn = NodeEditor.connections.some(c =>
                c.from === conjNode.id && c.to === outputNode.id
            );

            if (hasAiInput && hasOutputConn) {
                return conjNode;
            }
        }

        return null;
    };

    // Resolve conjunction template for a specific image using AI outputs
    NEExec._resolveConjunctionForImage = function(conjunctionNode, allCaptions, imageIndex) {
        const template = conjunctionNode.data.template || '';
        if (!template) return allCaptions[imageIndex] || '';

        const items = conjunctionNode.data.connectedItems || [];
        const refMap = {};

        // Build reference map with actual AI-generated captions
        items.forEach(item => {
            // If the source is an AI model, use the generated caption for this specific image
            if (item.content === '[Generated Captions]') {
                // Find which AI model this is from
                const sourceNode = NodeEditor.nodes.find(n => n.id === item.sourceId);
                if (sourceNode && sourceNode.type === 'aimodel') {
                    // Use the caption from this image index
                    refMap[item.refKey] = allCaptions[imageIndex] || '';
                }
            } else {
                // Use static content (from Prompt nodes, etc.)
                refMap[item.refKey] = item.content;
            }
        });

        // Replace placeholders with actual content
        const resolved = template.replace(/\{([^}]+)\}/g, (match, key) => {
            return refMap[key] !== undefined ? refMap[key] : match;
        });

        return resolved;
    };

    // Build prompt string for a specific AI node based on connected Prompt or Conjunction nodes
    NEExec._buildPromptForNode = function(aiNode) {
        let prompt = '';

        // Direct prompt connection from a Prompt node
        const promptNode = NodeEditor.nodes.find(n => n.type === 'prompt');
        const hasPromptConn = promptNode && NodeEditor.connections.some(c =>
            c.from === promptNode.id && c.to === aiNode.id
        );
        if (hasPromptConn) {
            prompt = promptNode.data.text || '';
        }

        // Conjunction node connection (takes precedence if present)
        const conjunctionNode = NodeEditor.nodes.find(n => n.type === 'conjunction');
        const hasConjunctionConn = conjunctionNode && NodeEditor.connections.some(c =>
            c.from === conjunctionNode.id && c.to === aiNode.id
        );
        if (hasConjunctionConn) {
            let template = conjunctionNode.data.template || '';
            const items = conjunctionNode.data.connectedItems || [];
            const refMap = {};
            items.forEach(item => {
                refMap[item.refKey] = item.content;
            });
            prompt = template.replace(/\{([^}]+)\}/g, (match, key) =>
                (refMap[key] !== undefined ? refMap[key] : match)
            );
        }

        return prompt;
    };

    // Toggle visual processing highlight on a node element
    NEExec._setNodeProcessing = function(nodeId, isProcessing) {
        const el = document.getElementById('node-' + nodeId);
        if (!el) return;
        if (isProcessing) {
            el.classList.add('processing');
        } else {
            el.classList.remove('processing');
        }
    };

    window.NEExec = NEExec;
    // Back-compat globals
    window.executeGraph = NEExec.executeGraph;
    window.processGraph = NEExec.processGraph;
})();

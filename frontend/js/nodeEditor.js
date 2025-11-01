function initNodeEditor() {
    // Initialize grid size CSS variable
    const canvas = document.getElementById('nodeCanvas');
    if (canvas && NodeEditor.settings) {
        canvas.style.setProperty('--grid-size', `${NodeEditor.settings.gridSize}px`);
    }

    NEToolbar.setupToolbar();
    NEViewport.initCanvasPanning();
    NEConnections.createConnectionGradient();
    NEMinimap.createMinimap();

    document.getElementById('executeGraphBtn').onclick = executeGraph;
    document.getElementById('clearGraphBtn').onclick = clearGraph;
    document.getElementById('fullscreenBtn').onclick = NEFullscreen.openFullscreen;

    document.getElementById('closeNodeFullscreen').onclick = NEFullscreen.closeFullscreen;
    document.getElementById('nodeFullscreenBackdrop').onclick = (e) => {
        if (e.target.id === 'nodeFullscreenBackdrop') {
            NEFullscreen.closeFullscreen();
        }
    };

    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('nodeFullscreenModal');
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            NEFullscreen.closeFullscreen();
        }
    });

    const { wrapper } = NEUtils.getElements();
    if (wrapper) {
        wrapper.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (typeof AppContextMenu === 'undefined' || typeof GridSettings === 'undefined') return;
            AppContextMenu.open(e.pageX, e.pageY, [
                { label: 'Grid…', onClick: () => GridSettings.openAt(e.pageX + 8, e.pageY + 8) }
            ]);
        });
    }

    try {
        if (NodeEditor.nodes.length === 0) {
            const beforeCount = NodeEditor.nodes.length;
            if (typeof addNode === 'function') {
                addNode('input');
                const inputNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

                addNode('prompt');
                const promptNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

                addNode('aimodel');
                const aiNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

                addNode('output');
                const outputNode = NodeEditor.nodes[NodeEditor.nodes.length - 1];

                const { wrapper } = NEUtils.getElements();
                const rect = wrapper.getBoundingClientRect();
                const center = NEUtils.wrapperToCanvas(rect.width / 2, rect.height / 2);
                const rowY = center.y - 20;

                const layout = [
                    { node: inputNode, x: center.x + 100, y: rowY + 250},
                    { node: promptNode, x: center.x + 100, y: rowY + 550 },
                    { node: aiNode, x: center.x + 500, y: rowY + 400 },
                    { node: outputNode, x: center.x + 900, y: rowY + 407 }
                ];
                layout.forEach(({ node, x, y }) => {
                    node.x = x; node.y = y;
                    const el = document.getElementById('node-' + node.id);
                    if (el) { el.style.left = x + 'px'; el.style.top = y + 'px'; }
                });

                // Connect: Input(images:0) → AI(images:0)
                //          Prompt(text:0) → AI(prompt:1)
                //          AI(captions:0) → Output(data:0)
                if (typeof NEConnections !== 'undefined' && typeof NEConnections.addConnection === 'function') {
                    NEConnections.addConnection(inputNode.id, 0, aiNode.id, 0);
                    NEConnections.addConnection(promptNode.id, 0, aiNode.id, 1);
                    NEConnections.addConnection(aiNode.id, 0, outputNode.id, 0);
                }

                // Ensure minimap and connections reflect initial graph
                // Defer updates to ensure DOM layout is complete
                const updateGraphVisuals = () => {
                    if (typeof NEMinimap !== 'undefined') NEMinimap.updateMinimap();
                    if (typeof NEConnections !== 'undefined' && typeof NEConnections.updateConnections === 'function') {
                        NEConnections.updateConnections();
                    }
                };
                
                // Try immediately (works if node tab is already visible)
                requestAnimationFrame(updateGraphVisuals);
                
                // Retry after delay to handle tab visibility issues
                setTimeout(updateGraphVisuals, 50);
            }
        }
    } catch (e) {
        console.warn('Failed to scaffold default graph:', e);
    }
}

// Fullscreen functions moved to NEFullscreen

// Setup toolbar moved to NEToolbar

// applyTransform, initCanvasPanning, handleZoom moved to NEUtils/NEViewport

// Minimap moved to NEMinimap

// ============================================================================
// Node Management
// ============================================================================

// Node creation moved to NENodes.addNode

// Node rendering and handlers moved to NENodes.renderNode

// Get conjunction references HTML (separate from body)
// getConjunctionReferencesHtml moved to NENodes.getConjunctionReferencesHtml

// Get node content HTML
// getNodeContent moved to NENodes.getNodeContent

// Node updates moved to NENodeUpdate (updateInputNodes, updateConjunctionNode)

// Highlight placeholders in conjunction template
// highlightPlaceholders moved to NENodes.highlightPlaceholders

// Fetch model parameters from API
async function fetchModelParameters(modelName) {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/model/info?model=${modelName}`);
        if (!response.ok) throw new Error('Failed to fetch model parameters');
        const data = await response.json();
        return data.parameters || [];
    } catch (error) {
        console.error('Error fetching model parameters:', error);
        return [];
    }
}

// Build parameter input HTML based on parameter definition
function buildParameterInput(param, currentValue, nodeId) {
    const value = currentValue !== undefined ? currentValue : '';
    const inputId = `node-${nodeId}-param-${param.param_key}`;
    const inputName = `${param.param_key}-${nodeId}`;

    if (param.type === 'number') {
        return `
            <div class="param-group">
                <label class="param-label" for="${inputId}" title="${param.description}">${param.name}</label>
                <input type="number"
                       id="${inputId}"
                       name="${inputName}"
                       class="param-input"
                       data-param-key="${param.param_key}"
                       min="${param.min}"
                       max="${param.max}"
                       step="${param.step}"
                       value="${value}"
                       placeholder="${param.min}-${param.max}">
            </div>
        `;
    } else if (param.type === 'select') {
        const options = param.options.map(opt =>
            `<option value="${opt.value}" ${value === opt.value ? 'selected' : ''}>${opt.label}</option>`
        ).join('');
        return `
            <div class="param-group">
                <label class="param-label" for="${inputId}" title="${param.description}">${param.name}</label>
                <select id="${inputId}"
                        name="${inputName}"
                        class="param-input"
                        data-param-key="${param.param_key}">
                    <option value="">Default</option>
                    ${options}
                </select>
            </div>
        `;
    } else if (param.type === 'checkbox') {
        return `
            <div class="param-group">
                <label class="param-label param-checkbox-label" for="${inputId}" title="${param.description}">
                    <input type="checkbox"
                           id="${inputId}"
                           name="${inputName}"
                           class="param-checkbox"
                           data-param-key="${param.param_key}"
                           ${value ? 'checked' : ''}>
                    ${param.name}
                </label>
            </div>
        `;
    } else if (param.type === 'text') {
        return `
            <div class="param-group">
                <label class="param-label" for="${inputId}" title="${param.description}">${param.name}</label>
                <input type="text"
                       id="${inputId}"
                       name="${inputName}"
                       class="param-input"
                       data-param-key="${param.param_key}"
                       value="${value}"
                       placeholder="${param.description || ''}">
            </div>
        `;
    }
    return '';
}

// Load and display model parameters
async function loadModelParameters(nodeId, modelName) {
    const node = NodeEditor.nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    const paramsContainer = document.getElementById(`params-${nodeId}`);
    if (!paramsContainer) return;
    
    // Fetch parameters
    const parameters = await fetchModelParameters(modelName);
    
    if (parameters.length === 0) {
        paramsContainer.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 4px; font-size: 0.75rem;">No parameters available</div>';
        return;
    }
    
    // Build parameters UI
    const currentParams = node.data.parameters || {};
    const paramsHtml = parameters.map(param => buildParameterInput(param, currentParams[param.param_key], nodeId)).join('');
    paramsContainer.innerHTML = paramsHtml;
    
    // Add event listeners for parameter inputs
    paramsContainer.querySelectorAll('.param-input, .param-checkbox').forEach(input => {
        input.addEventListener('input', (e) => {
            const paramKey = e.target.dataset.paramKey;
            if (!paramKey) return;
            
            if (e.target.type === 'checkbox') {
                node.data.parameters[paramKey] = e.target.checked;
            } else if (e.target.type === 'number') {
                const value = parseFloat(e.target.value);
                if (!isNaN(value)) {
                    node.data.parameters[paramKey] = value;
                } else {
                    delete node.data.parameters[paramKey];
                }
            } else {
                const value = e.target.value;
                if (value) {
                    node.data.parameters[paramKey] = value;
                } else {
                    delete node.data.parameters[paramKey];
                }
            }
        });
    });
}

// Node dragging moved to NEDrag

// Connections moved to NEConnections

// clearGraph moved to NEGraphOps.clearGraph

// Graph execution moved to NEExec.executeGraph/processGraph

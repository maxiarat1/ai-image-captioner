// Building and managing the parameter controls UI inside a node

function buildParameterInput(param, currentValue, nodeId) {
    const value = currentValue !== undefined ? currentValue : '';
    const inputId = `node-${nodeId}-param-${param.param_key}`;
    const inputName = `${param.param_key}-${nodeId}`;

    const groupClass = param.group ? `param-group-${param.group}` : '';
    const dependsOn = param.depends_on ? `data-depends-on="${param.depends_on}"` : '';

    if (param.type === 'number') {
        return `
            <div class="param-group ${groupClass}" ${dependsOn}>
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
            <div class="param-group ${groupClass}" ${dependsOn}>
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
            <div class="param-group ${groupClass}" ${dependsOn}>
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
            <div class="param-group ${groupClass}" ${dependsOn}>
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

async function loadModelParameters(nodeId, modelName) {
    const node = NodeEditor.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const paramsContainer = document.getElementById(`params-${nodeId}`);
    if (!paramsContainer) return;

    const parameters = await fetchModelParameters(modelName);

    if (parameters.length === 0) {
        paramsContainer.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 4px; font-size: 0.75rem;">No parameters available</div>';
        return;
    }

    const currentParams = node.data.parameters || {};
    const paramsHtml = parameters.map(param => buildParameterInput(param, currentParams[param.param_key], nodeId)).join('');
    paramsContainer.innerHTML = paramsHtml;

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

            updateParameterDependencies(nodeId, parameters);
        });
    });

    updateParameterDependencies(nodeId, parameters);
}

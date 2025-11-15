// Logic for parameter dependencies, enabling/disabling groups and showing warnings

function updateParameterDependencies(nodeId, parameters) {
    const node = NodeEditor.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const paramsContainer = document.getElementById(`params-${nodeId}`);
    if (!paramsContainer) return;

    const currentParams = node.data.parameters || {};

    const doSample = currentParams.do_sample || false;
    const numBeams = currentParams.num_beams || 1;

    if (doSample && numBeams > 1) {
        showParameterWarning(paramsContainer, 'Sampling mode (do_sample) conflicts with beam search (num_beams>1). Sampling will be disabled.');
    } else {
        clearParameterWarning(paramsContainer);
    }

    parameters.forEach(param => {
        const paramGroup = paramsContainer.querySelector(`[data-depends-on="${param.param_key}"], #node-${nodeId}-param-${param.param_key}`);
        if (!paramGroup) return;

        // Sampling-specific parameters
        if (param.group === 'sampling' && param.param_key !== 'do_sample') {
            const paramContainer = paramGroup.closest('.param-group');
            if (paramContainer) {
                if (doSample && numBeams === 1) {
                    paramContainer.classList.remove('param-disabled');
                } else {
                    paramContainer.classList.add('param-disabled');
                }
            }
        }

        // Beam-search-specific parameters
        if (param.group === 'beam_search' && param.param_key !== 'num_beams') {
            const paramContainer = paramGroup.closest('.param-group');
            if (paramContainer) {
                if (numBeams > 1 && !doSample) {
                    paramContainer.classList.remove('param-disabled');
                } else {
                    paramContainer.classList.add('param-disabled');
                }
            }
        }
    });
}

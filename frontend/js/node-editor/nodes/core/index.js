// Aggregator for modularized node core pieces.
// Ensures a single shared NENodes object and exposes legacy global aliases.
(function(){
    const NENodes = window.NENodes = window.NENodes || {};

    // List of legacy globals to expose from NENodes
    const aliases = [
        'createPort', 'getOutputPortType', 'getOutputPortName',
        'addOutputPort', 'removeOutputPort', 'updateNodePorts',
        'addNode', 'renderNode', 'getNodeContent',
        'getConjunctionReferencesHtml', 'highlightPlaceholders', 'resolveConjunctionTemplate', 'updateConjunctionPreview',
        'updateOutputStats', 'resetOutputStats',
        'attachCurateHandlers', 'attachCurateModelDropdownHandlers', 'checkAndAddNewPort', 'filterModelsForCurateType'
    ];

    aliases.forEach(name => {
        try {
            if (typeof NENodes[name] !== 'undefined') {
                window[name] = NENodes[name];
            }
        } catch (e) {
            // ignore
        }
    });

    // Ensure window.NENodes remains the canonical object
    window.NENodes = NENodes;
})();

// Entry point for the node-editor bundle.
// This file wires up the core init function and parameter helpers
// as globals, so existing HTML that calls initNodeEditor() continues to work.

// Expose init function
window.initNodeEditor = initNodeEditor;

// Expose parameter API/UI helpers for nodes that need them
window.fetchModelParameters = fetchModelParameters;
window.buildParameterInput = buildParameterInput;
window.loadModelParameters = loadModelParameters;
window.updateParameterDependencies = updateParameterDependencies;
window.showParameterWarning = showParameterWarning;
window.clearParameterWarning = clearParameterWarning;

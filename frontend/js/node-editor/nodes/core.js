// DEPRECATED: core.js has been split into modular files under nodes/core/
// The real implementations now live in the `nodes/core/` directory and
// an aggregator `nodes/core/index.js` exposes legacy globals.
(function(){
    // keep a harmless shim to avoid 404s for any external references until
    // callers are updated. The real NENodes object is created by the
    // modular files and the aggregator.
    window.NENodes = window.NENodes || {};
})();

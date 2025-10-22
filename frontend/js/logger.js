(function () {
  // Enable via localStorage.APP_DEBUG=1 or ?debug=1
  const params = new URLSearchParams(window.location.search);
  const enableDebug = (localStorage.getItem('APP_DEBUG') === '1') || (params.get('debug') === '1');

  const orig = {
    log: console.log.bind(console),
    debug: console.debug.bind(console),
    info: console.info.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console)
  };

  function compactArgs(args) {
    return args
      .map((a) => {
        if (typeof a === 'string') return a;
        try {
          return JSON.stringify(a);
        } catch (_) {
          return String(a);
        }
      })
      .join(' ')
      .replace(/\s+/g, ' ') // collapse whitespace/newlines
      .trim();
  }

  // Gate verbose logs
  console.debug = function (...args) {
    if (!enableDebug) return;
    orig.debug(compactArgs(args));
  };
  console.log = function (...args) {
    if (!enableDebug) return;
    orig.log(compactArgs(args));
  };

  // Provide a simple logger API if needed elsewhere
  window.Logger = {
    debug: (...args) => console.debug(...args),
    log: (...args) => console.log(...args),
    info: (...args) => orig.info(compactArgs(args)),
    warn: (...args) => orig.warn(compactArgs(args)),
    error: (...args) => orig.error(compactArgs(args))
  };

  // Expose flag for quick toggling/inspection
  window.__APP_DEBUG__ = enableDebug;
})();

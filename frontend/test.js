// Simple ASCII flame frames
const frames = [
  " ( ) \n(.) \n(.) ",
  " ( ) \n .( )\n(.) ",
  "  ( )\n(.)\n ( )"
];

// Start an animation that renders to either a DOM element or a writer function.
// Returns a stop() function to end the animation and clear output.
function startFlameAnimation({ element, write, intervalMs = 200 }) {
  let i = 0;

  const render = (text) => {
    if (element) {
      element.textContent = text;
    } else if (write) {
      write(text);
    }
  };

  const timer = setInterval(() => {
    const frame = frames[i % frames.length];
    render(frame);
    i++;
  }, intervalMs);

  return function stop() {
    clearInterval(timer);
    if (element) {
      element.textContent = "";
    } else if (write) {
      write("");
    }
  };
}

// Browser environment: attach to element with id "flame" if present
if (typeof document !== "undefined") {
  const flameElement = document.getElementById("flame");
  if (flameElement) {
    // Expose a global stop function consumers can call when processing finishes
    // Example: window.stopFlameAnimation();
    window.stopFlameAnimation = startFlameAnimation({ element: flameElement });
  }
}

// Node environment: render to console as a quick demo and auto-stop after a few seconds
if (typeof document === "undefined" && typeof process !== "undefined" && process.stdout && process.stdout.write) {
  const write = (text) => {
    // Render on a single line in the console
    const singleLine = text.replace(/\n/g, " ");
    process.stdout.write("\r" + singleLine + "   ");
  };

  const stop = startFlameAnimation({ write, intervalMs: 200 });

  // Auto-stop after 3 seconds for CLI demo
  setTimeout(() => {
    stop();
    // Clear the line
    process.stdout.write("\r" + " ".repeat(40) + "\r");
  }, 3000);
}

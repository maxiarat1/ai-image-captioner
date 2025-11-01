// Canvas constants and coordinate system
// Centralized configuration for the Node Editor canvas
(function() {
    const NECanvas = {
        // Canvas physical dimensions
        SIZE: 5000,
        
        // Computed constants
        get CENTER() { return this.SIZE / 2; }, // 2500
        get MIN_WORLD() { return -this.CENTER; }, // -2500
        get MAX_WORLD() { return this.CENTER; },  // 2500
        
        /**
         * Convert canvas coordinates to world coordinates
         * @param {number} canvasX - Canvas X position (0 to 5000)
         * @param {number} canvasY - Canvas Y position (0 to 5000)
         * @returns {{x: number, y: number}} World coordinates (-2500 to 2500)
         */
        toWorld(canvasX, canvasY) {
            return {
                x: canvasX - this.CENTER,
                y: canvasY - this.CENTER
            };
        },
        
        /**
         * Convert world coordinates to canvas coordinates
         * @param {number} worldX - World X position (-2500 to 2500)
         * @param {number} worldY - World Y position (-2500 to 2500)
         * @returns {{x: number, y: number}} Canvas coordinates (0 to 5000)
         */
        toCanvas(worldX, worldY) {
            return {
                x: worldX + this.CENTER,
                y: worldY + this.CENTER
            };
        },
        
        /**
         * Check if world coordinates are within valid canvas bounds
         * @param {number} worldX - World X position
         * @param {number} worldY - World Y position
         * @returns {boolean} True if within bounds
         */
        isInBounds(worldX, worldY) {
            return worldX >= this.MIN_WORLD && worldX <= this.MAX_WORLD &&
                   worldY >= this.MIN_WORLD && worldY <= this.MAX_WORLD;
        }
    };
    
    // Export to global scope
    window.NECanvas = NECanvas;
})();

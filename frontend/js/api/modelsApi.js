// ============================================================================
// Models API - Fetch model metadata from backend
// ============================================================================

const ModelsAPI = {
    baseUrl: 'http://localhost:5000',
    metadata: null,
    loading: false,
    error: null,

    /**
     * Fetch comprehensive model metadata from backend
     * @returns {Promise<Object>} Model metadata
     */
    async fetchMetadata() {
        if (this.metadata) {
            return this.metadata;
        }

        if (this.loading) {
            // Wait for ongoing fetch
            while (this.loading) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return this.metadata;
        }

        this.loading = true;
        this.error = null;

        try {
            const response = await fetch(`${this.baseUrl}/models/metadata`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            this.metadata = await response.json();
            console.log('Models metadata loaded:', this.metadata);
            return this.metadata;
        } catch (error) {
            console.error('Failed to fetch models metadata:', error);
            this.error = error.message;
            // Return fallback data
            return this.getFallbackMetadata();
        } finally {
            this.loading = false;
        }
    },

    /**
     * Get model count
     * @returns {Promise<number>}
     */
    async getModelCount() {
        const metadata = await this.fetchMetadata();
        return metadata.model_count || 0;
    },

    /**
     * Get all models
     * @returns {Promise<Object>}
     */
    async getModels() {
        const metadata = await this.fetchMetadata();
        return metadata.models || {};
    },

    /**
     * Get specific model by name
     * @param {string} modelName
     * @returns {Promise<Object|null>}
     */
    async getModel(modelName) {
        const metadata = await this.fetchMetadata();
        return metadata.models?.[modelName] || null;
    },

    /**
     * Get VRAM range string
     * @returns {Promise<string>}
     */
    async getVRAMRange() {
        const metadata = await this.fetchMetadata();
        return metadata.vram_range || '2-16';
    },

    /**
     * Get export formats count
     * @returns {Promise<number>}
     */
    async getExportFormatsCount() {
        const metadata = await this.fetchMetadata();
        return metadata.export_formats || 4;
    },

    /**
     * Get tech stack
     * @returns {Promise<Array>}
     */
    async getTechStack() {
        const metadata = await this.fetchMetadata();
        return metadata.tech_stack || [];
    },

    /**
     * Fallback metadata if backend is unavailable
     * Returns minimal default data - backend should be the source of truth
     * @returns {Object}
     */
    getFallbackMetadata() {
        return {
            model_count: 0,
            models: {},
            export_formats: 4,
            vram_range: '2-16',
            tech_stack: [
                {name: 'PyTorch', description: 'Deep learning framework'},
                {name: 'Flask', description: 'REST API backend'},
                {name: 'Vanilla JavaScript', description: 'No-build frontend'},
                {name: 'CUDA', description: 'GPU acceleration'},
                {name: 'DuckDB', description: 'Embedded analytics database'}
            ]
        };
    },

    /**
     * Clear cached metadata (force refresh)
     */
    clearCache() {
        this.metadata = null;
        this.error = null;
    }
};

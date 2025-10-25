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
     * @returns {Object}
     */
    getFallbackMetadata() {
        return {
            model_count: 2,
            models: {
                blip: {
                    display_name: 'BLIP',
                    full_name: 'Salesforce BLIP',
                    description: 'Fast image captioning',
                    speed_score: 80,
                    quality_score: 60,
                    vram_gb: 2,
                    vram_label: '2GB',
                    speed_label: 'Fast',
                    quality_label: 'Good',
                    features: ['Fast processing', 'Low VRAM usage', 'General-purpose captions'],
                    use_cases: ['Batch processing', 'Quick previews', 'Resource-constrained systems']
                },
                r4b: {
                    display_name: 'R-4B',
                    full_name: 'R-4B Advanced Reasoning',
                    description: 'Advanced reasoning model',
                    speed_score: 40,
                    quality_score: 95,
                    vram_gb: 8,
                    vram_label: '8GB (fp16)',
                    speed_label: 'Medium',
                    quality_label: 'Excellent',
                    features: ['Advanced reasoning', 'Configurable precision', 'Detailed captions'],
                    use_cases: ['High-quality descriptions', 'Complex scenes', 'Fine-grained control'],
                    precision_variants: [
                        {name: 'fp16', vram_gb: 8, speed_score: 40, quality_score: 95},
                        {name: '4bit', vram_gb: 2, speed_score: 60, quality_score: 75}
                    ]
                }
            },
            export_formats: 4,
            vram_range: '2-16',
            tech_stack: [
                {name: 'Salesforce BLIP', description: 'Fast image captioning'},
                {name: 'R-4B', description: 'Advanced reasoning model'},
                {name: 'PyTorch', description: 'Deep learning framework'},
                {name: 'Flask', description: 'REST API backend'},
                {name: 'Vanilla JavaScript', description: 'No-build frontend'},
                {name: 'CUDA', description: 'GPU acceleration'}
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

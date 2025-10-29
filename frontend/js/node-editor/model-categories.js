// Model Categories - Dynamic API-based Configuration
// Automatically fetches and organizes models from backend

(function() {
    const ModelCategories = {
        categories: [],
        loaded: false,

        // Fetch categories from backend
        async fetchCategories() {
            try {
                const response = await fetch(`${AppState.apiBaseUrl}/models/categories`);
                if (!response.ok) {
                    throw new Error('Failed to fetch categories');
                }
                const data = await response.json();

                // Transform API response to include model names array for backward compatibility
                this.categories = data.categories.map(cat => ({
                    ...cat,
                    models: cat.models.map(m => m.name)
                }));

                this.loaded = true;
                console.log('Model categories loaded:', this.categories);
                return this.categories;
            } catch (error) {
                console.error('Error fetching model categories:', error);
                // Return empty array on error
                this.categories = [];
                this.loaded = false;
                return [];
            }
        },

        // Get category for a specific model
        getCategoryForModel(modelName) {
            for (const category of this.categories) {
                if (category.models.includes(modelName)) {
                    return category;
                }
            }
            return null;
        },

        // Get all models in a category
        getModelsInCategory(categoryId) {
            const category = this.categories.find(c => c.id === categoryId);
            return category ? category.models : [];
        },

        // Get category by ID
        getCategory(categoryId) {
            return this.categories.find(c => c.id === categoryId);
        }
    };

    // Expose globally
    window.ModelCategories = ModelCategories;
})();

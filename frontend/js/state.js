const AppState = {
    uploadQueue: [],
    apiBaseUrl: 'http://localhost:5000',
    selectedModel: 'wdvit',
    customPrompt: '',
    availableModels: [],
    processedResults: [],
    userConfig: null,
    allResults: [],
    currentPage: 1,
    itemsPerPage: 15,
    uploadCurrentPage: 1,
    uploadSearchQuery: '',
    resultsSearchQuery: '',

    /**
     * Update results from an array of images with captions.
     * @param {Array} imagesWithCaptions - Images that have caption property
     */
    updateResultsFromImages(imagesWithCaptions) {
        this.allResults = imagesWithCaptions.map(img => ({
            queueItem: img,
            data: { caption: img.caption }
        }));

        this.processedResults = imagesWithCaptions.map(img => ({
            filename: img.filename,
            caption: img.caption,
            path: img.path
        }));
    }
};

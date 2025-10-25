async function exportAsTextZip() {
    const zip = new JSZip();
    AppState.processedResults.forEach((result) => {
        const txtFilename = result.filename.replace(/\.[^/.]+$/, '.txt');
        zip.file(txtFilename, result.caption);
    });

    const blob = await zip.generateAsync({
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 9 }
    });

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.zip`);
    showToast(`Exported ${AppState.processedResults.length} text files`);
}

async function exportAsJson() {
    const data = {};
    AppState.processedResults.forEach((result) => {
        data[result.filename] = result.caption;
    });

    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.json`);
    showToast('Exported as JSON');
}

async function exportAsCsv() {
    let csv = 'filename,caption\n';
    AppState.processedResults.forEach((result) => {
        const escapedCaption = `"${result.caption.replace(/"/g, '""')}"`;
        csv += `"${result.filename}",${escapedCaption}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadBlob(blob, `captions_${timestamp}.csv`);
    showToast('Exported as CSV');
}

async function exportWithExif() {
    showToast('Preparing images for EXIF embedding...', true);

    try {
        const hasFiles = AppState.uploadQueue.some(item => item.file);

        if (hasFiles) {
            const formData = new FormData();

            for (const result of AppState.processedResults) {
                const queueItem = AppState.uploadQueue.find(item => item.filename === result.filename);
                if (queueItem && queueItem.file) {
                    formData.append('images', queueItem.file);
                    formData.append('captions', result.caption);
                }
            }

            const response = await fetch(`${AppState.apiBaseUrl}/export/metadata`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to embed metadata');
            }

            const blob = await response.blob();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            downloadBlob(blob, `images_with_metadata_${timestamp}.zip`);
            showToast('Exported images with EXIF metadata');
        } else {
            const image_paths = AppState.processedResults.map(r => r.path);
            const captions = AppState.processedResults.map(r => r.caption);

            const response = await fetch(`${AppState.apiBaseUrl}/export/metadata`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_paths, captions })
            });

            if (!response.ok) {
                throw new Error('Failed to embed metadata');
            }

            const blob = await response.blob();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            downloadBlob(blob, `images_with_metadata_${timestamp}.zip`);
            showToast('Exported images with EXIF metadata');
        }
    } catch (error) {
        console.error('Error exporting with EXIF:', error);
        showToast('Failed to embed EXIF metadata');
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function initDownloadButton() {
    const downloadBtn = document.getElementById('downloadAllBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', openExportModal);
    }
}

function openExportModal() {
    if (AppState.processedResults.length === 0) {
        showToast('No results to export');
        return;
    }
    const modal = document.getElementById('exportModal');
    openModal(modal);
}

function initExportModal() {
    const modal = document.getElementById('exportModal');
    const closeBtn = document.getElementById('closeExportModal');
    const cancelBtn = document.getElementById('cancelExport');
    const confirmBtn = document.getElementById('confirmExport');

    closeBtn?.addEventListener('click', () => closeModal(modal));
    cancelBtn?.addEventListener('click', () => closeModal(modal));

    confirmBtn?.addEventListener('click', async () => {
        const format = document.querySelector('input[name="exportFormat"]:checked')?.value;
        closeModal(modal);

        const downloadBtn = document.getElementById('downloadAllBtn');
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<span class="download-icon">⏳</span> Exporting...';
        downloadBtn.disabled = true;

        try {
            if (format === 'text') await exportAsTextZip();
            else if (format === 'json') await exportAsJson();
            else if (format === 'csv') await exportAsCsv();
            else if (format === 'exif') await exportWithExif();
        } catch (error) {
            console.error('Export error:', error);
            showToast('Export failed');
        } finally {
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
        }
    });

    // Close on backdrop click
    const backdrop = modal.querySelector('.config-modal-backdrop');
    backdrop?.addEventListener('click', (e) => {
        if (e.target === backdrop) closeModal(modal);
    });
}

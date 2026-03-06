/**
 * main.js
 * Asynchronous interaction logic for Transparent Mini-Med.
 */

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const thresholdInput = document.getElementById('threshold-range');
    const thresholdValue = document.getElementById('threshold-value');

    // Results elements
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    const verdictBanner = document.getElementById('verdict-banner');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const overlayDisplay = document.getElementById('overlay-display');
    const overlayDisplay2 = document.getElementById('overlay-display-2');
    const originalDisplay = document.getElementById('original-display');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    const navItems = document.querySelectorAll('.nav-item');
    const viewContainers = document.querySelectorAll('.view-container');
    const historyList = document.getElementById('history-list');
    const refreshHistoryBtn = document.getElementById('refresh-history');
    const viewTitle = document.getElementById('view-title');
    const viewSubtitle = document.getElementById('view-subtitle');

    let selectedFile = null;

    // -- NAVIGATION --

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewId = item.getAttribute('data-view');
            switchView(viewId);
        });
    });

    function switchView(viewId) {
        // Update Nav
        navItems.forEach(nav => nav.classList.toggle('active', nav.getAttribute('data-view') === viewId));

        // Update Containers
        viewContainers.forEach(container => {
            const isMatch = container.id === `${viewId}-view`;
            container.classList.toggle('hidden', !isMatch);
            container.classList.toggle('active', isMatch);
        });

        // Update Titles
        const titles = {
            'dashboard': ['Transparent Medical Diagnosis', 'Explainable AI for Chest X-Ray Analysis'],
            'history': ['Patient History', 'Review past diagnostic sessions and results'],
            'settings': ['System Settings', 'Configure model thresholds and system parameters']
        };
        viewTitle.textContent = titles[viewId][0];
        viewSubtitle.textContent = titles[viewId][1];

        if (viewId === 'history') loadHistory();
        if (viewId === 'settings') loadSettings();
    }

    // -- DASHBOARD INTERACTION --

    // Range update
    thresholdInput.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // File selection
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid X-ray image (JPG/PNG).');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            dropZone.classList.add('hidden');

            // Clear old results
            resultsSection.classList.add('empty');
            resultsContent.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // -- ANALYSIS --

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = '<div class="spinner"></div><p style="margin-left:15px">Analyzing high-level features...</p>';
        resultsSection.appendChild(loadingOverlay);
        resultsSection.classList.remove('empty');

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('threshold', thresholdInput.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
            } else {
                alert(data.error || 'Diagnostic processing failed.');
                resultsSection.classList.add('empty');
            }
        } catch (error) {
            console.error('API Error:', error);
            alert('Connection to clinical server lost.');
            resultsSection.classList.add('empty');
        } finally {
            loadingOverlay.remove();
        }
    });

    function renderResults(data) {
        resultsContent.classList.remove('hidden');
        resultsSection.classList.remove('empty');

        // Demo Warning logic
        const demoWarning = document.getElementById('demo-mode-warning');
        if (data.is_demo) {
            demoWarning.classList.remove('hidden');
        } else {
            demoWarning.classList.add('hidden');
        }

        // Verdict Styling
        const isPn = data.is_pneumonia;
        const color = isPn ? '#f85149' : '#3fb950';

        verdictBanner.className = 'verdict-banner ' + (isPn ? 'pneumonia' : 'normal');
        predictionText.textContent = data.label;
        predictionText.style.color = color;
        confidenceText.textContent = `${data.confidence} (${data.probability})`;

        // Populate Clinical Summary View (New)
        const summaryVerdictBadge = document.getElementById('summary-verdict-badge');
        summaryVerdictBadge.textContent = isPn ? 'Critical' : 'Normal';
        summaryVerdictBadge.style.background = isPn ? 'var(--accent-red)' : 'var(--accent-green)';

        document.getElementById('summary-prediction-text').textContent = data.label;
        document.getElementById('summary-prediction-text').style.color = color;
        document.getElementById('summary-confidence-text').textContent = `${data.confidence} (${data.probability})`;
        document.getElementById('summary-overlay-display').src = data.overlay_image;

        // Visuals
        overlayDisplay.src = data.overlay_image;
        overlayDisplay2.src = data.overlay_image;
        originalDisplay.src = data.original_image;

        // Reset to first tab (Summary)
        switchTab('summary');
    }

    // -- HISTORY --

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                historyList.innerHTML = data.history.map(item => `
                    <tr>
                        <td>${item.timestamp}</td>
                        <td><code class="small-code">${item.id.substring(0, 8)}...</code></td>
                        <td style="color: ${item.is_pneumonia ? 'var(--accent-red)' : 'var(--accent-green)'}">
                            <strong>${item.label}</strong>
                        </td>
                        <td>${item.confidence}</td>
                        <td>
                            <button class="small-btn" onclick="openScan('${item.id}')">View</button>
                        </td>
                    </tr>
                `).join('');
            } else {
                historyList.innerHTML = '<tr><td colspan="5" class="empty-row text-dim">No diagnostic history found.</td></tr>';
            }
        } catch (e) {
            console.error(e);
            historyList.innerHTML = '<tr><td colspan="5" class="empty-row text-dim">Error loading history.</td></tr>';
        }
    }

    refreshHistoryBtn.addEventListener('click', loadHistory);

    // -- SETTINGS --

    async function loadSettings() {
        try {
            const response = await fetch('/api/settings');
            const data = await response.json();

            document.getElementById('weights-info').textContent = data.weights.split('\\').pop().split('/').pop();
            document.getElementById('device-info').textContent = data.device;
            document.getElementById('default-threshold').value = data.threshold;
        } catch (e) {
            console.error(e);
        }
    }

    // Global helper for history table
    window.openScan = async function (scanId) {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            const scan = data.history.find(s => s.id === scanId);

            if (scan) {
                switchView('dashboard');
                renderResults({
                    ...scan,
                    is_demo: scan.original_image.includes('demo') || document.getElementById('weights-info').textContent.includes('demo'),
                    success: true
                });
            }
        } catch (e) {
            console.error(e);
        }
    };

    // -- TAB MANAGEMENT --

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    function switchTab(tabId) {
        tabBtns.forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === tabId));
        tabContents.forEach(c => {
            const isMatch = c.id === `${tabId}-view`;
            c.classList.toggle('hidden', !isMatch);
        });
    }
});

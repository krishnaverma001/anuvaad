document.addEventListener('DOMContentLoaded', async () => {
    const translateButton = document.getElementById('translate-button');
    const clearButton = document.getElementById('clear-button');
    const copyButton = document.getElementById('copy-button');
    const downloadButton = document.getElementById('download-button');
    const shareButton = document.getElementById('share-button');
    const textToTranslate = document.getElementById('text-to-translate');
    const sourceLanguage = document.getElementById('source-language');
    const outputDiv = document.getElementById('output');
    const batchToggle = document.getElementById('batch-toggle');
    const langDetectHint = document.getElementById('lang-detect-hint');
    const themeToggle = document.getElementById('theme-toggle');
    const processDataButton = document.getElementById('process-data-button');
    const processDataStatus = document.getElementById('process-data-status');

    // Debounce timer for detection
    let detectTimer = null;
    const detectDelay = 250;

    // Populate languages dynamically
    try {
        const langRes = await fetch('/languages');
        const langData = await langRes.json();
        const langs = (langData && langData.supported_languages) || ['nepali', 'sinhala'];
        sourceLanguage.innerHTML = '';
        langs.forEach(l => {
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = l.charAt(0).toUpperCase() + l.slice(1);
            sourceLanguage.appendChild(opt);
        });
    } catch (e) {
        // Fallback
        sourceLanguage.innerHTML = '<option value="nepali">Nepali</option><option value="sinhala">Sinhala</option>';
    }

    // Theme toggle
    // Ensure default gradient theme on first load unless user saved preference
    (function() {
      const savedTheme = localStorage.getItem('theme');
      if (!savedTheme) {
        document.documentElement.setAttribute('data-theme', 'gradient');
      }
    })();
    
    themeToggle.addEventListener('click', () => {
        const html = document.documentElement;
        const isDark = html.getAttribute('data-theme') === 'dark';
        html.setAttribute('data-theme', isDark ? 'light' : 'dark');
        themeToggle.textContent = isDark ? 'Light mode' : 'Dark mode';
        localStorage.setItem('anuvaad_theme', isDark ? 'light' : 'dark');
    });
    const savedTheme = localStorage.getItem('anuvaad_theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        themeToggle.textContent = savedTheme === 'dark' ? 'Dark mode' : 'Light mode';
    }

    function setLoading(isLoading) {
        translateButton.disabled = isLoading;
        translateButton.textContent = isLoading ? 'Translating…' : 'Translate';
        outputDiv.setAttribute('aria-busy', String(isLoading));
    }

    // Basic language auto-detect by script characters (debounced)
    textToTranslate.addEventListener('input', () => {
        clearTimeout(detectTimer);
        detectTimer = setTimeout(async () => {
            const sample = (textToTranslate.value || '').slice(0, 200);
            let detected = '';
            // Backend-assisted detection for robustness
            try {
                const res = await fetch('/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
                    body: JSON.stringify({ text: sample })
                });
                if (res.ok) {
                    const data = await res.json();
                    detected = data.detected_language || '';
                }
            } catch (e) {
                // ignore detection errors, fallback to script-based
            }
            if (!detected) {
                const hasDevanagari = /[\u0900-\u097F]/.test(sample);
                const hasSinhala = /[\u0D80-\u0DFF]/.test(sample);
                if (hasDevanagari) detected = 'nepali';
                else if (hasSinhala) detected = 'sinhala';
            }
            if (detected) {
                sourceLanguage.value = detected;
                langDetectHint.textContent = `Detected: ${detected}`;
            } else {
                langDetectHint.textContent = '';
            }
        }, detectDelay);
    });

    translateButton.addEventListener('click', async () => {
        const text = (textToTranslate.value || '').trim();
        const lang = sourceLanguage.value;
        const isBatch = batchToggle && batchToggle.checked;
        const borrowedFixEl = document.getElementById('borrowed-toggle');
        const borrowedFix = borrowedFixEl ? borrowedFixEl.checked : true;

        outputDiv.innerHTML = '';

        if (!text) {
            outputDiv.innerText = 'Please enter some text to translate.';
            return;
        }

        setLoading(true);
        try {
            let response;
            if (isBatch) {
                const texts = text.split('\n').map(t => t.trim()).filter(Boolean);
                if (texts.length === 0) {
                    outputDiv.innerText = 'Please provide at least one non-empty line for batch translation.';
                    setLoading(false);
                    return;
                }
                response = await fetch('/batch-translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'accept': 'application/json'
                    },
                    body: JSON.stringify({ texts, source_language: lang, borrowed_fix: borrowedFix })
                });
            } else {
                response = await fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'accept': 'application/json'
                    },
                    body: JSON.stringify({ text, source_language: lang, borrowed_fix: borrowedFix })
                });
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'An error occurred while translating.');
            }

            const data = await response.json();
            if (isBatch) {
                const results = data.translated_texts || [];
                const table = document.createElement('table');
                table.className = 'result-table';
                const thead = document.createElement('thead');
                thead.innerHTML = '<tr><th>#</th><th>Source</th><th>Translation</th></tr>';
                table.appendChild(thead);
                const tbody = document.createElement('tbody');
                const sources = text.split('\n').map(t => t.trim()).filter(Boolean);
                results.forEach((t, idx) => {
                    const tr = document.createElement('tr');
                    const tdIdx = document.createElement('td'); tdIdx.textContent = String(idx + 1);
                    const tdSrc = document.createElement('td'); tdSrc.textContent = sources[idx] || '';
                    const tdDst = document.createElement('td'); tdDst.textContent = t;
                    tr.appendChild(tdIdx); tr.appendChild(tdSrc); tr.appendChild(tdDst);
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
                outputDiv.appendChild(table);
                downloadButton.dataset.csv = toCSV(sources, results);
            } else {
                outputDiv.innerText = data.translated_text || data.translation || '';
                downloadButton.dataset.csv = toCSV([text], [outputDiv.innerText]);
            }
        } catch (error) {
            outputDiv.innerText = `Error: ${error.message}`;
        } finally {
            setLoading(false);
        }
    });

    // Allow Ctrl+Enter to trigger translation
    textToTranslate.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            translateButton.click();
        }
    });

    // Clear button
    clearButton.addEventListener('click', () => {
        textToTranslate.value = '';
        outputDiv.innerHTML = '';
        downloadButton.removeAttribute('data-csv');
    });

    // Hide dataset processing controls from users (keep in DOM, but not visible)
    if (processDataButton) {
        const datasetControl = processDataButton.closest('.control');
        if (datasetControl) datasetControl.hidden = true;
        const datasetLabel = document.querySelector('label[for="process-data-button"]');
        if (datasetLabel) datasetLabel.hidden = true;
        if (processDataStatus) processDataStatus.hidden = true;
    }

    // Hide borrowed words/names UI text while preserving functionality
    const borrowedFixEl = document.getElementById('borrowed-toggle');
    if (borrowedFixEl) {
        const borrowedControl = borrowedFixEl.closest('.control');
        // Keep the control present only during translation flow but hidden from display
        if (borrowedControl) borrowedControl.hidden = true;
        const borrowedLabel = borrowedFixEl.closest('label');
        if (borrowedLabel) {
            borrowedLabel.hidden = true;
            // Remove any visible text nodes to avoid displaying borrowed words/names text
            borrowedLabel.childNodes.forEach(node => {
                if (node.nodeType === Node.TEXT_NODE) {
                    node.textContent = '';
                }
            });
        }
        const borrowedHint = borrowedControl ? borrowedControl.querySelector('small.hint') : null;
        if (borrowedHint) {
            borrowedHint.hidden = true;
            borrowedHint.textContent = '';
        }
        // Remove the input element itself to ensure it never appears on screen
        borrowedFixEl.remove();
    }

    // Helper to trigger dataset processing without user interaction
    async function triggerProcessData() {
        if (!processDataStatus) return;
        try {
            const res = await fetch('/process-data', { method: 'POST' });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'Failed to process dataset');
            }
            const data = await res.json();
            // Update hidden status for diagnostics; users won't see it
            processDataStatus.textContent = `Processed: ${data.processed_files} files, ${data.total_lines} lines`;
        } catch (e) {
            processDataStatus.textContent = `Error: ${e.message}`;
        }
    }

    // Automatically process dataset on page load (runs once)
    triggerProcessData();

    // Dataset processing trigger (kept inside DOMContentLoaded for scope safety)
    if (processDataButton) {
        processDataButton.addEventListener('click', async () => {
            // Even if clicked (hidden), keep behavior consistent
            processDataStatus.textContent = 'Processing dataset…';
            try {
                const res = await fetch('/process-data', { method: 'POST' });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || 'Failed to process dataset');
                }
                const data = await res.json();
                processDataStatus.textContent = `Processed: ${data.processed_files} files, ${data.total_lines} lines`;
            } catch (e) {
                processDataStatus.textContent = `Error: ${e.message}`;
            }
        });
    }

    // Copy button
    copyButton.addEventListener('click', async () => {
        const text = outputDiv.innerText || '';
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
        } catch (e) {
            // Fallback for older browsers
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
    });

    // Download CSV
    downloadButton.addEventListener('click', () => {
        const csv = downloadButton.dataset.csv;
        if (!csv) return;
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'translations.csv';
        a.click();
        URL.revokeObjectURL(url);
    });

    // Share result
    shareButton.addEventListener('click', async () => {
        const text = outputDiv.innerText || '';
        if (!text) return;
        try {
            await navigator.share({ text });
        } catch (e) {
            // Ignore if not supported
        }
    });

    function toCSV(sources, results) {
        const rows = sources.map((s, i) => [s, results[i] || '']);
        const csvRows = rows.map(r => r.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(','));
        return 'source,translation\n' + csvRows.join('\n');
    }

    // Theme select
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
      const saved = localStorage.getItem('theme');
      const initial = saved || 'gradient';
      document.documentElement.setAttribute('data-theme', initial);
      themeSelect.value = initial;
      themeSelect.addEventListener('change', (e) => {
        const v = e.target.value;
        document.documentElement.setAttribute('data-theme', v);
        localStorage.setItem('theme', v);
      });
    }

    const themeToggleEl = document.getElementById('theme-toggle');
    if (themeToggleEl) {
      themeToggleEl.addEventListener('click', () => {
        const html = document.documentElement;
        const isDark = html.getAttribute('data-theme') === 'dark';
        html.setAttribute('data-theme', isDark ? 'light' : 'dark');
        localStorage.setItem('theme', isDark ? 'light' : 'dark');
      });
    }
});

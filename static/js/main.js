// Debug interface
const DebugAPI = {
    state: {
        currentMode: 'summary',
        isLoading: false,
        lastResponse: null,
        templates: []
    },
    
    init() {
        console.log('Debug API initialized');
        window.DebugAPI = this;
    },
    
    getCurrentMode() {
        return this.state.currentMode;
    },
    
    getIsLoading() {
        return this.state.isLoading;
    },
    
    getTemplates() {
        return this.state.templates;
    },
    
    updateState(key, value) {
        this.state[key] = value;
        console.log(`State updated - ${key}:`, value);
    }
};

// Initialize debug API
DebugAPI.init();

// Global state
let currentMode = 'summary';
let isLoading = false;
window.allPrompts = [];

// API Configuration
const API_ENDPOINTS = {
    summary: '/api/generate-summary',
    data: '/api/analyze-data', 
    report: '/api/generate-report',
    datasets: '/api/datasets',
    prompts: '/api/prompts'
};

// DOM Elements
const modeButtons = document.querySelectorAll('.mode-button');
const summaryFilters = document.getElementById('summaryFilters');
const dataFilters = document.getElementById('dataFilters');
const reportFilters = document.getElementById('reportFilters');
const messagesContainer = document.getElementById('messages');
const inputForm = document.getElementById('inputForm');
const speciesInput = document.getElementById('speciesInput');
const submitButton = document.getElementById('submitButton');
const summarySelect = document.getElementById('promptTemplate');
const analysisSelect = document.getElementById('analysisPromptTemplate');

// Template Management
async function fetchLangfuseTemplates() {
    console.log('Fetching templates...');
    try {
        const response = await fetch(API_ENDPOINTS.prompts);
        if (!response.ok) {
            throw new Error('Failed to fetch templates');
        }

        const data = await response.json();
        console.log('Received templates:', data);

        if (!data.templates || !Array.isArray(data.templates)) {
            throw new Error('Invalid template data received');
        }

        window.allPrompts = data.templates.map(template => ({
            id: template.name,
            content: template.content,
            config: template.config,
            labels: template.labels,
            version: template.version
        }));

        // Populate categories in both dropdowns
        if (data.categories) {
            populateCategories('promptFolder', data.categories);
            populateCategories('templateCategory', data.categories);
        }

        // Initialize template dropdowns
        filterTemplates('promptTemplate');
        filterTemplates('analysisPromptTemplate');

        return true;
    } catch (error) {
        console.error('Template fetch error:', error);
        showError(`Failed to load templates: ${error.message}`);
        return false;
    }
}

function populateCategories(selectId, categories) {
    const select = document.getElementById(selectId);
    if (!select) return;

    select.innerHTML = '<option value="">All Categories</option>';
    categories.forEach(category => {
        select.appendChild(new Option(category, category));
    });
}

function filterTemplates(templateSelectId, category = null) {
    const templateSelect = document.getElementById(templateSelectId);
    if (!templateSelect) return;

    templateSelect.innerHTML = '<option value="">Select Template...</option>';

    const filteredPrompts = window.allPrompts.filter(prompt => {
        if (category) {
            return prompt.labels.includes(category);
        }
        return true;
    });

    console.log(`Filtered prompts for ${templateSelectId}:`, filteredPrompts);

    filteredPrompts.forEach(prompt => {
        const option = new Option(prompt.id, prompt.id);
        option.dataset.content = prompt.content;
        option.dataset.config = JSON.stringify(prompt.config);
        option.dataset.labels = prompt.labels.join(',');
        templateSelect.appendChild(option);
    });
}

function handleTemplateSelection(templateSelectId) {
    const templateSelect = document.getElementById(templateSelectId);
    const descriptionId = templateSelectId === 'analysisPromptTemplate' ? 
        'analysisPromptDescription' : 
        (templateSelectId === 'promptTemplate' ? 'templateDescription' : null);

    if (!descriptionId) return;

    const descriptionDiv = document.getElementById(descriptionId);
    const selectedOption = templateSelect.options[templateSelect.selectedIndex];

    if (selectedOption && selectedOption.value) {
        const content = selectedOption.dataset.content;
        const config = JSON.parse(selectedOption.dataset.config || '{}');
        const labels = selectedOption.dataset.labels?.split(',') || [];

        let descriptionHtml = '<div class="template-info">';
        if (content) {
            descriptionHtml += `<p><strong>Template:</strong> ${content}</p>`;
        }
        if (config) {
            descriptionHtml += `
                <p><strong>Model:</strong> ${config.model || 'Not specified'}</p>
                <p><strong>Temperature:</strong> ${config.temperature || 'Not specified'}</p>
            `;
        }
        if (labels.length > 0) {
            descriptionHtml += `<p><strong>Labels:</strong> ${labels.join(', ')}</p>`;
        }
        descriptionHtml += '</div>';

        descriptionDiv.innerHTML = descriptionHtml;
        descriptionDiv.classList.add('active');
    } else {
        descriptionDiv.innerHTML = '';
        descriptionDiv.classList.remove('active');
    }
}

// Dataset Management
async function loadDatasets() {
    console.log('Loading datasets');
    const datasetSelect = document.getElementById('dataset');
    try {
        const response = await fetch(API_ENDPOINTS.datasets);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const datasets = await response.json();
        if (!Array.isArray(datasets)) throw new Error('Invalid datasets data');

        datasetSelect.innerHTML = '<option value="">Select Dataset...</option>';
        datasets.forEach(dataset => {
            datasetSelect.appendChild(new Option(dataset, dataset));
        });
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showError('Failed to load datasets');
    }
}

// UI Utilities
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    messageDiv.textContent = typeof content === 'object' ? 
        JSON.stringify(content, null, 2) : content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showError(message) {
    console.error('Error:', message);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error';
    errorDiv.textContent = `Error: ${message}`;
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateFiltersVisibility(mode) {
    summaryFilters.style.display = mode === 'summary' ? 'block' : 'none';
    dataFilters.style.display = mode === 'data' ? 'block' : 'none';
    reportFilters.style.display = mode === 'report' ? 'block' : 'none';
    speciesInput.placeholder = mode === 'data' ? 
        "Enter analysis query..." : "Enter species name...";
}

// Form Handling
async function handleFormSubmit(e) {
    e.preventDefault();
    
    if (isLoading) {
        console.log('Already loading, submission blocked');
        return;
    }

    const inputText = speciesInput.value.trim();
    if (!inputText) {
        showError('Please enter ' + (currentMode === 'data' ? 'an analysis query' : 'a species name'));
        return;
    }

    let payload = {};
    let endpoint = API_ENDPOINTS[currentMode];

    try {
        switch(currentMode) {
            case 'data':
                const dataset = document.getElementById('dataset').value;
                if (!dataset) {
                    throw new Error('Please select a dataset');
                }
                
                payload = {
                    query: inputText,
                    dataset: dataset,
                    tool_name: "Analysis Agent",
                    prompt_name: analysisSelect.value,
                    analysis_type: "general"
                };
                break;
                
            case 'summary':
                payload = {
                    query: inputText,
                    tool_name: "General Agent",
                    prompt_name: summarySelect.value
                };
                break;
                
            case 'report':
                payload = {
                    species: inputText,
                    template: document.getElementById('reportTemplate').value
                };
                break;
        }

        isLoading = true;
        DebugAPI.updateState('isLoading', true);
        submitButton.disabled = true;
        
        addMessage(inputText, true);
        addMessage('Loading...', false);

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload),
        });

        const responseText = await response.text();
        let result = JSON.parse(responseText);

        if (response.ok) {
            const loadingMessage = messagesContainer.querySelector('.message:last-child');
            if (loadingMessage?.textContent === 'Loading...') {
                messagesContainer.removeChild(loadingMessage);
            }

            let formatted = 'No content';
            if (currentMode === 'summary') {
                formatted = result.summary || (result.content || [])
                    .map(item => `Title: ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`)
                    .join('\n\n');
            } else if (currentMode === 'data') {
                formatted = result.analysis || 'No analysis results available';
            } else {
                formatted = result;
            }

            addMessage(formatted, false);
            DebugAPI.updateState('lastResponse', result);
        } else {
            throw new Error(result.detail || result.message || 'Server returned an error');
        }
    } catch (error) {
        console.error('Error during submission:', error);
        const loadingMessage = messagesContainer.querySelector('.message:last-child');
        if (loadingMessage?.textContent === 'Loading...') {
            messagesContainer.removeChild(loadingMessage);
        }
        showError(error.message || 'Failed to submit the query');
    } finally {
        isLoading = false;
        DebugAPI.updateState('isLoading', false);
        submitButton.disabled = false;
        speciesInput.value = '';
    }
}

// Category Filtering
document.getElementById('promptFolder')?.addEventListener('change', (e) => {
    filterTemplates('promptTemplate', e.target.value);
});

document.getElementById('templateCategory')?.addEventListener('change', (e) => {
    filterTemplates('analysisPromptTemplate', e.target.value);
});

// Template Selection
summarySelect?.addEventListener('change', () => {
    handleTemplateSelection('promptTemplate');
});

analysisSelect?.addEventListener('change', () => {
    handleTemplateSelection('analysisPromptTemplate');
});

// Mode Selection
modeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const mode = button.getAttribute('data-mode');
        currentMode = mode;
        DebugAPI.updateState('currentMode', mode);
        
        modeButtons.forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        updateFiltersVisibility(mode);
    });
});

// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        document.documentElement.classList.toggle('dark');
        themeToggle.textContent = document.documentElement.classList.contains('dark') ? '🌙' : '☀️';
    });
}

// Form Submit
inputForm?.addEventListener('submit', handleFormSubmit);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Application initializing...');
    Promise.all([
        fetchLangfuseTemplates(),
        loadDatasets()
    ]).then(() => {
        updateFiltersVisibility(currentMode);
        console.log('Application initialized successfully');
    }).catch(error => {
        console.error('Initialization error:', error);
        showError('Failed to initialize application');
    });
});
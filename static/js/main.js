// Debug interface
const DebugAPI = {
    state: {
        currentMode: 'summary',
        isLoading: false,
        lastResponse: null
    },
    
    init() {
        console.log('Debug API initialized');
        window.DebugAPI = this;
    },
    
    getCurrentMode() {
        console.log('Current mode:', this.state.currentMode);
        return this.state.currentMode;
    },
    
    getIsLoading() {
        console.log('Loading state:', this.state.isLoading);
        return this.state.isLoading;
    },
    
    getLastResponse() {
        console.log('Last response:', this.state.lastResponse);
        return this.state.lastResponse;
    },
    
    updateState(key, value) {
        this.state[key] = value;
        console.log(`State updated - ${key}:`, value);
    }
};

// Initialize debug API
DebugAPI.init();

// State management
let currentMode = 'summary';
let isLoading = false;

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

async function loadPromptTemplates() {
    console.log('Loading prompt templates');
    try {
        const response = await fetch(API_ENDPOINTS.prompts);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        console.log('Received templates:', data);
        
        populateSelect('promptFolder', data.folders);
        populateSelect('analysisFolder', data.folders);
        
        window.allPrompts = data.prompts;
        filterPromptTemplates();
        filterAnalysisTemplates();
    } catch (error) {
        console.error('Failed to load templates:', error);
        showError('Failed to load templates');
    }
}

function populateSelect(selectId, folders) {
    const select = document.getElementById(selectId);
    if (!select) return;
    
    select.innerHTML = '<option value="">All Folders</option>';
    folders.forEach(folder => {
        select.appendChild(new Option(folder, folder));
    });
}

function filterPromptTemplates() {
    filterTemplates('promptFolder', 'promptTemplate');
}

function filterAnalysisTemplates() {
    filterTemplates('analysisFolder', 'analysisPromptTemplate');
}

function filterTemplates(folderSelectId, templateSelectId) {
    const selectedFolder = document.getElementById(folderSelectId).value;
    const templateSelect = document.getElementById(templateSelectId);
    
    templateSelect.innerHTML = '<option value="">Select Template...</option>';
    
    if (selectedFolder && window.allPrompts) {
        const filteredPrompts = window.allPrompts.filter(p => 
            p.metadata.folder === selectedFolder
        );
        filteredPrompts.forEach(prompt => {
            const fileName = prompt.id.split('/').pop().replace('.txt', '');
            templateSelect.appendChild(new Option(fileName, prompt.id));
        });
    }
}

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

function addMessage(content, isUser = false) {
    console.log('Adding message:', { content, isUser });
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
    errorDiv.className = 'error message';
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

function formatDataAnalysisResponse(data) {
    if (!data) return 'No response data received';
    
    let formattedResponse = 'Analysis Results:\n\n';
    
    if (data.analysis) formattedResponse += data.analysis + '\n\n';
    
    if (data.metrics) {
        formattedResponse += 'Metrics:\n';
        Object.entries(data.metrics).forEach(([key, value]) => {
            formattedResponse += `${key}: ${typeof value === 'object' ? 
                JSON.stringify(value) : value}\n`;
        });
    }
    
    if (data.metadata) {
        formattedResponse += '\nMetadata:\n';
        if (data.metadata.dataset) formattedResponse += `Dataset: ${data.metadata.dataset}\n`;
        if (data.metadata.prompt_name) formattedResponse += `Prompt: ${data.metadata.prompt_name}\n`;
        if (data.metadata.timestamp) formattedResponse += `Timestamp: ${data.metadata.timestamp}`;
    }
    
    return formattedResponse;
}

async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('Form submission started');
    
    if (isLoading) {
        console.log('Already loading, submission blocked');
        return;
    }
 
    const inputText = speciesInput.value.trim();
    console.log('Input text:', inputText);
    
    if (!inputText) {
        const errorMessage = 'Please enter ' + (currentMode === 'data' ? 'an analysis query' : 'a species name');
        console.log('Empty input error:', errorMessage);
        showError(errorMessage);
        return;
    }
 
    let payload = {};
    let endpoint = API_ENDPOINTS[currentMode];
    console.log('Selected endpoint:', endpoint);
 
    try {
        switch(currentMode) {
            case 'data':
                const dataset = document.getElementById('dataset').value;
                if (!dataset) {
                    throw new Error('Please select a dataset');
                }
                
                if (inputText.length < 2) {
                    throw new Error('Query must be at least 2 characters');
                }
 
                if (inputText.length > 200) {
                    throw new Error('Query must not exceed 200 characters');
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
        
        console.log('Request payload:', payload);
        isLoading = true;
        submitButton.disabled = true;
        addMessage(inputText, true);
        addMessage('Loading...', false);
 
        console.log('Sending request to:', endpoint);
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload),
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers));
 
        // Log the raw response
        const responseText = await response.text();
        console.log('Raw response:', responseText);
        
        // Try parsing the response
        let result;
        try {
            result = JSON.parse(responseText);
            console.log('Parsed response:', result);
        } catch (error) {
            console.error('Failed to parse response:', error);
            throw new Error(`Invalid response format: ${error.message}`);
        }
 
        if (response.ok) {
            // Remove loading message before adding the response
            const loadingMessage = messagesContainer.querySelector('.message:last-child');
            if (loadingMessage && loadingMessage.textContent === 'Loading...') {
                messagesContainer.removeChild(loadingMessage);
            }
 
            let formatted = 'No content';
            if (currentMode === 'summary') {
                if (result.summary) {
                    formatted = result.summary;
                } else if (result.content && result.content.length > 0) {
                    formatted = result.content.map(item => 
                        `Title: ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`
                    ).join('\n\n');
                }
            } else if (currentMode === 'data') {
                if (result.analysis) {
                    formatted = result.analysis;
                    console.log('Analysis received:', formatted);
                } else {
                    console.log('No analysis in response');
                    formatted = 'No analysis results available';
                }
            } else {
                formatted = result;
            }
 
            console.log('Adding formatted response to UI:', formatted);
            addMessage(formatted, false);
        } else {
            throw new Error(result.detail || result.message || 'Server returned an error');
        }
    } catch (error) {
        console.error('Error during submission:', error);
        // Remove loading message before showing error
        const loadingMessage = messagesContainer.querySelector('.message:last-child');
        if (loadingMessage && loadingMessage.textContent === 'Loading...') {
            messagesContainer.removeChild(loadingMessage);
        }
        showError(error.message || 'Failed to submit the query');
    } finally {
        isLoading = false;
        submitButton.disabled = false;
        console.log('Form submission completed');
    }
 
    speciesInput.value = '';
 }
// Theme toggle
const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        document.documentElement.classList.toggle('dark');
        themeToggle.textContent = document.documentElement.classList.contains('dark') ? '🌙' : '☀️';
    });
}

// Event Listeners
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

document.getElementById('promptFolder')?.addEventListener('change', filterPromptTemplates);
document.getElementById('analysisFolder')?.addEventListener('change', filterAnalysisTemplates);
inputForm.addEventListener('submit', handleFormSubmit);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Application initialized');
    loadPromptTemplates();
    loadDatasets();
    updateFiltersVisibility(currentMode);
});
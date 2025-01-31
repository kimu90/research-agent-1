// State management
let currentMode = 'summary';
let isLoading = false;

// API endpoints
const API_ENDPOINTS = {
    summary: '/api/generate-summary',
    data: '/api/analyze-data',
    report: '/api/generate-report',
    datasets: '/api/datasets',
    prompts: '/api/prompts'
};

// DOM Elements - Get all elements only once
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

// Load prompt templates - fixed to properly handle the response
async function loadPromptTemplates() {
    try {
        const response = await fetch(API_ENDPOINTS.prompts);
        const prompts = await response.json();

        // Clear existing options before appending new ones
        summarySelect.innerHTML = "";
        analysisSelect.innerHTML = "";

        prompts.forEach(prompt => {
            // Create the option for summary select
            const summaryOption = document.createElement("option");
            summaryOption.value = prompt.id;  // This should match the prompt's ID
            summaryOption.textContent = prompt.content;

            // Append to the summary select menu
            summarySelect.appendChild(summaryOption);

            // Create the option for analysis select
            const analysisOption = document.createElement("option");
            analysisOption.value = prompt.id;  // Same ID for analysis, or modify for specific use
            analysisOption.textContent = prompt.content;

            // Append to the analysis select menu
            analysisSelect.appendChild(analysisOption);
        });
    } catch (error) {
        showError('Failed to load prompt templates');
    }
}

// Load datasets
async function loadDatasets() {
    const datasetSelect = document.getElementById('dataset');
    try {
        const response = await fetch(API_ENDPOINTS.datasets);
        if (!response.ok) throw new Error('Failed to fetch datasets');
        
        const datasets = await response.json();
        if (!Array.isArray(datasets)) throw new Error('Invalid datasets data');

        datasetSelect.innerHTML = '<option value="">Select Dataset...</option>';
        datasets.forEach(dataset => {
            const option = new Option(dataset, dataset);
            datasetSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Dataset loading error:', error);
        showError('Failed to load datasets');
    }
}

// Message handling
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    messageDiv.textContent = typeof content === 'object' ? 
        JSON.stringify(content, null, 2) : content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error message';
    errorDiv.textContent = `Error: ${message}`;
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update filters visibility
function updateFiltersVisibility(mode) {
    summaryFilters.style.display = mode === 'summary' ? 'block' : 'none';
    dataFilters.style.display = mode === 'data' ? 'block' : 'none';
    reportFilters.style.display = mode === 'report' ? 'block' : 'none';
    speciesInput.placeholder = mode === 'data' ? 
        "Enter analysis query..." : "Enter species name...";
}

// Format response
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

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    if (isLoading) return;

    const inputText = speciesInput.value.trim();
    if (!inputText) {
        showError('Please enter ' + (currentMode === 'data' ? 'an analysis query' : 'a species name'));
        return;
    }

    let payload = {};
    let endpoint = API_ENDPOINTS[currentMode];

    switch(currentMode) {
        case 'data':
            const dataset = document.getElementById('dataset').value;
            if (!dataset) {
                showError('Please select a dataset');
                return;
            }
            
            if (inputText.length < 2) {
                showError('Query must be at least 2 characters');
                return;
            }

            if (inputText.length > 200) {
                showError('Query must not exceed 200 characters');
                return;
            }

            payload = {
                query: inputText,
                dataset: dataset,
                tool_name: "Analysis Agent",
                prompt_name: analysisSelect.value
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
    addMessage(inputText, true);
    addMessage('Loading...', false);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        const result = await response.json();

        if (response.ok) {
            const formatted = currentMode === 'data' ? formatDataAnalysisResponse(result) : result.text || 'No content';
            addMessage(formatted, false);
        } else {
            showError(result.message || 'An error occurred');
        }
    } catch (error) {
        showError(error.message || 'Failed to submit the query');
    } finally {
        isLoading = false;
    }

    speciesInput.value = ''; // Clear input field
}

// Event Listeners
modeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const mode = button.getAttribute('data-mode');
        currentMode = mode;
        
        modeButtons.forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        updateFiltersVisibility(mode);
    });
});

inputForm.addEventListener('submit', handleFormSubmit);

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    loadPromptTemplates();
    loadDatasets();
    updateFiltersVisibility(currentMode);
});

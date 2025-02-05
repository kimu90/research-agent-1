// State management
let currentMode = 'summary';
let isLoading = false;

// API endpoints
const API_ENDPOINTS = {
    summary: '/api/generate-summary',
    data: '/api/analyze-data',
    report: '/api/generate-report',
    datasets: '/api/datasets',
    prompts: '/api/prompts',
    promptFolders: '/api/prompt-folders'
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

// Summary mode elements
const summaryFolderSelect = document.getElementById('promptFolder');
const summarySelect = document.getElementById('promptTemplate');

// Data Analysis mode elements
const dataAnalysisFolderSelect = document.getElementById('promptAnalysisFolder');
const analysisSelect = document.getElementById('analysisPromptTemplate');
const datasetSelect = document.getElementById('dataset');

// Load prompt folders
async function loadPromptFolders(mode) {
    const folderSelect = mode === 'summary' ? summaryFolderSelect : dataAnalysisFolderSelect;
    try {
        const response = await fetch(API_ENDPOINTS.promptFolders);
        const folders = await response.json();

        // Clear existing options before appending new ones
        folderSelect.innerHTML = '<option value="">Select Prompt Folder...</option>';

        folders.forEach(folder => {
            const option = document.createElement('option');
            option.value = folder;
            option.textContent = folder;
            folderSelect.appendChild(option);
        });
    } catch (error) {
        showError(`Failed to load prompt folders for ${mode}`);
        console.error(error);
    }
}

// Load prompt templates for a specific folder
async function loadPromptTemplates(mode, folder) {
    const promptSelect = mode === 'summary' ? summarySelect : analysisSelect;
    
    try {
        const response = await fetch(`${API_ENDPOINTS.prompts}?folder=${encodeURIComponent(folder)}`);
        const prompts = await response.json();

        // Clear existing options before appending new ones
        promptSelect.innerHTML = '<option value="">Select Prompt Template...</option>';

        prompts.forEach(prompt => {
            const option = document.createElement('option');
            option.value = prompt.id;
            option.textContent = prompt.content;
            promptSelect.appendChild(option);
        });

        // Enable the select after populating
        promptSelect.disabled = false;
    } catch (error) {
        showError(`Failed to load prompt templates for ${mode}`);
        console.error(error);
    }
}

// Load datasets
async function loadDatasets() {
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
            const dataset = datasetSelect.value;
            const analysisFolder = dataAnalysisFolderSelect.value;
            const analysisPrompt = analysisSelect.value;

            if (!dataset) {
                showError('Please select a dataset');
                return;
            }

            if (!analysisFolder) {
                showError('Please select an analysis prompt folder');
                return;
            }

            if (!analysisPrompt) {
                showError('Please select an analysis prompt template');
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
                prompt_name: analysisPrompt,
                prompt_folder: analysisFolder
            };
            break;
            
        case 'summary':
            const summaryFolder = summaryFolderSelect.value;
            const summaryPrompt = summarySelect.value;

            if (!summaryFolder) {
                showError('Please select a summary prompt folder');
                return;
            }

            if (!summaryPrompt) {
                showError('Please select a summary prompt template');
                return;
            }

            payload = {
                query: inputText,
                tool_name: "General Agent",
                prompt_name: summaryPrompt,
                prompt_folder: summaryFolder
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
        console.log('Full server response:', result); // Detailed logging

        if (response.ok) {
            // More comprehensive handling for summary mode
            let formatted = 'No content';
            if (currentMode === 'summary') {
                // Try multiple ways to extract content
                if (result.summary) {
                    formatted = result.summary;
                } else if (result.content && result.content.length > 0) {
                    // If summary is missing, compile content
                    formatted = result.content.map(item => 
                        `Title: ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`
                    ).join('\n\n');
                }
            } else {
                formatted = currentMode === 'data' ? formatDataAnalysisResponse(result) : 'No content';
            }

            addMessage(formatted, false);
        } else {
            showError(result.message || 'An error occurred');
        }
    } catch (error) {
        console.error('Submission error:', error);
        showError(error.message || 'Failed to submit the query');
    } finally {
        isLoading = false;
        speciesInput.value = ''; // Clear input field
    }
}

// Event Listeners for folder selection
document.addEventListener('DOMContentLoaded', () => {
    // Load datasets
    loadDatasets();

    // Load prompt folders for both summary and data analysis modes
    loadPromptFolders('summary');
    loadPromptFolders('data');

    // Update filters visibility for initial mode
    updateFiltersVisibility(currentMode);

    // Add event listeners for folder selection
    summaryFolderSelect.addEventListener('change', (e) => {
        const selectedFolder = e.target.value;
        if (selectedFolder) {
            loadPromptTemplates('summary', selectedFolder);
        } else {
            summarySelect.innerHTML = '<option value="">Select Prompt Template...</option>';
            summarySelect.disabled = true;
        }
    });

    dataAnalysisFolderSelect.addEventListener('change', (e) => {
        const selectedFolder = e.target.value;
        if (selectedFolder) {
            loadPromptTemplates('data', selectedFolder);
        } else {
            analysisSelect.innerHTML = '<option value="">Select Analysis Prompt...</option>';
            analysisSelect.disabled = true;
        }
    });
});

// Mode button event listener
modeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const mode = button.getAttribute('data-mode');
        currentMode = mode;
        
        modeButtons.forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        updateFiltersVisibility(mode);
    });
});

// Form submission event listener
inputForm.addEventListener('submit', handleFormSubmit);
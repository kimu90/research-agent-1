import { Langfuse } from "langfuse";

// Initialize Langfuse
const langfuse = new Langfuse({
    secretKey: "sk-lf-7dea79c4-9fc8-45b7-b28c-aa021adac010",
    publicKey: "pk-lf-f51fc564-c24e-4f1e-a771-525c8dd43eb5",
    baseUrl: "https://fantastic-waddle-694pv64vjwqv244jp-3000.app.github.dev"
});

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
    const trace = langfuse.trace({
        name: 'load_prompt_templates',
        metadata: { action: 'template_loading' }
    });
    
    try {
        const response = await fetch(API_ENDPOINTS.prompts);
        const data = await response.json();
        
        trace.update({ metadata: { success: true, templateCount: data.prompts.length } });
        
        populateSelect('promptFolder', data.folders);
        populateSelect('analysisFolder', data.folders);
        
        window.allPrompts = data.prompts;
        filterPromptTemplates();
        filterAnalysisTemplates();
    } catch (error) {
        trace.update({ 
            status: 'failure',
            metadata: { error: error.message }
        });
        showError('Failed to load templates');
    } finally {
        trace.end();
    }
}

function populateSelect(selectId, folders) {
    const select = document.getElementById(selectId);
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
    
    if (selectedFolder) {
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
    const trace = langfuse.trace({
        name: 'load_datasets',
        metadata: { action: 'dataset_loading' }
    });
    
    const datasetSelect = document.getElementById('dataset');
    try {
        const response = await fetch(API_ENDPOINTS.datasets);
        if (!response.ok) throw new Error('Failed to fetch datasets');
        
        const datasets = await response.json();
        if (!Array.isArray(datasets)) throw new Error('Invalid datasets data');

        trace.update({ metadata: { success: true, datasetCount: datasets.length } });

        datasetSelect.innerHTML = '<option value="">Select Dataset...</option>';
        datasets.forEach(dataset => {
            datasetSelect.appendChild(new Option(dataset, dataset));
        });
    } catch (error) {
        trace.update({ 
            status: 'failure',
            metadata: { error: error.message }
        });
        showError('Failed to load datasets');
    } finally {
        trace.end();
    }
}

function addMessage(content, isUser = false) {
    console.log("Adding message:", { content, isUser });
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
    if (isLoading) return;

    const inputText = speciesInput.value.trim();
    console.log("Submitting:", inputText);

    const trace = langfuse.trace({
        name: `${currentMode}_request`,
        metadata: { 
            mode: currentMode,
            input_length: inputText.length
        }
    });

    if (!inputText) {
        trace.update({
            status: 'failure',
            metadata: { error: 'Empty input' }
        });
        trace.end();
        showError('Please enter ' + (currentMode === 'data' ? 'an analysis query' : 'a species name'));
        return;
    }

    let payload = {};
    let endpoint = API_ENDPOINTS[currentMode];

    switch(currentMode) {
        case 'data':
            const dataset = document.getElementById('dataset').value;
            if (!dataset) {
                trace.update({
                    status: 'failure',
                    metadata: { error: 'Dataset not selected' }
                });
                trace.end();
                showError('Please select a dataset');
                return;
            }
            
            if (inputText.length < 2) {
                trace.update({
                    status: 'failure',
                    metadata: { error: 'Query too short' }
                });
                trace.end();
                showError('Query must be at least 2 characters');
                return;
            }

            if (inputText.length > 200) {
                trace.update({
                    status: 'failure',
                    metadata: { error: 'Query too long' }
                });
                trace.end();
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
            const selectedTemplate = summarySelect.value;
            if (!selectedTemplate) {
                trace.update({
                    status: 'failure',
                    metadata: { error: 'Template not selected' }
                });
                trace.end();
                showError('Please select a template');
                return;
            }
            payload = {
                query: inputText,
                tool_name: "General Agent",
                prompt_name: selectedTemplate
            };
            break;
            
        case 'report':
            const reportTemplate = document.getElementById('reportTemplate').value;
            if (!reportTemplate) {
                trace.update({
                    status: 'failure',
                    metadata: { error: 'Report template not selected' }
                });
                trace.end();
                showError('Please select a report template');
                return;
            }
            payload = {
                species: inputText,
                template: reportTemplate
            };
            break;
    }

    trace.update({ metadata: { payload } });
    console.log("Sending payload:", payload);
    console.log("To endpoint:", endpoint);

    isLoading = true;
    addMessage(inputText, true);
    const loadingMsg = 'Loading...';
    addMessage(loadingMsg, false);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        const result = await response.json();
        console.log("Received response:", result);

        if (response.ok) {
            let formatted = 'No content';
            if (currentMode === 'summary') {
                if (result.summary) {
                    formatted = result.summary;
                } else if (result.content && result.content.length > 0) {
                    formatted = result.content.map(item => 
                        `Title: ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`
                    ).join('\n\n');
                }
            } else {
                formatted = currentMode === 'data' ? formatDataAnalysisResponse(result) : result;
            }
            console.log("Formatted response:", formatted);
            
            // Remove loading message
            if (messagesContainer.lastChild) {
                messagesContainer.removeChild(messagesContainer.lastChild);
            }
            
            // Add formatted response
            addMessage(formatted, false);
            
            trace.update({
                status: 'success',
                metadata: { 
                    response_length: formatted.length,
                    response_type: currentMode
                }
            });
        } else {
            if (messagesContainer.lastChild) {
                messagesContainer.removeChild(messagesContainer.lastChild);
            }
            showError(result.message || 'An error occurred');
            
            trace.update({
                status: 'failure',
                metadata: { 
                    error: result.message || 'API Error',
                    status_code: response.status
                }
            });
        }
    } catch (error) {
        console.error("Error:", error);
        if (messagesContainer.lastChild) {
            messagesContainer.removeChild(messagesContainer.lastChild);
        }
        showError(error.message || 'Failed to submit the query');
        
        trace.update({
            status: 'failure',
            metadata: { 
                error: error.message,
                error_type: error.name
            }
        });
    } finally {
        isLoading = false;
        speciesInput.value = '';
        trace.end();
    }
}

// Event Listeners
modeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const mode = button.getAttribute('data-mode');
        currentMode = mode;
        
        langfuse.trace({
            name: 'mode_change',
            metadata: { new_mode: mode }
        }).end();
        
        modeButtons.forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        updateFiltersVisibility(mode);
    });
});

document.getElementById('promptFolder').addEventListener('change', filterPromptTemplates);
document.getElementById('analysisFolder').addEventListener('change', filterAnalysisTemplates);
inputForm.addEventListener('submit', handleFormSubmit);

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    loadPromptTemplates();
    loadDatasets();
    updateFiltersVisibility(currentMode);
});
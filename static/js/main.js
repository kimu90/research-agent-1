// State management
let currentMode = 'summary';
let isLoading = false;

// DOM Elements
const modeButtons = document.querySelectorAll('.mode-button');
const summaryFilters = document.getElementById('summaryFilters');
const dataFilters = document.getElementById('dataFilters');
const reportFilters = document.getElementById('reportFilters');
const messagesContainer = document.getElementById('messages');
const inputForm = document.getElementById('inputForm');
const speciesInput = document.getElementById('speciesInput');
const submitButton = document.getElementById('submitButton');

// Descriptions for different options
const DESCRIPTIONS = {
    summaryTypes: {
        general: "A broad overview of the species including basic characteristics, habitat, and behavior.",
        detailed: "In-depth analysis including taxonomy, lifecycle, ecology, and interactions with other species.",
        conservation: "Focus on conservation status, threats, and protection measures."
    },
    promptTemplates: {
        basic: "Simple, straightforward prompts for general information.",
        detailed: "Comprehensive prompts that cover multiple aspects of the species.",
        scientific: "Technical prompts focused on academic and research perspectives.",
        custom: "Create your own custom prompt template."
    },
    analysisTypes: {
        basic: "Simple statistical analysis of population and distribution patterns.",
        trends: "Time-series analysis of population changes and movement patterns.",
        geographic: "Spatial analysis of species distribution and habitat preferences."
    }
};

// API endpoints
const API_ENDPOINTS = {
    summary: '/api/generate-summary',
    data: '/api/analyze-data',
    report: '/api/generate-report',
    datasets: '/api/datasets'
};

// Update filter visibility based on mode
function updateFiltersVisibility(mode) {
    summaryFilters.style.display = mode === 'summary' ? 'block' : 'none';
    dataFilters.style.display = mode === 'data' ? 'block' : 'none';
    reportFilters.style.display = mode === 'report' ? 'block' : 'none';
    
    // Update placeholder text based on mode
    speciesInput.placeholder = mode === 'data' ? 
        "Enter analysis query..." : 
        "Enter species name...";
}

// Mode switching
modeButtons.forEach(button => {
    button.addEventListener('click', () => {
        modeButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        currentMode = button.getAttribute('data-mode');
        updateFiltersVisibility(currentMode);
    });
});

// Update descriptions based on selections
function updateDescriptions() {
    // Summary Type description
    const summaryDesc = document.getElementById('summaryDescription');
    const summaryType = document.getElementById('summaryType');
    if (summaryDesc && summaryType) {
        summaryDesc.textContent = DESCRIPTIONS.summaryTypes[summaryType.value] || '';
    }

    // Prompt Template description
    const promptDesc = document.getElementById('promptDescription');
    const promptTemplate = document.getElementById('promptTemplate');
    if (promptDesc && promptTemplate) {
        promptDesc.textContent = DESCRIPTIONS.promptTemplates[promptTemplate.value] || '';
    }

    // Analysis Type description
    const analysisDesc = document.getElementById('analysisDescription');
    const analysisType = document.getElementById('analysisType');
    if (analysisDesc && analysisType) {
        analysisDesc.textContent = DESCRIPTIONS.analysisTypes[analysisType.value] || '';
    }
}

// Event listeners for description updates
document.getElementById('summaryType')?.addEventListener('change', updateDescriptions);
document.getElementById('promptTemplate')?.addEventListener('change', (e) => {
    updateDescriptions();
    const customPromptGroup = document.getElementById('customPromptGroup');
    if (customPromptGroup) {
        customPromptGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
    }
});
document.getElementById('analysisType')?.addEventListener('change', updateDescriptions);

// Message handling functions
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    if (typeof content === 'object') {
        messageDiv.innerHTML = `<pre>${JSON.stringify(content, null, 2)}</pre>`;
    } else {
        messageDiv.textContent = content;
    }
    
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

// Format responses
function formatDataAnalysisResponse(data) {
    if (!data) return 'No response data received';
    
    let formattedResponse = 'Analysis Results:\n\n';
    
    // Add analysis if available
    if (data.analysis) {
        formattedResponse += data.analysis + '\n\n';
    }
    
    // Add metrics if available
    if (data.metrics) {
        formattedResponse += 'Metrics:\n';
        try {
            Object.entries(data.metrics).forEach(([key, value]) => {
                formattedResponse += `${key}: ${
                    typeof value === 'object' ? JSON.stringify(value) : value
                }\n`;
            });
        } catch (e) {
            formattedResponse += 'Error formatting metrics\n';
        }
    }
    
    // Add metadata if available
    if (data.metadata) {
        formattedResponse += '\nMetadata:\n';
        if (data.metadata.dataset) formattedResponse += `Dataset: ${data.metadata.dataset}\n`;
        if (data.metadata.analysis_type) formattedResponse += `Analysis Type: ${data.metadata.analysis_type}\n`;
        if (data.metadata.timestamp) formattedResponse += `Timestamp: ${data.metadata.timestamp}`;
    }
    
    return formattedResponse;
}

// Form submission handler
inputForm.addEventListener('submit', async (e) => {
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
            const analysisType = document.getElementById('analysisType').value;
            
            // Add validation
            if (!dataset) {
                showError('Please select a dataset');
                return;
            }
            
            if (!inputText || inputText.length < 2) {
                showError('Query must be at least 2 characters');
                return;
            }

            if (inputText.length > 200) {
                showError('Query must not exceed 200 characters');
                return;
            }

            // Match EXACTLY what FastAPI expects
            payload = {
                query: inputText,          // Required: 2-200 chars
                dataset: dataset,          // Required: must exist
                tool_name: "Analysis Agent",
                analysis_type: analysisType || "basic"  
            };
            break;
            
        case 'summary':
            payload = {
                query: inputText,
                tool_name: "General Agent",
                prompt_template: document.getElementById('promptTemplate').value,
               
            };
            break;
            
        case 'report':
            payload = {
                species: inputText,
                template: document.getElementById('reportTemplate').value,
                included_sections: ["summary", "data", "conclusions"]
            };
            break;
    }

    console.log('Sending payload:', payload);
    addMessage(inputText, true);

    isLoading = true;
    submitButton.disabled = true;
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading message';
    loadingDiv.textContent = 'Processing...';
    messagesContainer.appendChild(loadingDiv);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        // Get response text first for more detailed error handling
        const responseText = await response.text();

        // Check if response is empty
        if (!responseText.trim()) {
            throw new Error('Received an empty response from the server');
        }

        // Check if response is ok
        if (!response.ok) {
            console.error('Server response:', response.status, responseText);
            
            // Try to parse error details
            let errorDetails;
            try {
                errorDetails = JSON.parse(responseText);
            } catch {
                // If not JSON, use the raw text
                throw new Error(responseText || 'Request failed');
            }

            // Extract and throw more specific error
            if (errorDetails && errorDetails.detail) {
                throw new Error(errorDetails.detail);
            }
            throw new Error(responseText || 'Request failed');
        }

        // Try to parse the JSON
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (jsonError) {
            console.error('JSON Parse Error:', jsonError);
            console.error('Raw response text:', responseText);
            throw new Error(`Invalid JSON response: ${jsonError.message}`);
        }

        // Validate data structure
        if (!data) {
            throw new Error('Parsed response is empty or null');
        }

        // Format response based on mode
        let formattedResponse;
        switch(currentMode) {
            case 'data':
                if (data.analysis) {
                    formattedResponse = formatDataAnalysisResponse(data);
                } else {
                    console.error('Received data:', data);
                    throw new Error('No analysis data in response');
                }
                break;
            default:
                formattedResponse = JSON.stringify(data, null, 2);
        }

        addMessage(formattedResponse);
        
    } catch (error) {
        console.error('Complete error details:', error);
        
        // More informative error messages
        let errorMessage = error.message;
        
        // Specific error handling for known issues
        if (error.message.includes('Unexpected end of JSON input')) {
            errorMessage = 'Server did not return a complete response. Please try again.';
        } else if (error.message.includes('Invalid JSON response')) {
            errorMessage = 'Received an invalid response from the server. Check server logs.';
        } else if (error.message.includes("'QueryTrace' object has no attribute 'timestamp'")) {
            errorMessage = 'Server encountered an internal tracing error. Please contact support.';
        }
        
        showError(errorMessage);
    } finally {
        isLoading = false;
        submitButton.disabled = false;
        const loadingElement = messagesContainer.querySelector('.loading');
        if (loadingElement) loadingElement.remove();
        speciesInput.value = '';
    }
});

// Load available datasets
async function loadDatasets() {
    const datasetSelect = document.getElementById('dataset');
    
    try {
        console.log('Fetching datasets...');
        const response = await fetch(API_ENDPOINTS.datasets);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch datasets: ${response.status}`);
        }
        
        const datasets = await response.json();
        console.log('Available datasets:', datasets);
        
        // Clear existing options except the first one
        while (datasetSelect.options.length > 1) {
            datasetSelect.remove(1);
        }
        
        // Add new options
        datasets.forEach(dataset => {
            const option = new Option(dataset, dataset);
            datasetSelect.add(option);
        });
        
        // Fetch dataset info
        datasets.forEach(async (dataset) => {
            try {
                const infoResponse = await fetch(`/api/dataset/${dataset}/info`);
                const datasetInfo = await infoResponse.json();
                console.log(`Dataset ${dataset} info:`, datasetInfo);
            } catch (error) {
                console.error(`Error fetching info for ${dataset}:`, error);
            }
        });
        
    } catch (error) {
        console.error('Error loading datasets:', error);
        showError('Failed to load available datasets');
    }
}

// Add dark mode functionality
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

// Check for saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    html.className = savedTheme;
    themeToggle.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
}

themeToggle.addEventListener('click', () => {
    if (html.className === 'dark') {
        html.className = 'light';
        themeToggle.textContent = '🌙';
        localStorage.setItem('theme', 'light');
    } else {
        html.className = 'dark';
        themeToggle.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    updateDescriptions();
    updateFiltersVisibility(currentMode);
});
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
    datasets: {
        population: "Historical population counts and demographic data.",
        distribution: "Geographic distribution and habitat range information.",
        genetic: "Genetic diversity and evolutionary data."
    },
    analysisTypes: {
        basic: "Simple statistical analysis of the selected dataset.",
        trends: "Time-series analysis and trend identification.",
        geographic: "Spatial analysis and mapping of distribution patterns."
    }
};

// Update filter visibility based on mode
// Update filter visibility based on mode
function updateFiltersVisibility(mode) {
    summaryFilters.style.display = mode === 'summary' ? 'block' : 'none';
    dataFilters.style.display = mode === 'data' ? 'block' : 'none';
    reportFilters.style.display = mode === 'report' ? 'block' : 'none';
}

// API endpoints
const API_ENDPOINTS = {
    summary: '/api/generate-summary',
    data: '/api/analyze-data',
    report: '/api/generate-report'
};

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

    // Dataset description
    const datasetDesc = document.getElementById('datasetDescription');
    const dataset = document.getElementById('dataset');
    if (datasetDesc && dataset) {
        datasetDesc.textContent = DESCRIPTIONS.datasets[dataset.value] || '';
    }

    // Analysis Type description
    const analysisDesc = document.getElementById('analysisDescription');
    const analysisType = document.getElementById('analysisType');
    if (analysisDesc && analysisType) {
        analysisDesc.textContent = DESCRIPTIONS.analysisTypes[analysisType.value] || '';
    }
}

// Event listeners for description updates and custom prompt toggle
document.getElementById('summaryType')?.addEventListener('change', updateDescriptions);
document.getElementById('promptTemplate')?.addEventListener('change', (e) => {
    updateDescriptions();
    const customPromptGroup = document.getElementById('customPromptGroup');
    if (customPromptGroup) {
        customPromptGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
    }
});
document.getElementById('dataset')?.addEventListener('change', updateDescriptions);
document.getElementById('analysisType')?.addEventListener('change', updateDescriptions);

// Chat message functions
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    messageDiv.textContent = content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatResponse(data, mode) {
    switch(mode) {
        case 'summary':
            return `Summary: ${data.summary}\n\nSources:\n${
                data.content.map(item => 
                    `- ${item.title}\n  ${item.snippet}`
                ).join('\n\n')
            }`;
        
        case 'data':
            return `Analysis Summary: ${data.analysis_summary}\n\nData Points:\n${
                data.data_points.map(point => 
                    `- Value: ${point.value}, Date: ${point.timestamp}, Location: ${point.location}`
                ).join('\n')
            }`;
        
        case 'report':
            return `${data.report_content}\n\nGenerated: ${data.metadata.generated_at}`;
        
        default:
            return JSON.stringify(data, null, 2);
    }
}

// Form submission handler
inputForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (isLoading) return;
    
    const species = speciesInput.value.trim();
    if (!species) {
        showError('Please enter a species name');
        return;
    }

    let payload = {};
    let endpoint = API_ENDPOINTS[currentMode];

    switch(currentMode) {
        case 'summary':
            payload = {
                query: species,
                tool_name: "General Agent",
                summary_type: document.getElementById('summaryType').value,
                prompt_template: document.getElementById('promptTemplate').value,
                custom_prompt: document.getElementById('customPrompt')?.value || '',
                data_sources: Array.from(document.querySelectorAll('.checkbox-group input:checked'))
                    .map(cb => cb.value)
            };
            break;
        
        case 'data':
            const dataset = document.getElementById('dataset').value;
            if (!dataset) {
                showError('Please select a dataset');
                return;
            }
            payload = {
                species: species,
                dataset: dataset,
                analysis_type: document.getElementById('analysisType').value
            };
            break;
        
        case 'report':
            payload = {
                species: species,
                template: document.getElementById('reportTemplate').value,
                included_sections: ["summary", "data", "conclusions"]
            };
            break;
    }

    addMessage(species, true);

    isLoading = true;
    submitButton.disabled = true;
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    messagesContainer.appendChild(loadingDiv);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'An error occurred');
        }

        const formattedResponse = formatResponse(data, currentMode);
        addMessage(formattedResponse);
    } catch (error) {
        showError(error.message);
    } finally {
        isLoading = false;
        submitButton.disabled = false;
        const loadingElement = messagesContainer.querySelector('.loading');
        if (loadingElement) loadingElement.remove();
        speciesInput.value = '';
    }
});

async function loadDatasets() {
    const datasetSelect = document.getElementById('dataset');
    
    try {
        console.log('Fetching datasets...');
        const response = await fetch('/api/datasets');
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to fetch datasets:', response.status, errorText);
            return;
        }
        
        const datasets = await response.json();
        console.log('Received datasets:', datasets);
        
        // Clear existing options except the first one
        while (datasetSelect.options.length > 1) {
            datasetSelect.remove(1);
        }
        
        // Add new options for each CSV in the data folder
        datasets.forEach(dataset => {
            console.log('Adding dataset:', dataset);
            const option = new Option(dataset, dataset);
            datasetSelect.add(option);
        });
    } catch (error) {
        console.error('Detailed dataset load error:', error);
    }
}

// Ensure it's called after DOM is loaded
document.addEventListener('DOMContentLoaded', loadDatasets);

// Initialize descriptions on page load
updateDescriptions();
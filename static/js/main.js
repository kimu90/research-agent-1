// static/js/main.js
// State management
let currentMode = 'summary';
let isLoading = false;

// DOM Elements
const modeButtons = document.querySelectorAll('.mode-button');
const modeSettings = document.querySelectorAll('.mode-settings');
const messagesContainer = document.getElementById('messages');
const inputForm = document.getElementById('inputForm');
const speciesInput = document.getElementById('speciesInput');
const submitButton = document.getElementById('submitButton');

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
        modeSettings.forEach(settings => {
            settings.style.display = 'none';
        });
        document.getElementById(`${currentMode}Settings`).style.display = 'block';
    });
});

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
                tool_name: "General Agent"
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
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
   try {
       const response = await fetch(API_ENDPOINTS.prompts);
       const data = await response.json();
       
       populateSelect('promptFolder', data.folders);
       populateSelect('analysisFolder', data.folders);
       
       window.allPrompts = data.prompts;
       filterPromptTemplates();
       filterAnalysisTemplates();
   } catch (error) {
       showError('Failed to load templates');
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
   const datasetSelect = document.getElementById('dataset');
   try {
       const response = await fetch(API_ENDPOINTS.datasets);
       if (!response.ok) throw new Error('Failed to fetch datasets');
       
       const datasets = await response.json();
       if (!Array.isArray(datasets)) throw new Error('Invalid datasets data');

       datasetSelect.innerHTML = '<option value="">Select Dataset...</option>';
       datasets.forEach(dataset => {
           datasetSelect.appendChild(new Option(dataset, dataset));
       });
   } catch (error) {
       showError('Failed to load datasets');
   }
}

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

           addMessage(formatted, false);
       } else {
           showError(result.message || 'An error occurred');
       }
   } catch (error) {
       showError(error.message || 'Failed to submit the query');
   } finally {
       isLoading = false;
       messagesContainer.removeChild(messagesContainer.lastChild);
   }

   speciesInput.value = '';
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

document.getElementById('promptFolder').addEventListener('change', filterPromptTemplates);
document.getElementById('analysisFolder').addEventListener('change', filterAnalysisTemplates);
inputForm.addEventListener('submit', handleFormSubmit);

// Initialization
document.addEventListener('DOMContentLoaded', () => {
   loadPromptTemplates();
   loadDatasets();
   updateFiltersVisibility(currentMode);
});
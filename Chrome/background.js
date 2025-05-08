// Initialize storage for page data
let pageData = new Map();

// List of domains to skip
const SKIP_DOMAINS = [
  'gmail.com',
  'whatsapp.com',
  'facebook.com',
  'linkedin.com',
  'twitter.com',
  'instagram.com',
  'banking',
  'paypal.com',
  'amazon.com',
  'netflix.com'
];

// Backend server URL
const BACKEND_URL = 'http://localhost:5000';

// Check if server is available
async function checkServerHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Server health check failed:', error);
    return false;
  }
}

// Initialize when extension is installed
chrome.runtime.onInstalled.addListener(async () => {
  try {
    // Load existing page data from storage
    const data = await chrome.storage.local.get(['pageData']);
    if (data.pageData) {
      pageData = new Map(Object.entries(data.pageData));
    }
    
    // Check server health
    const isServerHealthy = await checkServerHealth();
    if (!isServerHealthy) {
      console.error('Backend server is not available. Please start the server.');
    }
  } catch (error) {
    console.error('Error initializing storage:', error);
  }
});

// Listen for tab updates to index new pages
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    const url = new URL(tab.url);
    
    // Skip if domain is in the skip list or if it's a chrome:// URL
    if (SKIP_DOMAINS.some(domain => url.hostname.includes(domain)) || url.protocol === 'chrome:') {
      return;
    }

    try {
      // Check server health first
      const isServerHealthy = await checkServerHealth();
      if (!isServerHealthy) {
        console.error('Cannot index page: Backend server is not available');
        return;
      }

      // Get page content
      const [{result}] = await chrome.scripting.executeScript({
        target: { tabId },
        func: () => {
          return {
            title: document.title,
            content: document.body.innerText,
            url: window.location.href
          };
        }
      });

      // Send to backend for indexing
      const response = await fetch(`${BACKEND_URL}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: result.title,
          content: result.content,
          url: result.url
        })
      });

      if (response.ok) {
        const { id, status } = await response.json();
        if (status === 'success') {
          pageData.set(id, {
            title: result.title,
            url: result.url,
            content: result.content
          });

          // Save to storage
          await chrome.storage.local.set({
            pageData: Object.fromEntries(pageData)
          });
        }
      } else {
        const error = await response.json();
        console.error('Error indexing page:', error);
      }
    } catch (error) {
      console.error('Error indexing page:', error);
    }
  }
});

// Handle search requests
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'search') {
    handleSearch(request.query)
      .then(sendResponse)
      .catch(error => {
        console.error('Search error:', error);
        sendResponse({ error: 'Search failed. Please make sure the backend server is running.' });
      });
    return true; // Keep the message channel open for async response
  }
});

async function handleSearch(query) {
  try {
    // Check server health first
    const isServerHealthy = await checkServerHealth();
    if (!isServerHealthy) {
      throw new Error('Backend server is not available');
    }

    // Send search request to backend
    const response = await fetch(`${BACKEND_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Search request failed');
    }

    const { results, message } = await response.json();
    return { results, message };
  } catch (error) {
    console.error('Search error:', error);
    return { 
      error: error.message || 'Search failed. Please make sure the backend server is running.',
      results: [] 
    };
  }
} 
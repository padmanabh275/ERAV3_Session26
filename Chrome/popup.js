document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('searchInput');
  const searchButton = document.getElementById('searchButton');
  const resultsDiv = document.getElementById('results');
  const statusDiv = document.getElementById('status');

  searchButton.addEventListener('click', performSearch);
  searchInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      performSearch();
    }
  });

  async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    statusDiv.textContent = 'Searching...';
    resultsDiv.innerHTML = '';

    try {
      // Send message to background script to perform search
      const response = await chrome.runtime.sendMessage({
        action: 'search',
        query: query
      });

      if (response.results && response.results.length > 0) {
        displayResults(response.results);
      } else {
        resultsDiv.innerHTML = '<p>No results found</p>';
      }
    } catch (error) {
      statusDiv.textContent = 'Error performing search';
      console.error('Search error:', error);
    }
  }

  function displayResults(results) {
    resultsDiv.innerHTML = '';
    results.forEach(result => {
      const resultItem = document.createElement('div');
      resultItem.className = 'result-item';
      resultItem.innerHTML = `
        <div><strong>${result.title}</strong></div>
        <div>${result.url}</div>
        <div>${result.snippet}</div>
      `;
      resultItem.addEventListener('click', () => {
        chrome.tabs.create({ url: result.url });
      });
      resultsDiv.appendChild(resultItem);
    });
    statusDiv.textContent = `Found ${results.length} results`;
  }
}); 
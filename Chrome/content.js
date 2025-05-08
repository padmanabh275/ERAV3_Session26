// Listen for messages from the background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'highlight') {
    highlightText(request.text);
    sendResponse({ success: true });
  }
});

function highlightText(text) {
  // Remove any existing highlights
  const existingHighlights = document.querySelectorAll('.web-page-indexer-highlight');
  existingHighlights.forEach(el => {
    const parent = el.parentNode;
    parent.replaceChild(document.createTextNode(el.textContent), el);
    parent.normalize();
  });

  // Create a tree walker to find text nodes
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  const textToHighlight = text.toLowerCase();
  let node;
  const nodesToHighlight = [];

  // Find all text nodes containing the search text
  while (node = walker.nextNode()) {
    const nodeText = node.textContent.toLowerCase();
    if (nodeText.includes(textToHighlight)) {
      nodesToHighlight.push(node);
    }
  }

  // Highlight each matching node
  nodesToHighlight.forEach(node => {
    const span = document.createElement('span');
    span.className = 'web-page-indexer-highlight';
    span.style.backgroundColor = '#ffeb3b';
    span.style.padding = '2px';
    span.style.borderRadius = '2px';
    
    const text = node.textContent;
    const index = text.toLowerCase().indexOf(textToHighlight);
    
    const before = text.substring(0, index);
    const match = text.substring(index, index + textToHighlight.length);
    const after = text.substring(index + textToHighlight.length);
    
    const fragment = document.createDocumentFragment();
    fragment.appendChild(document.createTextNode(before));
    span.textContent = match;
    fragment.appendChild(span);
    fragment.appendChild(document.createTextNode(after));
    
    node.parentNode.replaceChild(fragment, node);
  });

  // Scroll to the first highlight
  const firstHighlight = document.querySelector('.web-page-indexer-highlight');
  if (firstHighlight) {
    firstHighlight.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  }
} 
# Web Page Indexer Chrome Extension

A Chrome extension that automatically indexes web pages you visit and allows you to search through them using semantic search powered by FAISS and Sentence Transformers.

## Features

- **Automatic Page Indexing**: Automatically indexes web pages as you browse
- **Semantic Search**: Search through indexed pages using natural language queries
- **Privacy-Focused**: All processing happens locally on your machine
- **Smart Snippets**: Shows relevant snippets from search results
- **Domain Filtering**: Automatically skips indexing of sensitive domains (email, banking, etc.)

## Prerequisites

- Python 3.7 or higher
- Chrome browser
- Required Python packages (install using `pip install -r requirements.txt`):
  - flask
  - flask-cors
  - faiss-cpu
  - sentence-transformers
  - numpy

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chrome-extension
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**:
   ```bash
   python server.py
   ```
   The server will start on `http://localhost:5000`

4. **Load the Chrome extension**:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select the `Chrome` directory from this project

## Usage

1. **Automatic Indexing**:
   - The extension automatically indexes web pages as you browse
   - Pages are processed in the background
   - Sensitive domains (email, banking, etc.) are automatically skipped

2. **Searching**:
   - Click the extension icon in Chrome
   - Enter your search query
   - Results will show relevant snippets from indexed pages
   - Click on results to visit the original pages

3. **API Endpoints**:
   - Health Check: `GET http://localhost:5000/health`
   - List Documents: `GET http://localhost:5000/documents`
   - Search: `POST http://localhost:5000/search`
     ```json
     {
       "query": "your search query"
     }
     ```

## Architecture

- **Backend Server** (`server.py`):
  - Flask server handling indexing and search
  - FAISS for efficient similarity search
  - Sentence Transformers for semantic embeddings
  - Local storage of index and metadata

- **Chrome Extension**:
  - Background script for page processing
  - Content script for page content extraction
  - Popup interface for search

## Security Features

- Automatic skipping of sensitive domains:
  - Email services (Gmail, etc.)
  - Banking websites
  - Social media platforms
  - E-commerce sites
  - Streaming services

## Troubleshooting

1. **Server Connection Issues**:
   - Ensure the Python server is running
   - Check if port 5000 is available
   - Verify firewall settings

2. **Indexing Problems**:
   - Check browser console for errors
   - Verify server logs
   - Ensure page content is accessible

3. **Search Issues**:
   - Verify documents are indexed
   - Check server health endpoint
   - Review search query format

## Development

- **Adding New Features**:
  - Backend: Modify `server.py`
  - Extension: Update files in `Chrome/` directory
  - Test changes using Chrome's developer tools

- **Customizing**:
  - Modify `SKIP_DOMAINS` in `background.js` to change domain filtering
  - Adjust search parameters in `server.py`
  - Customize UI in extension popup

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 
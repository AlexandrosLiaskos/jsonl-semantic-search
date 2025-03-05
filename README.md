# JSONL Semantic Search

A powerful semantic search tool for JSONL databases that combines vector embeddings and traditional keyword-based search techniques to provide highly relevant search results.

## Overview

JSONL Semantic Search is designed to index and search large collections of documents stored in JSONL format. It leverages modern NLP techniques including vector embeddings and TF-IDF to provide both semantic understanding and keyword matching capabilities.

## Features

- **Hybrid Search**: Combines semantic (vector-based) and lexical (keyword-based) search for optimal results
- **Vector Embeddings**: Uses Hugging Face's transformer models to generate high-quality embeddings
- **TF-IDF**: Implements advanced information retrieval techniques for keyword matching
- **Title Boosting**: Optionally gives higher weight to matches in document titles
- **Query Expansion**: Automatically expands search queries with semantically related terms using WordNet and Word2Vec
- **Relevance Scoring**: Sophisticated scoring algorithm that balances semantic similarity and keyword relevance
- **Configurable Thresholds**: Adjust relevance thresholds to control precision vs. recall
- **CLI & Programmatic API**: Use as a command-line tool or integrate into your Node.js applications

## Architecture

The tool consists of three main components:

1. **Analyzer**: Examines JSONL databases to provide statistics and insights
2. **Indexer**: Builds search indices including vector embeddings and TF-IDF matrices
3. **Searcher**: Processes search queries and retrieves relevant results

## Technologies Used

- **Node.js**: Runtime environment
- **Hugging Face Inference API**: For generating vector embeddings
- **Natural.js**: For NLP tasks including TF-IDF calculation
- **FAISS (optional)**: For efficient similarity search in high-dimensional spaces
- **Morpha**: For lemmatization of text
- **Stopword**: For removing common stopwords
- **WordNet**: For synonym expansion in queries
- **Word2Vec**: For finding semantically similar words
- **Commander.js**: For CLI interface
- **Chalk & Ora**: For terminal UI
- **P-Limit**: For controlling API concurrency

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jsonl-semantic-search.git
cd jsonl-semantic-search

# Install dependencies
npm install

# Make the CLI executable
chmod +x src/cli.js
```

## Usage

### Command Line Interface

The tool provides three main commands:

#### Analyze a JSONL database

```bash
node src/cli.js analyze path/to/database.jsonl [options]
```

Options:
- `-f, --fields [fields]`: Specific fields to analyze (comma-separated)
- `-s, --sample <n>`: Number of entries to sample

#### Build a search index

```bash
node src/cli.js index path/to/database.jsonl [options]
```

Options:
- `-o, --output <dir>`: Output directory for index (default: "./index")
- `-c, --content-field <field>`: Field containing main content (default: "content")
- `-t, --title-field <field>`: Field containing title (default: "title")
- `-m, --model <name>`: Embedding model to use (default: "universal-sentence-encoder")
- `--no-title-boost`: Disable title relevance boosting
- `--hf-api-key <key>`: Hugging Face API key for embedding generation

#### Search the index

```bash
node src/cli.js search "your search query" [options]
```

Options:
- `-i, --index <dir>`: Index directory (default: "./index")
- `-n, --limit <n>`: Maximum number of results (default: 10)
- `-t, --threshold <n>`: Relevance threshold (0-1) (default: 0.5)
- `--semantic-weight <n>`: Weight for semantic similarity (0-1) (default: 0.7)
- `--title-weight <n>`: Weight for title relevance (0-1) (default: 0.3)
- `--hf-api-key <key>`: Hugging Face API key for embedding generation

### Programmatic API

You can also use the tool programmatically in your Node.js applications:

```javascript
import { analyzeDatabase } from 'jsonl-semantic-search/src/analyzer.js';
import { buildIndex } from 'jsonl-semantic-search/src/indexer.js';
import { searchIndex } from 'jsonl-semantic-search/src/searcher.js';

// Analyze a database
const stats = await analyzeDatabase('path/to/database.jsonl');

// Build an index
await buildIndex('path/to/database.jsonl', {
  outputDir: './index',
  contentField: 'content',
  titleField: 'title'
});

// Search the index
const results = await searchIndex('your search query', {
  indexDir: './index',
  threshold: 0.5,
  semanticWeight: 0.7
});
```

## Implementation Details

### Text Preprocessing

Before indexing or searching, text is preprocessed through several steps:
1. Conversion to lowercase
2. Removal of special characters
3. Tokenization
4. Stopword removal
5. Lemmatization using Morpha

### Embedding Generation

The tool uses Hugging Face's transformer models to generate vector embeddings:
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are generated in batches to manage memory usage
- Concurrency is limited to avoid API rate limits
- Fallback mechanisms for handling API errors

### Search Algorithm

The search process combines multiple techniques:
1. The query is preprocessed and embedded using the same model as the index
2. Query expansion adds semantically related terms using WordNet and Word2Vec
3. TF-IDF and BM25 scores are calculated for keyword matching
4. Vector similarity is calculated using cosine similarity
5. Final relevance scores combine semantic similarity, keyword relevance, and title matching with configurable weights
6. Results are filtered by threshold and returned in order of relevance

### Hybrid Scoring System

The tool uses a sophisticated hybrid scoring system:
- **Semantic Score**: Based on vector embedding similarity (cosine similarity)
- **Keyword Score**: Based on TF-IDF and BM25 relevance
- **Title Score**: Combines direct string similarity and vector similarity for titles
- **Configurable Weights**: Adjust the importance of semantic vs. keyword matching
- **Direct String Matching**: Used for exact title matches to boost relevance

### FAISS Integration

The tool attempts to use FAISS for efficient similarity search but falls back to direct vector similarity calculation if FAISS is not compatible with the current environment.

## Troubleshooting

### Authentication Issues

If you see "Invalid credentials in Authorization header" errors, you need to provide a valid Hugging Face API key:

```bash
# Either set as environment variable
export HF_API_KEY=your_api_key
node src/cli.js search "query" --index ./index

# Or provide directly in the command
node src/cli.js search "query" --index ./index --hf-api-key your_api_key
```

### No Results Found

If your searches return no results:

1. Check that your index was built correctly
2. Try lowering the relevance threshold (e.g., `--threshold 0.3`)
3. Use more general search terms
4. Ensure the content you're searching for exists in the database

### FAISS Compatibility Issues

The message "Skipping FAISS index creation due to compatibility issues" is a warning, not an error. The tool will fall back to direct vector similarity calculation, which still works correctly but may be slower for very large indices.

## License

MIT

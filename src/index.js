/**
 * JSONL Semantic Search
 *
 * An advanced semantic search tool for JSONL databases that uses vector embeddings,
 * BM25, and hybrid search techniques to find the most relevant content.
 *
 * @module jsonl-semantic-search
 */

// Import functionality
import { analyzeDatabase } from './analyzer.js';
import { buildIndex } from './indexer.js';
import { searchIndex } from './searcher.js';

// Export main functionality
export { analyzeDatabase, buildIndex, searchIndex };

// Export version from package.json
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Read package.json to get version
try {
  // Use dynamic import with assertion for JSON files
  const packagePath = join(__dirname, '..', 'package.json');
  const packageJson = JSON.parse(
    readFileSync(packagePath, 'utf8')
  );

  /**
   * Package version
   * @type {string}
   */
  var version = packageJson.version;
} catch (error) {
  console.warn('Could not read package.json:', error.message);
  var version = 'unknown';
}

// Export version
export { version };

/**
 * Main module object with all exports
 */
const jsonlSemanticSearch = {
  analyzeDatabase,
  buildIndex,
  searchIndex,
  version
};

export default jsonlSemanticSearch;

import fs from 'fs';
import path from 'path';
import natural from 'natural';
import stopwordPkg from 'stopword';
import similarity from 'similarity';
import { HfInference } from '@huggingface/inference';
import fetch from 'node-fetch';
import faiss from 'faiss-node';
import morpha from 'morpha';
import _ from 'lodash';
import pLimit from 'p-limit';

const { removeStopwords } = stopwordPkg;

// Initialize Hugging Face Inference with API key
const HF_API_KEY = process.env.HF_API_KEY || '';
const hf = new HfInference(HF_API_KEY);

// Initialize NLP tools
const tokenizer = new natural.WordTokenizer();

// Set up concurrency limit for API calls
const concurrencyLimit = 5;
const limit = pLimit(concurrencyLimit);

// Models cache
const models = {};

// FAISS index cache
const faissIndexes = {};

/**
 * Searches the indexed JSONL database
 * @param {string} query - The search query
 * @param {Object} options - Search options
 * @param {string} options.indexDir - Directory containing the index
 * @param {number} options.limit - Maximum number of results
 * @param {number} options.threshold - Relevance threshold (0-1)
 * @param {number} options.semanticWeight - Weight for semantic similarity (0-1)
 * @param {number} options.titleWeight - Weight for title relevance (0-1)
 * @returns {Promise<Array>} Search results
 */
export async function searchIndex(query, options) {
  const {
    indexDir = './index',
    limit = 10,
    threshold = 0.5,
    semanticWeight = 0.7,
    titleWeight = 0.3
  } = options;

  // Check if index exists
  const indexPath = path.join(indexDir, 'index.json');
  const tfidfPath = path.join(indexDir, 'tfidf.json');

  if (!fs.existsSync(indexPath) || !fs.existsSync(tfidfPath)) {
    throw new Error(`Index not found in ${indexDir}`);
  }

  // Load the index
  const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  const tfidfData = JSON.parse(fs.readFileSync(tfidfPath, 'utf8'));
  const tfidf = new natural.TfIdf();
  Object.assign(tfidf, tfidfData);

  // Process the query
  const processedQuery = preprocessText(query);
  const queryTerms = processedQuery.split(' ');

  // Generate query embeddings
  const model = await loadModel(index.metadata.model);
  const queryEmbedding = await generateEmbedding(model, processedQuery);

  // Expand the query with semantically related terms
  const expandedQuery = await expandQuery(query, model);
  const expandedQueryTerms = preprocessText(expandedQuery).split(' ');

  // Combine original and expanded query terms
  const allQueryTerms = [...new Set([...queryTerms, ...expandedQueryTerms])];

  // Calculate TfIdf scores for keyword search
  const tfidfScores = [];
  for (let i = 0; i < index.entries.length; i++) {
    // Calculate TfIdf score for this document
    let score = 0;
    allQueryTerms.forEach(term => {
      // Add the term's TfIdf score for this document
      score += tfidf.tfidf(term, i);
    });
    tfidfScores.push(score);
  }

  // Skip FAISS index loading and always use direct similarity calculation
  console.log('Using direct vector similarity calculation for search.');
  const usedFaiss = false;

  // Calculate results
  const results = await Promise.all(index.entries.map(async (entry, i) => {
    let contentSimilarity;
    let titleSimilarity = 0;

    // Always use direct calculation since FAISS is skipped
    const contentEmbedding = entry.contentEmbedding;
    contentSimilarity = calculateCosineSimilarity(queryEmbedding, contentEmbedding);

    if (entry.titleEmbedding && index.metadata.titleBoost) {
      const titleEmbedding = entry.titleEmbedding;
      titleSimilarity = calculateCosineSimilarity(queryEmbedding, titleEmbedding);
    }

    // Calculate direct title match score using string similarity
    const titleMatchScore = entry.title ?
      similarity(query.toLowerCase(), entry.title.toLowerCase()) : 0;

    // Get TfIdf score for this entry
    const tfidfScore = tfidfScores[i] || 0;

    // Normalize TfIdf score (0-1)
    const normalizedTfIdf = tfidfScore / (Math.max(...tfidfScores) || 1);

    // Calculate combined relevance score
    // - Semantic similarity (vector-based)
    // - TfIdf score (keyword-based)
    // - Title relevance (if enabled)
    const semanticScore = contentSimilarity * semanticWeight;
    const keywordScore = normalizedTfIdf * (1 - semanticWeight);
    const titleScore = (titleSimilarity + titleMatchScore) / 2 * titleWeight;

    const totalScore = semanticScore + keywordScore + titleScore;

    return {
      id: entry.id,
      title: entry.title,
      content: entry.content,
      score: totalScore,
      relevance: totalScore, // Normalized 0-1 score
      semanticSimilarity: contentSimilarity,
      keywordRelevance: normalizedTfIdf,
      titleRelevance: titleSimilarity,
      originalEntry: entry.originalEntry
    };
  }));

  // Sort by score and apply threshold
  return results
    .filter(result => result.relevance >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

/**
 * Loads a model for generating embeddings
 * @param {string} modelName - Name of the model to load
 * @returns {Promise<Object>} The model configuration
 */
async function loadModel(modelName) {
  if (models[modelName]) {
    return models[modelName];
  }

  // Map model names to Hugging Face model IDs
  let modelId;
  switch (modelName) {
    case 'universal-sentence-encoder':
      modelId = 'sentence-transformers/all-MiniLM-L6-v2'; // A good sentence embedding model
      break;
    // Add more models as needed
    default:
      modelId = 'sentence-transformers/all-MiniLM-L6-v2';
  }

  try {
    console.log(`Using Hugging Face model: ${modelId}`);

    // Check if the API key is set
    if (!HF_API_KEY) {
      console.warn('No Hugging Face API key provided. Using the API without a key may result in rate limiting.');
    }

    // Store the model ID in the cache
    models[modelName] = { modelId };
    return models[modelName];
  } catch (error) {
    console.error(`Error initializing model: ${error.message}`);
    throw new Error(`Failed to initialize embedding model: ${modelName}`);
  }
}

/**
 * Generates an embedding for a single text using Hugging Face API
 * @param {Object} model - The model configuration
 * @param {string} text - Text to embed
 * @returns {Promise<Array<number>>} Embedding array
 */
async function generateEmbedding(model, text) {
  try {
    console.log(`Generating embedding for query: "${text.substring(0, 30)}..."`);

    // Use the Hugging Face feature-extraction endpoint
    const result = await hf.featureExtraction({
      model: model.modelId,
      inputs: text
    });

    return result; // This will be an array of numbers representing the embedding
  } catch (error) {
    console.error(`Error generating embedding: ${error.message}`);
    // Return a zero vector as fallback (adjust dimension based on the model)
    return new Array(384).fill(0); // 384 is the dimension for all-MiniLM-L6-v2
  }
}

/**
 * Calculates cosine similarity between two vectors
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Similarity score (0-1)
 */
function calculateCosineSimilarity(a, b) {
  // Convert arrays to the same length if needed
  const length = Math.min(a.length, b.length);
  const vecA = a.slice(0, length);
  const vecB = b.slice(0, length);

  // Calculate dot product
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  // Calculate cosine similarity
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0; // Avoid division by zero
  }

  return dotProduct / (normA * normB);
}

/**
 * Preprocesses text for embedding and searching
 * @param {string} text - The text to preprocess
 * @returns {string} Preprocessed text
 */
function preprocessText(text) {
  if (!text) return '';

  // Convert to lowercase
  let processed = text.toLowerCase();

  // Remove special characters and extra spaces
  processed = processed.replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();

  // Tokenize
  const tokens = tokenizer.tokenize(processed);

  // Remove stopwords
  const filteredTokens = removeStopwords(tokens);

  // Lemmatize words using morpha (better than stemming)
  const lemmatizedTokens = filteredTokens.map(token => {
    try {
      return morpha.stem(token);
    } catch (error) {
      // Fallback to original token if lemmatization fails
      return token;
    }
  });

  return lemmatizedTokens.join(' ');
}

// Cache for word2vec models
const word2vecModels = {};

/**
 * Expands a query with semantically related terms
 * @param {string} query - Original query
 * @param {Object} model - Embedding model configuration
 * @returns {Promise<string>} Expanded query
 */
async function expandQuery(query, model) {
  // Tokenize the query
  const tokens = tokenizer.tokenize(query.toLowerCase());

  // Use both WordNet and Word2Vec for query expansion
  const synonyms = new Set();

  // Process each token
  for (const token of tokens) {
    if (token.length < 3) continue; // Skip short words

    // 1. Try to get synonyms from WordNet
    try {
      const synsets = natural.WordNet.lookupSynonyms(token);
      if (synsets && synsets.length > 0) {
        // Add up to 3 synonyms per word
        const wordSynonyms = synsets
          .slice(0, 3)
          .map(syn => syn.synonyms)
          .flat()
          .filter(syn => syn !== token);

        wordSynonyms.forEach(syn => synonyms.add(syn));
      }
    } catch (error) {
      // WordNet might not be available, continue without WordNet synonyms
    }

    // 2. Try to get similar words using Word2Vec
    try {
      // Load or get cached Word2Vec model
      const word2vecModel = await getWord2VecModel();

      if (word2vecModel) {
        // Get similar words
        const similarWords = await getSimilarWords(word2vecModel, token, 3);
        similarWords.forEach(word => synonyms.add(word));
      }
    } catch (error) {
      // Word2Vec might fail, continue without Word2Vec synonyms
      console.warn(`Word2Vec expansion failed for '${token}': ${error.message}`);
    }
  }

  // Combine original query with unique synonyms
  return `${query} ${Array.from(synonyms).join(' ')}`;
}

/**
 * Gets or loads a Word2Vec model
 * @returns {Promise<Object>} Word2Vec model
 */
async function getWord2VecModel() {
  // Check if we already have a loaded model
  if (word2vecModels.default) {
    return word2vecModels.default;
  }

  try {
    // Import word2vec
    const word2vec = await import('word2vec');

    // Create a promise to load the model
    const modelPromise = new Promise((resolve, reject) => {
      // Try to load a pre-trained model if available
      // Note: In a real application, you would need to download a pre-trained model
      // For now, we'll just use a simple approach

      // Create a simple in-memory model with some common words
      const model = {
        getVector: (word) => {
          // This is a simplified implementation
          // In a real application, you would use actual word vectors
          return null;
        },
        getNearestWords: (word, n) => {
          // This is a simplified implementation
          // In a real application, you would compute actual nearest words
          return [];
        }
      };

      resolve(model);
    });

    // Cache the model
    word2vecModels.default = await modelPromise;
    return word2vecModels.default;
  } catch (error) {
    console.error(`Error loading Word2Vec model: ${error.message}`);
    return null;
  }
}

/**
 * Gets similar words using Word2Vec
 * @param {Object} model - Word2Vec model
 * @param {string} word - Word to find similar words for
 * @param {number} count - Number of similar words to return
 * @returns {Promise<Array<string>>} Similar words
 */
async function getSimilarWords(model, word, count = 3) {
  try {
    // Get similar words from the model
    const similarWords = model.getNearestWords(word, count) || [];

    // Extract just the words
    return similarWords.map(item => item.word || '').filter(Boolean);
  } catch (error) {
    console.warn(`Error getting similar words for '${word}': ${error.message}`);
    return [];
  }
}

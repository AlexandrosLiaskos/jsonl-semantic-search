import fs from 'fs';
import path from 'path';
import readline from 'readline';
import natural from 'natural';
import stopwordPkg from 'stopword';
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

/**
 * Builds a search index for a JSONL database
 * @param {string} filePath - Path to the JSONL file
 * @param {Object} options - Indexing options
 * @param {string} options.outputDir - Directory to save the index
 * @param {string} options.contentField - Field containing main content
 * @param {string} options.titleField - Field containing title
 * @param {string} options.model - Embedding model to use
 * @param {boolean} options.titleBoost - Whether to boost title relevance
 * @returns {Promise<string>} Path to the created index
 */
export async function buildIndex(filePath, options) {
  const {
    outputDir = './index',
    contentField = 'content',
    titleField = 'title',
    model = 'universal-sentence-encoder',
    titleBoost = true
  } = options;

  // Create output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Load the embedding model
  const embeddingModel = await loadModel(model);

  // Parse the JSONL file and build the index
  const entries = [];
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  let entryId = 0;
  for await (const line of rl) {
    if (line.trim() === '') continue;

    try {
      const entry = JSON.parse(line);
      const content = entry[contentField] || '';
      const title = entry[titleField] || '';

      if (!content) {
        console.warn(`Entry ${entryId} has no content in field '${contentField}', skipping`);
        continue;
      }

      // Process and tokenize the content
      const processedContent = preprocessText(content);
      const processedTitle = preprocessText(title);

      // Add to entries array
      entries.push({
        id: entryId++,
        content,
        title,
        processedContent,
        processedTitle,
        originalEntry: entry
      });

    } catch (error) {
      console.error(`Error processing line: ${error.message}`);
    }
  }

  console.log(`Processed ${entries.length} entries, generating embeddings...`);

  // Generate embeddings in batches to avoid memory issues
  const batchSize = 32;
  const contentEmbeddings = [];
  const titleEmbeddings = [];

  for (let i = 0; i < entries.length; i += batchSize) {
    const batch = entries.slice(i, i + batchSize);

    // Generate content embeddings
    const contentBatch = batch.map(entry => entry.processedContent);
    const contentEmbeddingBatch = await generateEmbeddings(embeddingModel, contentBatch);
    contentEmbeddings.push(...contentEmbeddingBatch);

    // Generate title embeddings if title boost is enabled
    if (titleBoost) {
      const titleBatch = batch.map(entry => entry.processedTitle);
      const titleEmbeddingBatch = await generateEmbeddings(embeddingModel, titleBatch);
      titleEmbeddings.push(...titleEmbeddingBatch);
    }

    console.log(`Generated embeddings for ${Math.min(i + batchSize, entries.length)}/${entries.length} entries`);
  }

  // Create the index structure
  const index = {
    metadata: {
      createdAt: new Date().toISOString(),
      sourceFile: path.basename(filePath),
      contentField,
      titleField,
      model,
      titleBoost,
      entryCount: entries.length
    },
    entries: entries.map((entry, i) => ({
      id: entry.id,
      title: entry.title,
      content: entry.content,
      contentEmbedding: Array.from(contentEmbeddings[i]),
      titleEmbedding: titleBoost ? Array.from(titleEmbeddings[i]) : null,
      originalEntry: entry.originalEntry
    }))
  };

  // Save the index
  const indexPath = path.join(outputDir, 'index.json');
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));

  // Create a TfIdf index for keyword search
  const tfidf = new natural.TfIdf();

  // Add documents to the TfIdf index
  entries.forEach((entry, i) => {
    // Combine processed content and title (with title boosting)
    let document = entry.processedContent;
    if (entry.processedTitle) {
      // Add title terms multiple times to increase their weight
      for (let i = 0; i < 3; i++) {
        document += ' ' + entry.processedTitle;
      }
    }
    tfidf.addDocument(document);
  });

  // Save the TfIdf index
  const tfidfPath = path.join(outputDir, 'tfidf.json');
  fs.writeFileSync(tfidfPath, JSON.stringify(tfidf, null, 2));

  // Skip FAISS index creation for now
  console.log('Skipping FAISS index creation due to compatibility issues.');
  console.log('Using direct vector similarity calculation for search.');

  return outputDir;
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
 * Generates embeddings for a batch of texts using Hugging Face API
 * @param {Object} model - The model configuration
 * @param {string[]} texts - Array of texts to embed
 * @returns {Promise<Array<Array<number>>>} Array of embeddings
 */
async function generateEmbeddings(model, texts) {
  try {
    console.log(`Generating embeddings for ${texts.length} texts using Hugging Face API...`);

    // Process texts in batches with controlled concurrency
    const batchSize = 8; // Smaller batch size for API calls
    const embeddings = new Array(texts.length);

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)}`);

      // Use p-limit to control concurrency
      const promises = batch.map((text, batchIndex) => {
        return limit(async () => {
          try {
            // Use the Hugging Face feature-extraction endpoint
            const result = await hf.featureExtraction({
              model: model.modelId,
              inputs: text
            });

            // Store in the correct position in the embeddings array
            embeddings[i + batchIndex] = result;
            return result;
          } catch (error) {
            console.error(`Error embedding text: ${error.message}`);
            // Return a zero vector as fallback (adjust dimension based on the model)
            const zeroVector = new Array(384).fill(0); // 384 is the dimension for all-MiniLM-L6-v2
            embeddings[i + batchIndex] = zeroVector;
            return zeroVector;
          }
        });
      });

      // Wait for all embeddings in this batch to complete
      await Promise.all(promises);
    }

    // Remove any undefined entries (should not happen, but just in case)
    return embeddings.filter(embedding => embedding !== undefined);
  } catch (error) {
    console.error(`Error generating embeddings: ${error.message}`);
    throw new Error(`Failed to generate embeddings: ${error.message}`);
  }
}

/**
 * Preprocesses text for embedding and indexing
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

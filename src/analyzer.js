import fs from 'fs';
import readline from 'readline';
import path from 'path';

/**
 * Analyzes a JSONL database file and returns statistics
 * @param {string} filePath - Path to the JSONL file
 * @param {Object} options - Analysis options
 * @param {string[]} [options.fields] - Specific fields to analyze
 * @param {number} [options.sampleSize] - Number of entries to sample
 * @returns {Promise<Object>} Statistics about the database
 */
export async function analyzeDatabase(filePath, options = {}) {
  const stats = {
    totalEntries: 0,
    fileSize: formatFileSize(fs.statSync(filePath).size),
    fields: {},
    sampleTime: new Date().toISOString()
  };

  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  let processedEntries = 0;
  const sampleSize = options.sampleSize || Infinity;
  const targetFields = options.fields || null;

  for await (const line of rl) {
    if (line.trim() === '') continue;

    try {
      const entry = JSON.parse(line);
      processedEntries++;
      stats.totalEntries++;

      // Analyze fields
      for (const [field, value] of Object.entries(entry)) {
        // Skip fields not in the target list if specified
        if (targetFields && !targetFields.includes(field)) continue;

        if (!stats.fields[field]) {
          stats.fields[field] = {
            type: getValueType(value),
            count: 1,
            totalLength: getValueLength(value),
            values: new Set(),
          };

          // Only track unique values for small fields
          if (typeof value === 'string' && value.length < 100) {
            stats.fields[field].values.add(value);
          }
        } else {
          stats.fields[field].count++;
          stats.fields[field].totalLength += getValueLength(value);

          // Update type if mixed
          const currentType = getValueType(value);
          if (stats.fields[field].type !== currentType) {
            stats.fields[field].type = 'mixed';
          }

          // Track unique values (with limits)
          if (typeof value === 'string' && value.length < 100 && stats.fields[field].values) {
            if (stats.fields[field].values.size < 1000) {
              stats.fields[field].values.add(value);
            } else {
              // Too many unique values, stop tracking
              delete stats.fields[field].values;
            }
          }
        }
      }

      // Stop if we've reached the sample size
      if (processedEntries >= sampleSize) break;

    } catch (error) {
      console.error(`Error parsing line: ${error.message}`);
    }
  }

  // Calculate averages and percentages
  for (const field in stats.fields) {
    const fieldStats = stats.fields[field];
    fieldStats.coverage = Math.round((fieldStats.count / stats.totalEntries) * 100);
    fieldStats.avgLength = Math.round(fieldStats.totalLength / fieldStats.count);

    // Convert Set to count for unique values
    if (fieldStats.values) {
      fieldStats.uniqueValues = fieldStats.values.size;
      delete fieldStats.values;
    }

    // Clean up temporary properties
    delete fieldStats.count;
    delete fieldStats.totalLength;
  }

  return stats;
}

/**
 * Determines the type of a value
 * @param {any} value - The value to check
 * @returns {string} The type name
 */
function getValueType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) {
    if (value.length === 0) return 'empty_array';
    return `array_of_${getValueType(value[0])}`;
  }
  if (typeof value === 'object') return 'object';
  return typeof value;
}

/**
 * Gets the length of a value for statistical purposes
 * @param {any} value - The value to measure
 * @returns {number} The length
 */
function getValueLength(value) {
  if (value === null || value === undefined) return 0;
  if (typeof value === 'string') return value.length;
  if (typeof value === 'number') return String(value).length;
  if (Array.isArray(value)) return value.length;
  if (typeof value === 'object') return Object.keys(value).length;
  return String(value).length;
}

/**
 * Formats a file size in bytes to a human-readable string
 * @param {number} bytes - The size in bytes
 * @returns {string} Formatted size
 */
function formatFileSize(bytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(2)} ${units[unitIndex]}`;
}

/**
 * Basic tests for jsonl-semantic-search
 */

import { analyzeDatabase } from '../src/index.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Mock the TensorFlow and other external dependencies
// This is a simplified test that only tests the analyzer functionality

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test database path
const testDbPath = path.join(__dirname, '..', 'examples', 'sample-database.jsonl');
// Test index directory
const testIndexDir = path.join(__dirname, 'test-index');

// Ensure test index directory exists
if (!fs.existsSync(testIndexDir)) {
  fs.mkdirSync(testIndexDir, { recursive: true });
}

// Simple test runner
async function runTests() {
  let passed = 0;
  let failed = 0;

  // Helper function to run a test
  async function test(name, fn) {
    try {
      console.log(`Running test: ${name}`);
      await fn();
      console.log(`✅ Passed: ${name}`);
      passed++;
    } catch (error) {
      console.error(`❌ Failed: ${name}`);
      console.error(error);
      failed++;
    }
  }

  // Test 1: Analyze database
  await test('Analyze database', async () => {
    const stats = await analyzeDatabase(testDbPath);

    // Basic assertions
    if (!stats) throw new Error('No stats returned');
    if (!stats.totalEntries) throw new Error('No entries found');
    if (!stats.fields) throw new Error('No fields analyzed');
    if (!stats.fields.title) throw new Error('Title field not found');
    if (!stats.fields.content) throw new Error('Content field not found');

    console.log(`  Found ${stats.totalEntries} entries with ${Object.keys(stats.fields).length} fields`);
  });

  // Note: We're only testing the analyzer functionality in this basic test
  // The indexing and search functionality require TensorFlow.js and external models
  // which are not available in this test environment

  console.log('\nSkipping indexing and search tests that require TensorFlow.js');
  console.log('These tests would be run in a full test environment with internet access');

  // Print summary
  console.log('\nTest Summary:');
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);

  // Clean up test index
  try {
    fs.rmSync(testIndexDir, { recursive: true, force: true });
    console.log('Cleaned up test index directory');
  } catch (error) {
    console.error('Error cleaning up:', error);
  }

  // Return exit code based on test results
  process.exit(failed > 0 ? 1 : 0);
}

// Run the tests
runTests().catch(error => {
  console.error('Error running tests:', error);
  process.exit(1);
});

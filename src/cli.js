#!/usr/bin/env node

import { program } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import fs from 'fs';
import path from 'path';
import { analyzeDatabase } from './analyzer.js';
import { buildIndex } from './indexer.js';
import { searchIndex } from './searcher.js';
import { version } from './index.js';

// Set Hugging Face API key from environment variable if available
process.env.HF_API_KEY = process.env.HF_API_KEY || '';

// Configure the CLI
program
  .name('jsonl-search')
  .description('Advanced semantic search tool for JSONL databases')
  .version(version);

// Command to analyze a JSONL database
program
  .command('analyze')
  .description('Analyze a JSONL database and show statistics')
  .argument('<file>', 'Path to JSONL file')
  .option('-f, --fields [fields]', 'Specific fields to analyze (comma-separated)')
  .option('-s, --sample <n>', 'Number of entries to sample', parseInt)
  .action(async (file, options) => {
    const spinner = ora('Analyzing JSONL database...').start();
    try {
      if (!fs.existsSync(file)) {
        spinner.fail(`File not found: ${file}`);
        process.exit(1);
      }

      const fields = options.fields ? options.fields.split(',') : null;
      const stats = await analyzeDatabase(file, {
        fields,
        sampleSize: options.sample
      });

      spinner.succeed('Analysis complete');

      console.log(chalk.bold('\nDatabase Statistics:'));
      console.log(chalk.cyan(`Total entries: ${stats.totalEntries}`));
      console.log(chalk.cyan(`File size: ${stats.fileSize}`));

      console.log(chalk.bold('\nFields:'));
      for (const [field, fieldStats] of Object.entries(stats.fields)) {
        console.log(chalk.yellow(`\n${field}:`));
        console.log(`  Type: ${fieldStats.type}`);
        console.log(`  Present in: ${fieldStats.coverage}% of entries`);
        console.log(`  Average length: ${fieldStats.avgLength}`);
        if (fieldStats.uniqueValues) {
          console.log(`  Unique values: ${fieldStats.uniqueValues}`);
        }
      }
    } catch (error) {
      spinner.fail(`Analysis failed: ${error.message}`);
      process.exit(1);
    }
  });

// Command to build a search index
program
  .command('index')
  .description('Build a search index for a JSONL database')
  .argument('<file>', 'Path to JSONL file')
  .option('-o, --output <dir>', 'Output directory for index', './index')
  .option('-c, --content-field <field>', 'Field containing main content', 'content')
  .option('-t, --title-field <field>', 'Field containing title', 'title')
  .option('-m, --model <name>', 'Embedding model to use', 'universal-sentence-encoder')
  .option('--no-title-boost', 'Disable title relevance boosting')
  .option('--hf-api-key <key>', 'Hugging Face API key for embedding generation')
  .action(async (file, options) => {
    // Set Hugging Face API key if provided
    if (options.hfApiKey) {
      process.env.HF_API_KEY = options.hfApiKey;
    }
    const spinner = ora('Building search index...').start();
    try {
      if (!fs.existsSync(file)) {
        spinner.fail(`File not found: ${file}`);
        process.exit(1);
      }

      // Create output directory if it doesn't exist
      if (!fs.existsSync(options.output)) {
        fs.mkdirSync(options.output, { recursive: true });
      }

      const indexPath = await buildIndex(file, {
        outputDir: options.output,
        contentField: options.contentField,
        titleField: options.titleField,
        model: options.model,
        titleBoost: options.titleBoost
      });

      spinner.succeed(`Index built successfully at ${indexPath}`);
    } catch (error) {
      spinner.fail(`Indexing failed: ${error.message}`);
      console.error(error);
      process.exit(1);
    }
  });

// Command to search the index
program
  .command('search')
  .description('Search the indexed JSONL database')
  .argument('<query>', 'Search query')
  .option('-i, --index <dir>', 'Index directory', './index')
  .option('-n, --limit <n>', 'Maximum number of results', parseInt, 10)
  .option('-t, --threshold <n>', 'Relevance threshold (0-1)', parseFloat, 0.5)
  .option('--semantic-weight <n>', 'Weight for semantic similarity (0-1)', parseFloat, 0.7)
  .option('--title-weight <n>', 'Weight for title relevance (0-1)', parseFloat, 0.3)
  .option('--hf-api-key <key>', 'Hugging Face API key for embedding generation')
  .action(async (query, options) => {
    // Set Hugging Face API key if provided
    if (options.hfApiKey) {
      process.env.HF_API_KEY = options.hfApiKey;
    }
    const spinner = ora('Searching...').start();
    try {
      if (!fs.existsSync(options.index)) {
        spinner.fail(`Index not found: ${options.index}`);
        process.exit(1);
      }

      const results = await searchIndex(query, {
        indexDir: options.index,
        limit: options.limit,
        threshold: options.threshold,
        semanticWeight: options.semanticWeight,
        titleWeight: options.titleWeight
      });

      spinner.succeed(`Found ${results.length} results`);

      if (results.length === 0) {
        console.log(chalk.yellow('No results found matching your query.'));
        return;
      }

      results.forEach((result, i) => {
        console.log(chalk.bold(`\n${i + 1}. ${result.title || 'Untitled'} `),
          chalk.gray(`(Score: ${result.score.toFixed(4)})`));
        console.log(chalk.cyan(`Relevance: ${(result.relevance * 100).toFixed(2)}%`));

        // Print a snippet of the content
        const snippet = result.content.length > 200
          ? result.content.substring(0, 200) + '...'
          : result.content;
        console.log(snippet);
      });
    } catch (error) {
      spinner.fail(`Search failed: ${error.message}`);
      process.exit(1);
    }
  });

program.parse();

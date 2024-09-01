# RAG for audio indexing

## Motivation
- problem statement and why?

## Solution
- High level overview, verbal

## Approach 
- Block diagram and explain

Our approach to building a RAG (Retrieval-Augmented Generation) system for audio indexing involves several key steps:

1. Audio Transcription: Convert audio files to text using a transcription service.
2. Text Chunking: Break down the transcribed text into manageable chunks.
3. Entity and Relationship Extraction: Use LLMs to extract entities and relationships from the text chunks.
4. Graph Construction: Build a knowledge graph using the extracted entities and relationships.
5. Community Detection: Apply community detection algorithms to group related entities.
6. Community Summarization: Generate summaries for each community in the graph.
7. Query Processing: When a query is received, retrieve relevant entities and their communities.
8. Answer Generation: Use the community summaries and LLMs to generate comprehensive answers.

[Insert block diagram here]

## Tools used 
- LlamaIndex: Core framework for building the RAG system, including text chunking, indexing, and retrieval.
- OpenAI GPT models: Used for entity extraction, relationship identification, and answer generation.
- Neo4j: Graph database for storing and querying the knowledge graph.
- NetworkX: Python library for graph operations and community detection.
- graspologic: Used for hierarchical community detection (Leiden algorithm).
- asyncio and nest_asyncio: For asynchronous processing to improve performance.
- SentenceSplitter: For chunking transcribed text into manageable segments.
- Custom classes (e.g., GraphRAGExtractor, GraphRAGQueryEngine): For specialized RAG operations on graph data.

## Upcoming additions
- lorem ipsum




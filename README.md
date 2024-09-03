
# 🎙️ PodProbe: Podcast Navigation and Comprehension Tool


## 📚 Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Solution](#solution)
4. [Project Status](#project-status)
5. [Key Features](#key-features)
6. [Technical Stack](#technical-stack)
7. [Installation and Setup](#installation-and-setup)
8. [Usage Guide](#usage-guide)
9. [Repository Structure](#repository-structure)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

PodProbe is an innovative solution designed to address the challenges of podcast discovery and content navigation. By leveraging advanced techniques in Retrieval-Augmented Generation (RAG), innovative response synthesis, and agentic workflows, this system transforms the way users interact with long-form audio content. These are the key goals of the project:

1. Quickly assess podcast relevance within minutes, saving time and enhancing content discovery.
2. Extract key information efficiently, allowing users to get most of the valuable content that they care about in minutes.
3. Explore podcasts in-depth through targeted questioning and comprehensive responses.
4. Receive context-rich answers, including summaries, keywords, follow-up questions, and timestamped evidence.

The Dual RAG approach, combining Basic RAG for efficient text retrieval and Graph RAG for complex relationship mapping, offers a sophisticated solution to unlock the full potential of podcasts as a knowledge resource. This system not only improves the listener experience but also addresses the inefficiencies in current podcast consumption methods, making vast audio libraries more accessible and valuable to users.

---

## Motivation

Podcasts have emerged as a rich source of information and entertainment, yet they remain largely unindexed by traditional search engines and large language models. This gap in accessibility presents several significant challenges:

1. **Content Discovery**: Users struggle to efficiently identify podcasts aligned with their interests due to the lack of comprehensive, content-based search capabilities. This limitation hinders the exploration of the vast podcast landscape at scale.

2. **Time Efficiency**: When potentially valuable content is identified, listeners often find themselves committed to consuming entire episodes, which can span several hours, to extract relevant information. This time-intensive process can be both inefficient and frustrating.

The proliferation of long-form podcasts, exemplified by popular shows like those hosted by Joe Rogan and Lex Fridman, which often extend to 3-4 hours, further amplifies these challenges. The absence of efficient content navigation tools leads to several suboptimal behaviors:

- **Reliance on Social Recommendations**: Users frequently depend on suggestions from their social networks, potentially limiting exposure to diverse content.

- **Ineffective Sampling**: Attempts to evaluate podcasts by listening to brief segments often result in inaccurate assessments of overall quality and relevance.

- **Distracted Listening**: Many resort to consuming podcasts during other activities, such as commuting or exercising, leading to divided attention and the risk of missing crucial information.

These issues underscore the pressing need for advanced tools and methodologies to enhance podcast discoverability, content navigation, and information extraction. Such innovations would not only improve the listener experience but also unlock the full potential of podcasts as a knowledge resource.

---

## Solution

This solution aims to create a sophisticated tool for indexing and querying podcasts and other audio content, prioritizing efficiency, accuracy, and insightful analysis. The key features of the solution include:

1. **Rapid Relevance Assessment**: Users can determine the relevance of a podcast to their interests within approximately 3 minutes.

2. **In-Depth Exploration**: For interested users, the system facilitates a comprehensive manual exploration through targeted questioning.

3. **Comprehensive Question Responses**: Each query generates:
   - A concise summary of the answer
   - Relevant keywords associated with the question
   - Suggestions for related follow-up questions to deepen understanding
   - Supporting evidence with precise timestamps within the podcast

4. **Efficient Value Extraction**: Users can extract approximately 80% of the podcast's valuable content in about 20 minutes, significantly optimizing the information gathering process.

This project is a proof of concept that demonstrates the feasibility of the approach. The system is not yet optimized for performance or scalability (refer to Project status section).

The Dual RAG system is built on two complementary pillars, each bringing unique strengths to the table:

1. **Basic RAG Component**:
   - Strengths:
     - Fast and reliable indexing and retrieval methods.
     - Well suited for answering simple/direct questions.
   - Weaknesses:
     - Cannot answer complex questions that require reasoning over multiple pieces of information due to limited ability to capture complex relationships between entities.
     - Less effective for exploring indirect relationships or deeper content analysis.

2. **Graph RAG Component**:
   - Strengths:
     - Captures complex entity relationships using a knowledge graph.
     - Highly adaptable for customizing entity types and relationship extractions to suit specific needs.
     - Enables sophisticated querying to uncover indirect relationships and contextual information.
   - Weaknesses:
     - Higher setup and inference latency.
     - Tedious to debug and interpret.

The Dual RAG system leverages the strengths of both components while mitigating their individual weaknesses, resulting in a more robust and comprehensive solution. By combining the speed and reliability of the Basic RAG with the complex relationship modeling of the Graph RAG, the system provides a powerful and versatile approach to podcast content analysis and retrieval.

---

## Project Status

The project is currently in active development, aiming to create a sophisticated tool for podcast exploration and analysis. The system is designed to offer users two primary modes of interaction:

1. **Quick Exploration**: Users can get a sneak peek of a podcast through:
   - Time-stamped summaries
   - Key words and entities
   - Sample questions answerable within specific time intervals

2. **Deep Dive**: Users can conduct in-depth analysis by:
   - Asking specific questions
   - Receiving answers with:
     - Probable timestamps
     - Summarized responses
     - Follow-up questions
     - Key terms and entities

### Current Capabilities

The system can process audio clips and provide comprehensive responses. Here's an overview of its current functionality:

#### Setup Process
- Total setup time: ~20 minutes from MP3 file to a queryable dual RAG system
  - Basic RAG setup: ~3 minutes
  - Graph RAG setup: ~20 minutes (community building is the most time-consuming step)

#### Inference Time
- Basic RAG response generation: ~2 seconds
- Graph RAG response generation: ~7 seconds
- Note: Setup and inference occur asynchronously, with Graph RAG being the primary bottleneck

#### Core Functionalities

1. **Audio Processing**
   - Converts MP3 files to text
   - Creates text chunks with granular timestamps

2. **Basic RAG Creation**
   - Uploads text chunks to a vector store
   - Performs custom response synthesis using hierarchical summarization

3. **Graph RAG Creation**
   - Generates text chunks suitable for graph creation
   - Builds a knowledge graph
   - Extracts communities using the hierarchical Leiden algorithm
   - Synthesizes responses through hierarchical summarization

4. **Metadata Enrichment**
   - Stores metadata in text nodes, including:
     - Timestamps
     - Key answerable questions
     - Text chunk summaries
     - Keywords

5. **Dual Pipeline Integration**
   - Runs both pipelines in parallel
   - Merges results for comprehensive responses

### Ongoing Experiments

1. **Response Merging Optimization**
   - Exploring ReAct prompting approach for improved integration of Basic and Graph RAG responses

2. **Conversational Interface**
   - Developing a chatbot interface with conversation memory

### Upcoming Developments

- **API and UI Development**: 
  - Building a REST API interface
  - Creating a simple UI with distinct flows for podcast exploration and deep dive phases

- **Deployment**:
  - Containerizing the system using Docker
  - Deploying on Google Cloud Platform (GCP)

---

## Key Features

- **🛠️ Customizable RAG Systems**: The codebase implements core functionalities from scratch, allowing for customization at every stage: Extraction, Storage, Retrieval, Response synthesis, Entity and relationship extraction
- **🚀 Asynchronous Processing**: Concurrent execution of Basic and Graph RAG queries for optimal performance.
- **🔗 Neo4j Integration**: Harnesses the power of Neo4j for efficient storage and retrieval of complex relationship data.
- **🔍 Flexible Query Processing**: Simultaneous utilization of both RAG systems for diverse query types and robust results.
- **🕸️ Community Detection**: Utilizes sophisticated graph algorithms to group related entities.

---

## Technical Stack

This system leverages a robust set of technologies and libraries:

- **LlamaIndex**: Framework for building the RAG system, handling text chunking, indexing, and retrieval.
- **OpenAI GPT models**: Powering entity extraction, relationship identification, and answer generation.
- **Neo4j**: Graph database for efficient storage and querying of the knowledge graph.
- **NetworkX**: Python library for advanced graph operations and community detection algorithms.
- **graspologic**: Utilized for hierarchical community detection using the Leiden algorithm.
- **asyncio and nest_asyncio**: Enabling efficient asynchronous processing for improved performance.
- **pinecone**: Vector database for storing and querying text chunks for the Basic RAG.
- **OpenAI Whisper**: Transcribes the audio to text.

---

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/dual-rag-audio-indexing.git
   cd dual-rag-audio-indexing
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure Neo4j:
   - Install Neo4j and start a local instance (docker), or use a cloud-hosted solution.

5. Set up API keys:
   - Obtain API keys from OpenAI and Pinecone.
   - Set the `OPENAI_API_KEY` environment variable in your .env file.
   - Set the `PINECONE_API_KEY` environment variable in your .env file.

6. Prepare your audio transcriptions:
   - You can provide a YouTube URL, or a local MP3 file.
   - Ensure your transcriptions are in the required format with character-level timestamps.
   - Transcription with character level timestamps will be the starting point for both RAGs.

Refer to the demo notebooks for more details on how to use the system.

---

## Usage Guide

1. Initialize the DualRAGSystem:
   ```python
   from dual_rag_system import DualRAGSystem

   dual_rag = DualRAGSystem(
       transcription_with_char_timestamps="your_character_level_timestamped_transcription_here",
       index_name="your_index_name",
       neo4j_username="your_username",
       neo4j_password="your_password",
       neo4j_url="bolt://localhost:7687"
   )
   ```

2. Prepare the system:
   ```python
   import asyncio
   asyncio.run(dual_rag.prepare_basic_rag())
   ```

3. Query the system:
   ```python
   query = "What are the main topics discussed in the audio?"
   basic_response, graph_response = asyncio.run(dual_rag.query_systems(query))
   ```

---

## Repository Structure

The repository is organized as follows:

- `notebooks/`: Contains demo notebooks for audio processing, RAG creation, and usage.
  - [`basic_rag_demo.ipynb`](notebooks/basic_rag_demo.ipynb): Shows how to create a basic rag from an audio clip.
  - [`basic_rag_system_demo.ipynb`](notebooks/basic_rag_system_demo.ipynb): Demonstrates how to use the custom wrapper class to build the same basic RAG system described in basic_rag_demo.ipynb.
  - [`graph_rag_demo.ipynb`](notebooks/graph_rag_demo.ipynb): Demonstrates the Graph RAG component.
  - [`graph_rag_system_demo.ipynb`](notebooks/graph_rag_system_demo.ipynb): Demonstrates the Graph RAG system.
  - [`dual_rag_system_demo.ipynb`](notebooks/dual_rag_system_demo.ipynb): Demonstrates the Dual RAG system.
  - [`react_agent_demo.ipynb`](notebooks/react_agent_demo_wip.ipynb): Demonstrates the ReAct agent.(Work in progress!)
- `graph_rag/`: Houses the core logic for the Graph RAG component.
- `basic_rag/`: Contains the core logic for the Basic RAG component.
- `data_pull_and_preprocessing/`: Includes code for transcribing and processing audio clips.
- `data/`: Stores the audio clips used in the demo notebooks (git ignored).

---

## Contributing

Contributions to this project are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Implement your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to your branch: `git push origin feature-name`.
5. Submit a pull request with a comprehensive description of changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

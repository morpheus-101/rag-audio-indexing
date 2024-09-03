import asyncio
from typing import Tuple
from basic_rag.basic_rag_system import BasicRAGSystem
from graph_rag.graph_rag_system import create_graph_rag_system, GraphRAGSystem


class DualRAGSystem:
    """
    A system that combines both Basic RAG and Graph
        RAG for enhanced query processing.

    This class initializes and manages two RAG
        (Retrieval-Augmented Generation) systems:
    1. A Basic RAG system for standard retrieval and generation.
    2. A Graph RAG system for graph-based retrieval and generation.

    Both systems are queried concurrently to provide comprehensive results.
    """

    def __init__(self, transcription_with_char_timestamps: str,
                 index_name: str,
                 neo4j_username: str,
                 neo4j_password: str,
                 neo4j_url: str) -> None:
        """
        Initialize the DualRAGSystem with both Basic and Graph RAG systems.

        Args:
            transcription_with_char_timestamps (str): The input transcription
                with character timestamps.
            index_name (str): The name of the index to be used in
                the Basic RAG system.
            neo4j_username (str): Username for Neo4j database connection.
            neo4j_password (str): Password for Neo4j database connection.
            neo4j_url (str): URL for Neo4j database connection.
        """
        self.basic_rag: BasicRAGSystem = BasicRAGSystem(
            transcription_with_char_timestamps,
            index_name
        )
        self.graph_rag: GraphRAGSystem = create_graph_rag_system(
            transcription_with_char_timestamps,
            neo4j_username,
            neo4j_password,
            neo4j_url
        )

    async def prepare_basic_rag(self) -> None:
        """
        Asynchronously prepare the Basic RAG system by processing its data.
        """
        await self.basic_rag.prepare_data()

    async def query_systems(self, query: str) -> Tuple[str, str]:
        """
        Query both RAG systems concurrently and return their responses.

        Args:
            query (str): The query string to be processed by both RAG systems.

        Returns:
            Tuple[str, str]: A tuple containing the responses from the
                Basic RAG and Graph RAG systems, respectively.
        """
        async def basic_rag_query() -> str:
            return await self.basic_rag.query(query)

        async def graph_rag_query() -> str:
            return self.graph_rag.query(query)

        basic_response, graph_response = await asyncio.gather(
            basic_rag_query(),
            graph_rag_query()
        )

        return basic_response, graph_response

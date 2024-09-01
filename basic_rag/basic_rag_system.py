from typing import List, Optional
import data_pull_and_prep.data_preparation as data_prep
import basic_rag.rag as rag
import basic_rag.response_synthesizer as response_synthesizer


class BasicRAGSystem:
    """
    A basic Retrieval-Augmented Generation (RAG) system for processing and
    querying transcriptions.

    This class encapsulates the functionality to prepare data from a
    transcription, create a vector store, and perform queries using a custom
    retriever and response synthesizer.

    Attributes:
        transcription_with_char_timestamps (str): The input transcription with
            character timestamps.
        index_name (str): The name of the index to be used in the vector store.
        custom_ingestion_obj (Optional[rag.CustomRAG]): Object for custom RAG
            ingestion.
        custom_retriever_obj (Optional[rag.CustomRetriever]): Object for custom
            retrieval.
        response_synthesizer_obj
        (Optional[response_synthesizer.aHierarchicalSummarizer]):
            Object for response synthesis.
    """

    def __init__(
        self, transcription_with_char_timestamps: str, index_name: str
    ):
        """
        Initialize the BasicRAGSystem.

        Args:
            transcription_with_char_timestamps (str): The input transcription
                with character timestamps.
            index_name (str): The name of the index to be used in the vector
                store.
        """
        self.transcription_with_char_timestamps: str = (
            transcription_with_char_timestamps
        )
        self.index_name: str = index_name
        self.custom_ingestion_obj: Optional[rag.CustomRAG] = None
        self.custom_retriever_obj: Optional[rag.CustomRetriever] = None
        self.response_synthesizer_obj: Optional[
            response_synthesizer.aHierarchicalSummarizer
        ] = None

    async def prepare_data(self) -> None:
        """
        Prepare the data for the RAG system.

        This method creates custom text chunks,
            initializes the CustomRAG object,
        creates text nodes and adds them to the vector store, initializes the
        CustomRetriever, and sets up the response synthesizer.
        """
        custom_chunking_obj: data_prep.CreateCustomTextChunks = (
            data_prep.CreateCustomTextChunks(
                self.transcription_with_char_timestamps
            )
        )
        text_chunks_with_timestamps: List[dict] = (
            custom_chunking_obj.create_custom_text_chunks()
        )

        self.custom_ingestion_obj = rag.CustomRAG(
            index_name=self.index_name,
            text_chunks_with_timestamps=text_chunks_with_timestamps,
        )
        await self.custom_ingestion_obj.create_text_nodes_and_add_to_vector_store()

        self.custom_retriever_obj = rag.CustomRetriever(
            embed_model=self.custom_ingestion_obj.embed_model,
            vector_store=self.custom_ingestion_obj.vector_store,
        )

        self.response_synthesizer_obj = (
            response_synthesizer.aHierarchicalSummarizer(
                llm=self.custom_ingestion_obj.llm
            )
        )

    async def query(self, query_str: str, num_children: int = 5) -> str:
        """
        Perform a query on the prepared RAG system.

        Args:
            query_str (str): The query string to process.
            num_children (int, optional): The number of child nodes to consider
                in the response synthesis. Defaults to 5.

        Returns:
            str: The synthesized response to the query.

        Raises:
            ValueError: If the system is not prepared (prepare_data() not
                called).
        """
        if not self.custom_retriever_obj or not self.response_synthesizer_obj:
            raise ValueError("System not prepared. Call prepare_data() first.")

        query_result = self.custom_retriever_obj.retrieve(query=query_str)
        return await self._run_aggregate_answers(
            self.response_synthesizer_obj,
            query_result.nodes,
            query_str,
            num_children,
        )

    @staticmethod
    async def _run_aggregate_answers(
        summarizer: response_synthesizer.aHierarchicalSummarizer,
        retrieved_nodes: List[dict],
        query_str: str,
        num_children: int = 10,
    ) -> str:
        """
        Run the aggregation of answers using the hierarchical summarizer.

        Args:
            summarizer (response_synthesizer.aHierarchicalSummarizer): The
                summarizer object.
            retrieved_nodes (List[dict]): The list of retrieved document nodes.
            query_str (str): The original query string.
            num_children (int, optional): Number of child nodes to consider.
                Defaults to 10.

        Returns:
            str: The aggregated and synthesized response.
        """
        return await summarizer.generate_response_hs(
            retrieved_nodes=retrieved_nodes,
            query_str=query_str,
            num_children=num_children,
        )


# Usage example:
# rag_system = BasicRAGSystem(transcription_with_char_timestamps,
#       "ivanka-08-28-via-class")
# await rag_system.prepare_data()
# response = await rag_system.query("Your query here")

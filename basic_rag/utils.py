import basic_rag.response_synthesizer as response_synthesizer
import basic_rag.rag as rag


class RetrieveAndAnswer:
    def __init__(self, ingestion_obj: rag.CustomRAG) -> None:
        self.ingestion_obj = ingestion_obj

    def answer(self, query_str: str) -> str:
        custom_retriever_obj = rag.CustomRetriever(
            embed_model=self.ingestion_obj.embed_model,
            vector_store=self.ingestion_obj.vector_store,
        )
        query_result = custom_retriever_obj.retrieve(query=query_str)

        response_synthesizer_obj = response_synthesizer.HierarchicalSummarizer(
            llm=self.ingestion_obj.llm
        )
        response = response_synthesizer_obj.generate_response_hs(
            retrieved_nodes=query_result.nodes,  # type: ignore
            query_str=query_str
        )
        return response

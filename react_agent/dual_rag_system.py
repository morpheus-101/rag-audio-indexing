import asyncio
from basic_rag.basic_rag_system import BasicRAGSystem
from graph_rag.graph_rag_system import create_graph_rag_system

class DualRAGSystem:
    def __init__(self, transcription_with_char_timestamps: str, neo4j_username: str, neo4j_password: str, neo4j_url: str):
        self.basic_rag = BasicRAGSystem(transcription_with_char_timestamps, "ivanka-08-31-via-class")
        self.graph_rag = create_graph_rag_system(
            transcription_with_char_timestamps,
            neo4j_username,
            neo4j_password,
            neo4j_url
        )

    async def prepare_basic_rag(self):
        await self.basic_rag.prepare_data()

    async def query_systems(self, query: str) -> tuple[str, str]:
        async def basic_rag_query():
            return await self.basic_rag.query(query)

        async def graph_rag_query():
            return self.graph_rag.query(query)

        basic_response, graph_response = await asyncio.gather(
            basic_rag_query(),
            graph_rag_query()
        )

        return basic_response, graph_response

async def main():
    neo4j_username = "neo4j"
    neo4j_password = "neo4j_rishi"
    neo4j_url = "bolt://localhost:7687"  # Adjust this URL as needed

    dual_rag = DualRAGSystem(transcription_with_char_timestamps, neo4j_username, neo4j_password, neo4j_url)
    await dual_rag.prepare_basic_rag()

    query = "What are some famous quotes mentioned in this podcast and who said them?"
    basic_response, graph_response = await dual_rag.query_systems(query)

    print("Basic RAG response:", basic_response)
    print("Graph RAG response:", graph_response)

if __name__ == "__main__":
    asyncio.run(main())
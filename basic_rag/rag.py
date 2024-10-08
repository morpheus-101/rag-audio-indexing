from dataclasses import dataclass
from typing import Optional
from typing import List, Tuple
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class APIKeys:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class CreatePineconeIndex:
    index_name: str = ""
    index_dimension: int = 1536
    index_metric: str = "euclidean"
    index_cloud: str = "aws"
    index_region: str = "us-east-1"


@dataclass
class OpenAIModel:
    model_name: str = "gpt-3.5-turbo"


@dataclass
class Extractors_config:
    no_of_questions: int = 5
    no_keywords_to_extract: int = 10
    summary_type: str = "self"


class CustomRAG:
    def __init__(
        self,
        index_name: str,
        text_chunks_with_timestamps: List[Tuple[str, Tuple[float, float]]],
        create_pinecone_index: Optional[CreatePineconeIndex] = None,
        openai_model: Optional[OpenAIModel] = None,
        extractors: Optional[Extractors_config] = None,
    ):
        self.api_keys = APIKeys()
        self.create_pinecone_index = (
            create_pinecone_index
            if create_pinecone_index is not None
            else CreatePineconeIndex()
        )
        self.create_pinecone_index.index_name = index_name
        self.openai_model = (
            openai_model if openai_model is not None else OpenAIModel()
        )
        self.extractors = (
            extractors if extractors is not None else Extractors_config()
        )
        self.text_chunks_with_timestamps = text_chunks_with_timestamps
        self.llm = OpenAI(
            model=self.openai_model.model_name,
            api_key=self.api_keys.openai_api_key,
        )

    def _setup_pinecone_vector_store(self) -> PineconeVectorStore:
        pc = Pinecone(api_key=self.api_keys.pinecone_api_key)
        if self.create_pinecone_index.index_name not in (
            pc.list_indexes().names()
        ):
            pc.create_index(
                name=self.create_pinecone_index.index_name,
                dimension=self.create_pinecone_index.index_dimension,
                metric=self.create_pinecone_index.index_metric,
                spec=ServerlessSpec(
                    cloud=self.create_pinecone_index.index_cloud,
                    region=self.create_pinecone_index.index_region,
                ),
            )

        pinecone_index = pc.Index(self.create_pinecone_index.index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        return vector_store

    def _create_text_nodes(self):
        nodes = []
        for i in range(len(self.text_chunks_with_timestamps)):
            node = TextNode(text=self.text_chunks_with_timestamps[i][0])
            nodes.append(node)
        return nodes

    def _generate_embeddings(self, nodes):
        self.embed_model = OpenAIEmbedding(
            api_key=self.api_keys.openai_api_key
        )
        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")  # type: ignore
            )
            node.embedding = node_embedding
        return nodes

    async def create_text_nodes_and_add_to_vector_store(self) -> None:
        self.vector_store = self._setup_pinecone_vector_store()
        nodes = self._create_text_nodes()
        list_of_extractors = [
            QuestionsAnsweredExtractor(
                questions=(self.extractors.no_of_questions), llm=self.llm
            ),
            SummaryExtractor(
                llm=self.llm, summaries=[self.extractors.summary_type]
            ),
            KeywordExtractor(
                llm=self.llm, keywords=self.extractors.no_keywords_to_extract
            ),
        ]
        pipeline = IngestionPipeline(
            transformations=list_of_extractors  # type: ignore
        )
        nodes = await pipeline.arun(nodes=nodes, in_place=False)
        nodes = self._generate_embeddings(nodes)

        for i, node in enumerate(nodes):
            node.extra_info["start_timestamp"] = float(
                self.text_chunks_with_timestamps[i][1][0]
            )
            node.extra_info["end_timestamp"] = float(
                self.text_chunks_with_timestamps[i][1][1]
            )
        self.all_nodes = nodes
        self.vector_store.add(nodes)  # type: ignore


class CustomRetriever:
    def __init__(
        self, embed_model: OpenAIEmbedding, vector_store: PineconeVectorStore
    ):
        self.embed_model = embed_model  # type: ignore
        self.vector_store = vector_store

    def retrieve(self, query: str) -> VectorStoreQueryResult:
        query_embedding = self.embed_model.get_query_embedding(query)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=3
        )
        query_result = self.vector_store.query(vector_store_query)
        return query_result

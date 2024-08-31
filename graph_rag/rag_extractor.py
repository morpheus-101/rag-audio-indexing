from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.async_utils import run_jobs
from typing import Any, List, Callable, Union
import asyncio
import nest_asyncio
import warnings
from graph_rag.prompts import KG_TRIPLET_EXTRACT_TMPL
from graph_rag.utils import default_parser
from dataclasses import dataclass

warnings.filterwarnings("ignore")

nest_asyncio.apply()


# @dataclass
# class GraphRAGConfig:
#     default_llm: LLM = OpenAI(model="gpt-3.5-turbo")


class GraphRAGExtractor(TransformComponent):
    """Extract entities and relationships from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths 
        (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        # llm: LLM = GraphRAGConfig().default_llm,
        llm: LLM = OpenAI(model="gpt-3.5-turbo"),
        extract_prompt: Union[
            str, PromptTemplate
        ] = KG_TRIPLET_EXTRACT_TMPL,
        parse_fn: Callable = default_parser,
        max_paths_per_chunk: int = 10,
        num_workers: int = 10,
    ) -> None:

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity_dict in entities:
            entity_metadata["entity_description"] = entity_dict[
                "entity_description"
            ]
            entity_node = EntityNode(
                name=entity_dict["entity_name"],
                label=entity_dict["entity_type"],
                properties=entity_metadata,
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for relationship_dict in entities_relationship:
            subj = relationship_dict["source_entity"]
            obj = relationship_dict["target_entity"]
            rel = relationship_dict["relation"]
            description = relationship_dict["relationship_description"]

            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.llms import LLM
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage, MessageRole
import re
from llama_index.core.prompts import PromptTemplate
import asyncio
import nest_asyncio  # type: ignore
import warnings
from graph_rag.rag_store import GraphRAGStore
from typing import List, Dict, Any, Tuple, Set

warnings.filterwarnings("ignore")

nest_asyncio.apply()


class GraphRAGQueryEngine:
    """
    A query engine for graph-based Retrieval-Augmented Generation (RAG).

    This class handles the process of querying a graph-based knowledge store,
    retrieving relevant information, and generating answers based on the query.

    Attributes:
        graph_store (GraphRAGStore): The graph-based knowledge store.
        index (PropertyGraphIndex): The index for efficient retrieval.
        llm (LLM): The language model used for generating answers.
        similarity_top_k (int): The number of top similar items to retrieve.
        num_children (int): The number of child nodes to process in parallel.
        qa_prompt (PromptTemplate): The template for
            question-answering prompts.
    """

    def __init__(
        self,
        graph_store: GraphRAGStore,
        index: PropertyGraphIndex,
        llm: LLM,
        similarity_top_k: int = 20,
        num_children: int = 3,
    ) -> None:
        """
        Initialize the GraphRAGQueryEngine.

        Args:
            graph_store (GraphRAGStore): The graph-based knowledge store.
            index (PropertyGraphIndex): The index for efficient retrieval.
            llm (LLM): The language model used for generating answers.
            similarity_top_k (int): The number of top similar
                items to retrieve.
            num_children (int): The number of child nodes to
                process in parallel.
        """
        self.graph_store: GraphRAGStore = graph_store
        self.index: PropertyGraphIndex = index
        self.llm: LLM = llm
        self.similarity_top_k: int = similarity_top_k
        self.num_children: int = num_children
        self.qa_prompt: PromptTemplate = PromptTemplate(
            """\
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query_str}
        Answer: \
        """
        )

    def custom_query(self, query_str: str) -> str:
        """
        Process all community summaries to generate answers to
            a specific query.

        Args:
            query_str (str): The query string to process.

        Returns:
            str: The final answer generated from the community summaries.
        """
        entities: List[str] = self.get_entities(query_str,
                                                self.similarity_top_k)
        self.entities: List[str] = entities
        community_ids: List[int] = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        self.community_ids: List[int] = community_ids
        community_summaries: Dict[int, str] = (
            self.graph_store.get_community_summaries()
        )
        self.community_summaries: Dict[int, str] = community_summaries
        community_answers: List[str] = self.parallel_generate_answers(
            community_summaries, community_ids, query_str
        )
        self.community_answers: List[str] = community_answers
        final_answer: str = self.aggregate_answers(
            texts=community_answers,
            query_str=query_str,
            qa_prompt=self.qa_prompt,
            llm=self.llm,
            num_children=self.num_children,
        )
        return final_answer

    def get_entities(self, query_str: str, similarity_top_k: int) -> List[str]:
        """
        Retrieve entities from the query string using the index.

        Args:
            query_str (str): The query string to process.
            similarity_top_k (int): The number of top similar
                items to retrieve.

        Returns:
            List[str]: A list of unique entities extracted from the query.
        """
        nodes_retrieved: List[Any] = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)
        entities: Set[str] = set()
        pattern: str = (
            r"(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)"  # type: ignore
        )

        for node in nodes_retrieved:
            matches: List[Tuple[str, str, str]] = re.findall(
                pattern, node.text, re.DOTALL
            )

            for match in matches:
                subject: str = match[0]
                obj: str = match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)

    def retrieve_entity_communities(
        self, entity_info: Dict[str, List[int]], entities: List[str]
    ) -> List[int]:
        """
        Retrieve cluster information for given entities, allowing for multiple
        clusters per entity.

        Args:
            entity_info (Dict[str, List[int]]): Dictionary mapping entities to
                their cluster IDs (list).
            entities (List[str]): List of entity names to
                retrieve information for.

        Returns:
            List[int]: List of community or cluster IDs to
                which an entity belongs.
        """
        community_ids: List[int] = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(
        self, community_summary: str, query: str
    ) -> str:
        """
        Generate an answer from a community summary based on a
            given query using LLM.

        Args:
            community_summary (str): The community summary to use
                for generating the answer.
            query (str): The query to answer.

        Returns:
            str: The generated answer based on the community summary and query.
        """
        prompt: str = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(
                role=MessageRole.USER,
                content="I need an answer based on the above information.",
            ),
        ]
        response: Any = self.llm.chat(messages)
        cleaned_response: str = re.sub(
            r"^assistant:\s*", "", str(response)
        ).strip()
        return cleaned_response

    def parallel_generate_answers(
        self,
        community_summaries: Dict[int, str],
        community_ids: List[int],
        query_str: str,
    ) -> List[str]:
        """
        Parallelize the generation of answers from community summaries using
        ThreadPoolExecutor.

        Args:
            community_summaries (Dict[int, str]): Dictionary of
                community summaries.
            community_ids (List[int]): List of community IDs to process.
            query_str (str): The query string to answer.

        Returns:
            List[str]: A list of generated answers from the
                community summaries.
        """

        def generate_answer(id: int,
                            community_summary: str) -> Tuple[int, str]:
            """Generate an answer from community summary based
                on a given query."""
            return id, self.generate_answer_from_summary(
                community_summary, query_str
            )

        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            future_to_id: Dict[Any, int] = {
                executor.submit(generate_answer, id, summary): id
                for id, summary in community_summaries.items()
                if id in community_ids
            }

            community_answers: List[str] = []
            # Process results as they complete
            for future in as_completed(future_to_id):
                try:
                    _, answer = future.result()
                    community_answers.append(answer)
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
        return community_answers

    async def acombine_results(
        self,
        texts: List[str],
        query_str: str,
        qa_prompt: PromptTemplate,
        llm: LLM,
        num_children: int = 3,
    ) -> str:
        """
        Asynchronously combine results from multiple texts.

        Args:
            texts (List[str]): List of text results to combine.
            query_str (str): The original query string.
            qa_prompt (PromptTemplate): The question-answering prompt template.
            llm (LLM): The language model to use for combining results.
            num_children (int): The number of child texts to
            process in parallel.

        Returns:
            str: The combined result.
        """
        fmt_prompts: List[str] = []
        for idx in range(0, len(texts), num_children):
            text_batch: List[str] = texts[idx: idx + num_children]
            context_str: str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt: str = qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
        tasks: List[Any] = [llm.acomplete(p) for p in fmt_prompts]
        combined_responses: List[Any] = await asyncio.gather(*tasks)
        new_texts: List[str] = [str(r) for r in combined_responses]

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return await self.acombine_results(
                new_texts, query_str, qa_prompt, llm, num_children=num_children
            )

    def aggregate_answers(
        self,
        texts: List[str],
        query_str: str,
        qa_prompt: PromptTemplate,
        llm: LLM,
        num_children: int = 3,
    ) -> str:
        """
        Aggregate answers from multiple texts.

        Args:
            texts (List[str]): List of text answers to aggregate.
            query_str (str): The original query string.
            qa_prompt (PromptTemplate): The question-answering prompt template.
            llm (LLM): The language model to use for aggregating answers.
            num_children (int): The number of child texts to
                process in parallel.

        Returns:
            str: The aggregated answer.
        """
        return asyncio.run(
            self.acombine_results(
                texts=texts,
                query_str=query_str,
                qa_prompt=qa_prompt,
                llm=llm,
                num_children=num_children,
            )
        )

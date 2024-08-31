from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.llms import LLM
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage, MessageRole
import re
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
import asyncio
import nest_asyncio  # type: ignore
import warnings
from dataclasses import dataclass
from graph_rag.rag_store import GraphRAGStore

warnings.filterwarnings("ignore")

nest_asyncio.apply()


# @dataclass
# class GraphRAGConfig:
#     default_llm: LLM = OpenAI(model="gpt-3.5-turbo")


class GraphRAGQueryEngine:
    def __init__(
        self,
        graph_store: GraphRAGStore,
        index: PropertyGraphIndex,
        llm: LLM,
        similarity_top_k: int = 20,
        num_children: int = 3,
    ):
        self.graph_store = graph_store
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.num_children = num_children
        self.qa_prompt = PromptTemplate(
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
        """Process all community summaries to generate 
            answers to a specific query."""
        entities = self.get_entities(query_str, self.similarity_top_k)
        self.entities = entities
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        self.community_ids = community_ids
        community_summaries = self.graph_store.get_community_summaries()
        self.community_summaries = community_summaries
        community_answers = []
        community_answers = self.parallel_generate_answers(
            community_summaries, community_ids, query_str
        )
        self.community_answers = community_answers
        final_answer = self.aggregate_answers(
            texts=community_answers,
            query_str=query_str,
            qa_prompt=self.qa_prompt,
            llm=self.llm,
            num_children=self.num_children,
        )
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)
        enitites = set()
        pattern = r"(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.DOTALL)

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            # f"Given the community summary: {community_summary}, "
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(
                role=MessageRole.USER,
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def parallel_generate_answers(
        self, community_summaries, community_ids, query_str
    ):
        """Parallelize the generation of answers from community
            summaries using ThreadPoolExecutor."""

        def generate_answer(id, community_summary):
            """Generate an answer from community summary
                based on a given query."""
            return id, self.generate_answer_from_summary(
                community_summary, query_str
            )

        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            future_to_id = {
                executor.submit(generate_answer, id, summary): id
                for id, summary in community_summaries.items()
                if id in community_ids
            }

            community_answers = []
            # Process results as they complete
            for future in as_completed(future_to_id):
                # id = future_to_id[future]
                try:
                    _, answer = future.result()
                    community_answers.append(answer)
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
        return community_answers

    async def acombine_results(
        self,
        texts,
        query_str,
        qa_prompt,
        llm,
        num_children=3,
    ):
        fmt_prompts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx: idx + num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
        tasks = [llm.acomplete(p) for p in fmt_prompts]
        combined_responses = await asyncio.gather(*tasks)
        new_texts = [str(r) for r in combined_responses]

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return await self.acombine_results(
                new_texts, query_str, qa_prompt, llm, num_children=num_children
            )

    def aggregate_answers(
        self, texts, query_str, qa_prompt, llm, num_children=3
    ):
        return asyncio.run(
            self.acombine_results(
                texts=texts,
                query_str=query_str,
                qa_prompt=qa_prompt,
                llm=llm,
                num_children=num_children,
            )
        )

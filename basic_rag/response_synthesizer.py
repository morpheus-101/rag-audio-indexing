from basic_rag.prompts import QUESTION_ANSWERING_PROMPT_BASIC
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from typing import Optional
from typing import List
from llama_index.core.schema import TextNode
import asyncio


class HierarchicalSummarizer:
    def __init__(
            self,
            llm: OpenAI,
            qa_prompt: Optional[PromptTemplate] = None):
        self.llm = llm
        self.qa_prompt = (qa_prompt if qa_prompt is not None
                          else QUESTION_ANSWERING_PROMPT_BASIC)

    def combine_results(
            self,
            texts: List[str],
            query_str: str,
            num_children=10) -> str:
        new_texts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx: idx + num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            combined_response = self.llm.complete(fmt_qa_prompt)
            new_texts.append(str(combined_response))

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return self.combine_results(
                new_texts, query_str, num_children=num_children)

    def generate_response_hs(
            self,
            retrieved_nodes: List[TextNode],
            query_str: str,
            num_children: int = 10) -> str:
        node_responses = []
        for node in retrieved_nodes:
            context_str = node.get_content()
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            node_response = self.llm.complete(fmt_qa_prompt)
            node_responses.append(node_response)
        response_txt = self.combine_results(
            [str(r) for r in node_responses], query_str,
            num_children=num_children
        )
        return response_txt


class aHierarchicalSummarizer:
    def __init__(self, llm: OpenAI,
                 qa_prompt: Optional[PromptTemplate] = None):
        self.llm = llm
        self.qa_prompt = (qa_prompt if qa_prompt is not None else
                          QUESTION_ANSWERING_PROMPT_BASIC)

    async def acombine_results(
        self,
        texts,
        query_str,
        # cur_prompt_list,
        num_children=3,
    ) -> str:
        fmt_prompts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx: idx + num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
            # cur_prompt_list.append(fmt_qa_prompt)

        tasks = [self.llm.acomplete(p) for p in fmt_prompts]
        combined_responses = await asyncio.gather(*tasks)
        new_texts = [str(r) for r in combined_responses]

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return await self.acombine_results(
                new_texts, query_str, num_children=num_children
            )

    async def generate_response_hs(self, retrieved_nodes: List[TextNode],
                                   query_str: str,
                                   num_children: int = 10) -> str:
        node_responses = []
        fmt_prompts = []
        for node in retrieved_nodes:
            context_str = node.get_content()
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
        tasks = [self.llm.acomplete(p) for p in fmt_prompts]
        node_responses = await asyncio.gather(*tasks)
        response_txt = await self.acombine_results(
            texts=[str(r) for r in node_responses],
            query_str=query_str,
            num_children=num_children)
        return response_txt

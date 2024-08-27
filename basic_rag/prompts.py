from llama_index.core import PromptTemplate

QUESTION_ANSWERING_PROMPT_BASIC = PromptTemplate(
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

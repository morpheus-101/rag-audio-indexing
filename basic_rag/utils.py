import basic_rag.response_synthesizer as response_synthesizer
import basic_rag.rag as rag
import random
import textwrap
from typing import List, Tuple


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


class RandomQuestionAndTopics:
    def __init__(self, ingestion_obj: rag.CustomRAG) -> None:
        self.ingestion_obj = ingestion_obj

    def _get_random_items_with_indices(
        self, input_list: List[str], n: int = 5
    ) -> List[Tuple[int, str]]:
        n = min(n, len(input_list))
        indexed_list = list(enumerate(input_list))
        random_samples = random.sample(indexed_list, n)
        return random_samples

    def _get_random_question(self, question_string: str) -> str:
        questions = question_string.split("\n")
        random_question = random.choice(questions)
        cleaned_question = random_question.split(".", 1)[-1].strip()
        return cleaned_question

    def _remove_duplicates(self, string_list: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in string_list:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def print_questions_and_topics(
        self, data_tuple: Tuple[List[str], List[str]]
    ) -> None:
        """
        Print questions and topics separately in a
            notebook format, wrapping long strings.

        Args:
        data_tuple (tuple):
            A tuple containing two lists - questions and topics.
        """
        questions, topics = data_tuple

        def print_section(title, items):
            box_width = 79
            content_width = box_width - 5

            print(f"╔{'═' * (box_width - 2)}╗")
            print(f"║ {title:<{box_width - 4}} ║")
            print(f"╠{'═' * (box_width - 2)}╣")

            for i, item in enumerate(items, 1):
                # Wrap the text
                wrapped_lines = textwrap.wrap(item, width=content_width)

                # Print the first line with the item number
                print(f"║ {i}. {wrapped_lines[0]:<{content_width}} ║")

                # Print any additional lines
                for line in wrapped_lines[1:]:
                    print(f"║    {line:<{content_width}} ║")

                if i < len(items):
                    print(f"║{' ' * (box_width - 2)}║")

            print(f"╚{'═' * (box_width - 2)}╝")
            print()  # Add a blank line between sections

        print_section("Questions", questions)
        print_section("Topics", topics)

    def get_random_questions_and_titles(self) -> Tuple[List[str], List[str]]:
        all_node_ids = []
        for node in self.ingestion_obj.all_nodes:
            all_node_ids.append(node.id_)

        random_items = self._get_random_items_with_indices(all_node_ids, 3)
        random_doc_titles = []
        random_questions = []
        for index, item in random_items:
            random_doc_titles.append(
                self.ingestion_obj.all_nodes[index].metadata["document_title"]
            )
            random_questions.append(
                self._get_random_question(
                    self.ingestion_obj.all_nodes[index].metadata[
                        "questions_this_excerpt_can_answer"
                    ]
                )
            )
        return (self._remove_duplicates(random_questions),
                self._remove_duplicates(random_doc_titles))

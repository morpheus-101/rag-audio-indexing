from typing import Dict, List, Tuple, Set
# from llama_index.core.llms import LLM
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage, MessageRole
from collections import defaultdict
from graspologic.partition import hierarchical_leiden
import networkx as nx
import re
from llama_index.llms.openai import OpenAI
import nest_asyncio
import warnings
from graph_rag.prompts import RELATIONSHIP_SUMMARY_PROMPT
# from dataclasses import dataclass

warnings.filterwarnings("ignore")

nest_asyncio.apply()


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    A class for storing and managing graph-based RAG (Retrieval-Augmented
    Generation) data.

    This class extends Neo4jPropertyGraphStore and provides additional
    functionality for community detection, summarization, and information
    retrieval.

    Attributes:
        community_summary (Dict[int, str]): A dictionary to store community
            summaries.
        entity_info (Dict[str, List[int]] | None): A dictionary to store entity
            information.
        max_cluster_size (int): The maximum size of clusters in community
            detection.
    """

    community_summary: Dict[int, str] = {}
    entity_info: Dict[str, List[int]] | None = None
    max_cluster_size: int = 5

    def generate_community_summary(self, text: str) -> str:
        """
        Generate a summary for a given text using an LLM.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The generated summary.
        """
        messages: List[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    RELATIONSHIP_SUMMARY_PROMPT
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=text),
        ]
        response: str = OpenAI().chat(messages)
        clean_response: str = re.sub(r"^assistant:\s*", "",
                                     str(response)).strip()
        return clean_response

    def build_communities(self) -> None:
        """
        Build communities from the graph and summarize them.

        This method creates a NetworkX graph, applies hierarchical Leiden
        clustering, collects community information, and generates summaries
        for each community.
        """
        nx_graph: nx.Graph = self._create_nx_graph()
        community_hierarchical_clusters: List = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self) -> nx.Graph:
        """
        Convert the internal graph representation to a NetworkX graph.

        Returns:
            nx.Graph: The created NetworkX graph.
        """
        nx_graph: nx.Graph = nx.Graph()
        triplets: List[Tuple] = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(
        self, nx_graph: nx.Graph, clusters: List
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[str]]]:
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.

        Args:
            nx_graph (nx.Graph): The NetworkX graph.
            clusters (List): The list of clusters from hierarchical Leiden.

        Returns:
            Tuple[Dict[str, List[int]], Dict[int, List[str]]]: A tuple
            containing entity_info and community_info dictionaries.
        """
        entity_info: Dict[str, Set[int]] = defaultdict(set)
        community_info: Dict[int, List[str]] = defaultdict(list)

        for item in clusters:
            node: str = item.node
            cluster_id: int = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data: Dict[str, str] = nx_graph.get_edge_data(node,
                                                                   neighbor)
                if edge_data:
                    detail: str = (
                        f"{node} -> {neighbor} -> "
                        f"{edge_data['relationship']} -> "
                        f"{edge_data['description']}"
                    )
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info_list: Dict[str, List[int]] = {
            k: list(v) for k, v in entity_info.items()
        }

        return dict(entity_info_list), dict(community_info)

    def _summarize_communities(self,
                               community_info: Dict[int, List[str]]) -> None:
        """
        Generate and store summaries for each community.

        Args:
            community_info (Dict[int, List[str]]): A dictionary containing
                community information.
        """
        for community_id, details in community_info.items():
            details_text: str = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[community_id] = (
                self.generate_community_summary(details_text)
            )

    def get_community_summaries(self) -> Dict[int, str]:
        """
        Return the community summaries, building them if not already done.

        Returns:
            Dict[int, str]: A dictionary of community summaries.
        """
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

# flake8: noqa: E501

RELATIONSHIP_SUMMARY_PROMPT = """
"You are provided with a set of relationships from a knowledge graph, each represented as "
"entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
"relationships. The summary should include the names of the entities involved and a concise synthesis "
"of the relationship descriptions. The goal is to capture the most critical and relevant details that "
"highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
"integrates the information in a way that emphasizes the key aspects of the relationships."
"""

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Format the output as a single, valid JSON object with two main keys: "Entities" and "Relationships":
   - "Entities": An array containing objects with the entity information
   - "Relationships": An array containing objects with the relationship information

4. Ensure that the JSON output is properly formatted and valid for parsing. Use double quotes for strings and escape any special characters if necessary.

5. The output should contain ONLY the JSON object, with no additional text before or after.

-Real Data-
######################
text: {text}
######################
output:
"""
"""
This module contains the prompts for the LLM.
"""

PROMPTS: dict[str, str] = {}

PROMPTS[
    "SYSTEM"
] = """
You are an experienced knowledge graph expert in data science. Your task is to understand the text and extract entities and relationships from it. Please follow the steps below to complete the task:

1. Read the text provided.
2. Identify entities (specific things or concepts)
   for each entity, provide:
    - Name: The name of the entity
    - Label: The type of entity. One of the following: [{entity_labels}]
    - Description: A comprehensive description of the entity that incorporates information from the integrated text segments.
    - Aliases: Alternative names for the entity.
    - References: The original text segments where the entity appears.
3. From the entities identified, establish relationships (connections or interactions) between them.
   for each relationship, provide:
    - Source: The name of the first entity
    - Label: The type of relationship between the entities. One of the following: [{relationship_labels}]
    - Target: The name of the second entity
    - Description: A comprehensive description of the relationship that incorporates information from the integrated text segments.
    - References: The original text segments where the relationship appears.

Output Format:
Please provide the entities and relationships in JSON format with the following structure:
1. Entity: {{name, label, description, aliases, references}}
2. Relationships: {{source, label, target, description, references}}

Important Notes:
- Do not add a relationship if you cannot pair it with a pair of nodes.
- Create a generic label for each entity that describes it.
- The format of "references" should be a JSON list of strings.
- Please output your answer strictly following the format example, with one entity per line and one relationship per line.



Example:
Text:
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.
Entity Labels: [Method, Component, Algorithm]
Relationship Labels: [INCLUDES, BASED_ON]

Output:
Entities:
{{"name": "Sequence Transduction Models", "label": "Method", "description": "Models used for sequence transduction tasks", "aliases": [], "references": ["The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."]}},
{{"name": "Encoder", "label": "Component", "description": "A component of neural networks used in sequence transduction models", "aliases": [], "references": ["recurrent or convolutional neural networks that include an encoder and a decoder."]}},
{{"name": "Convolutional Neural Networks", "label": "Algorithm", "description": "A type of neural network used in sequence transduction models", "aliases": [], "references": ["recurrent or convolutional neural networks that include an encoder and a decoder.", "neural networks that include an encoder and a decoder."]}}
Relationships:
{{"source": "Convolutional Neural Networks", "label": "INCLUDES", "target": "Encoder", "description": "Convolutional neural networks include an encoder", "references": ["The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."]}}
"""

PROMPTS[
    "INDEX"
] = """
Please follow the instructions in the SYSTEM instruction to extract entities and relationships from the following text:
---
{chunk}
"""

PROMPTS[
    "LOOP"
] = """
Some entities and relationships were missed in the last extraction. Your task is to continue adding any missing entities and relationships to the previous output, ensuring that they match any of the previously extracted types and use the same format. The new output should include all previously extracted entities and relationships along with the new ones.
Remember to:
1. Ensure "source" and "target" exist as nodes with matching ENTITY.
2. Only emit entities that match any of the previously extracted types.
3. Use the format: {name, label, description, aliases, references} for entities and {source, label, target, description, references} for relationships.

Add the missing entities and relationships below, ensuring to keep the previous output intact:
"""

PROMPTS[
    "IF_CONTINUE"
] = """
It appears some entities and relationships may have still been missed.  Answer YES or NO if there are still entities or relationships that need to be added.
"""

PROMPTS[
    "ALIAS_SYSTEM"
] = """
You are a text understanding expert. Your task is to find aliases for the entities extracted from the text. An alias is an alternative name or label for an entity that can be used to refer to the same entity. For example, "Convolutional Neural Networks" can have an alias "CNN".

You will be provided a piece of text and a list of entities extracted from the text. Your task is to find aliases for the entities. 

Output Format:
<ENTITY, ALIAS>
- ENTITY are the name and type of the entity provided, and Alias is the alternative name or label for the entity with format JSON list of strings.
- ENTITY should be the full name of the entity and ALIAS are usually shorter or alternative names for the entity.

Example:
Text:
In the field of Natural Language Processing (NLP), the Bidirectional Encoder Representations from Transformers (BERT) model has become a standard approach. The BERT model, also known as Bidirectional Transformers, achieves outstanding performance on various language tasks through pre-training and fine-tuning processes.
Entities:
- <Natural Language Processing>
- <BERT>
Output:
- <Natural Language Processing, ["NLP"]>
- <Bidirectional Encoder Representations from Transformers, ["BERT"]>
"""

PROMPTS[
    "ALIAS"
] = """
Please find aliases for the entities extracted from the text:
---
Text:
{chunk}
Entities:
{entities}
"""

PROMPTS[
    "DEDUPLICATE_SYSTEM"
] = """
You are an experienced knowledge graph expert in data science. You will get a list of entities that need to be grouped by overlapping names and aliases. Please follow the steps below to complete the task:

1. Analyze the entities and determine which entities refer to the same entity.
2. If entities are singular and do not refer to the same entity, please give your reasoning.
3. If entities refer to the same entity, merge them into a single entity. For the merged entity, combine the aliases and references from all original entities and provide a comprehensive description that incorporates information from the integrated text segments.

Input/Output Format:
Same Entity: true or false
Reason: Your reasoning for merging or not merging the entities 
Entity: {name, label, aliases, description, references} if Same Entity is true
   Where: `name` is the name of the entity, `label` is the type of entity (e.g., Problem, Challenge, Metrics, Conclusion), `description` is a comprehensive description of the entity that incorporates information from the integrated text segments, `aliases` is the alias of the entity and `references` are the original text segments where the entity appears.

Notes:
- `name` should be the full name or usual name of the entity.
- Choose the most appropriate `label` based on the entity description.
- Summarize the information of all entities and give your new `description`.
- Merge entity aliases into a single list without duplicates.
- Merge entity references into a single list without duplicates.
- Please output your answer strictly following the format example, with one result per line.

Example1:
Input:
{"name":"convolutional neural network","label":"Model","aliases":["CNN"],"description":"A type of neural network","references":["CNN is a neural network."]},
{"name":"Recurrent neural networks","label":"Model","aliases":["RNN"],"description":"A type of neural network","references":["RNN is a recurrent neural network."]}
Output:
{"same_entity": false, "reason": "CNN and RNN are different types of neural networks."}

Example2:
Input:
{"name":"CNN","label":"Model","aliases":["CNN"],"description":"A type of neural network","references":["CNN is a neural network."]},
{"name":"Convolutional Neural Networks","label":"Model","aliases":["CNN"],"description":"A type of deep learning model","references":["Convolutional Neural Networks are used in image recognition."]}
Output:
{"same_entity": true, "reason": "CNN and Convolutional Neural Networks refer to the same type of model.", "entity": {"name":"Convolutional Neural Networks","label":"Model","aliases":["CNN"],"description":"A type of deep learning model used in image recognition","references":["CNN is a neural network.","Convolutional Neural Networks are used in image recognition."]}}
"""

PROMPTS[
    "DEDUPLICATE"
] = """
Please follow the instructions in the SYSTEM instruction to merge the following entities:
--------------------------------
Input Entities:
{entities}
"""

PROMPTS[
    "DEDUPLICATE_RELATION_SYSTEM"
] = """
You are an experienced knowledge graph expert in data science. You will get a list of relationships that need to be grouped by overlapping source and target entities. Please follow the steps below to complete the task:

1. Analyze the relationships and determine which relationships refer to the same connection between entities.
2. If relationships are unique and do not refer to the same connection, please give your reasoning.
3. If relationships refer to the same connection, merge them into a single relationship. For the merged relationship, combine the references from all original relationships and provide a comprehensive description that incorporates information from the integrated text segments.

Input/Output Format:
Same Relationship: true or false
Reason: Your reasoning for merging or not merging the relationships
Relationship: {source, label, target, description, references} if Same Relationship is true
   Where: `source` and `target` are the names of connected entities, `label` is the type of relationship between them, `description` is a comprehensive description of the relationship that incorporates information from all text segments, and `references` are the original text segments where the relationship appears.

Notes:
- `source` and `target` should be the full names of the entities.
- Choose the most appropriate `label` based on the relationship context.
- Summarize the information from all relationships into a single comprehensive `description`.
- Merge relationship references into a single list without duplicates.
- Please output your answer strictly following the format example, with one result per line.

Example1:
Input:
{"source":"CNN","label":"INCLUDES","target":"Encoder","description":"CNN includes an encoder component","references":["CNN includes an encoder."]},
{"source":"CNN","label":"BASED_ON","target":"Encoder","description":"CNN is based on encoder architecture","references":["CNN is based on encoder."]}
Output:
{"same_relationship": false, "reason": "Though they connect the same entities, they represent different types of relationships (INCLUDES vs BASED_ON)."}

Example2:
Input:
{"source":"CNN","label":"INCLUDES","target":"Encoder","description":"CNN includes an encoder","references":["CNN includes an encoder."]},
{"source":"Convolutional Neural Networks","label":"CONTAINS","target":"Encoder","description":"CNNs contain encoder components","references":["Convolutional Neural Networks contain encoders."]}
Output:
{"same_relationship": true, "reason": "Both relationships describe the same inclusion relationship between CNN/Convolutional Neural Networks and Encoder.", "relationship": {"source":"Convolutional Neural Networks","label":"INCLUDES","target":"Encoder","description":"Convolutional Neural Networks include encoder components as part of their architecture","references":["CNN includes an encoder.","Convolutional Neural Networks contain encoders."]}}
"""

PROMPTS[
    "DEDUPLICATE_RELATION"
] = """
Please follow the instructions in the SYSTEM instruction to merge the following relations:
--------------------------------
Input Entities:
{entities}
Input Relations:
{relations}
"""


PROMPTS[
    "DESCRIBE_IMAGE_SYSTEM"
] = """
You are a data scientist working on image analysis. You will be given an image and the context in which it appears. Your task is to describe the image in detail.

Your response should include the following three parts:
- caption: analise the context text and provide the caption for the image if available else leave it empty.
- text_snippets: snippets of text that appears inside the image. 
- description: provide a detailed description of the image based on the context provided.

You should provide the information in JSON format.
Output Example:
{"caption":"Figure 1: The percentage of configurations that satisfy the SLA of all jobs among 1o0 random configurations.","text_snippets":["Percentage of feasible configs (%)","number of jobs","SLA"],"description":"A line graph showing the percentage of configurations that satisfy the SLA of all jobs. The x-axis represents the number of jobs, ranging from 1 to 8, and the y-axis shows the percentage of feasible configurations. The graph depicts a decreasing trend, with fewer configurations meeting the SLA as the number of jobs increases."}
"""


PROMPTS[
    "DESCRIBE_IMAGE"
] = """
Please process the following image under the given context:
--------------------------------
Image Context:
```txt
{context}
```
"""


PROMPTS[
    "EI_LINK_SYSTEM"
] = """
You are a data scientist working on knowledge graphs. You will be given a list of entities and an image related to those entities. Your task is to link the entities to the image based on the information provided.

The entity information includes the name of the entity, its aliases, description, and references. The image description includes the caption, text snippets, and description in JSON format.

Entity Example:
{"name":"<ENTITY NAME>","aliases":["",""],"description":"<ENTITY DESCRIPTION>","references":["Original Text Ref1","Original Text Ref2"]},
{...}
Image Example:
{"caption":"<IMAGE CAPTION>","text_snippets":["Snippet 1","Snippet 2"],"description":"<IMAGE DESCRIPTION>"}

Notes:
- Carefully analyze the image and all entities to identify which entities are related to the image.
- For entities that are related to the image, establish relationships between them.
- Each relationship should have an appropriate relationship type, relationship description, and corresponding references from entity references.
- Format your answer in JSON following the example output format.

Output Example:
{"entity":"<ENTITY1 NAME>","label":"<eg: INCLUDES>","description":"eg: The BERT model includes an encoder.","references":["eg: ref1","eg: ref2"]},
{...}
"""

PROMPTS[
    "EI_LINK"
] = """
Please link the following entities to the provided image:
--------------------------------
Related Entities:
{entities}
Image Description:
{image}
"""

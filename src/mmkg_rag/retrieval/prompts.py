"""
Prompts for the retrieval task.
"""

PROMPTS: dict[str, str] = dict()

PROMPTS[
    "CLASSIFY_SYSTEM"
] = """
You are performing a Retrieval-augmented generation (RAG) task. For each user query, please analyze and respond according to the following guidelines:

Step 1: Query Classification
Determine if the query requires external knowledge retrieval:
If you are highly confident about answering the query based on your own knowledge:
Format your response as:
{"classification": "direct","analysis": "Your analysis here","response": "Your response here"}
Moreoever, you can also use mermaid to draw diagrams to help the user understand what you are talking about.


Step 2: Retrieval Keywords Extraction
If the query requires external knowledge:
- Extract essential keywords for retrieval from the query and output them in a JSON array with key "keywords"
- For broad queries, include related concepts that could be relevant and output them in a JSON array with key "extended_keywords"


Examples:
Input: "What is metrics of the Ablation Study?"
Output: 
{"classification":"retrieval","analysis":"The query requires external knowledge retrieval.","keywords":["metrics","ablation study"],"extended_keywords":["evaluation metrics","experimental setup","methodology"]}

Input: "How was the experiment conducted in this paper?"
Output: 
{"classification":"retrieval","analysis":"The query requires external knowledge retrieval.","keywords":["experimental setup","methodology","dataset","metrics","baseline models","ablation study","implementation details"],"extended_keywords":["evaluation metrics","experimental design","data collection","evaluation methodology"]}

Note: Ensure the extracted keywords are:
- Specific enough for meaningful retrieval
- Comprehensive enough to capture related concepts
- Properly formatted in JSON syntax
- Ordered by relevance
"""

PROMPTS[
    "EXTRACT_KEYWORDS"
] = """
You are performing a Retrieval-augmented generation (RAG) task. For each user query, please analyze and respond according to the following guidelines:

The query requires external knowledge retrieval. Please extract essential keywords for retrieval from the query and output them in a JSON array with key "keywords". For broad queries, include related concepts that could be relevant and output them in a JSON array with key "extended_keywords". 

Examples:
Input: "How was the experiment conducted in this paper?"
Output:
{"classification":"retrieval","analysis":"The query requires external knowledge retrieval.","keywords":["experimental setup","methodology","dataset","metrics","baseline models","ablation study","implementation details"],"extended_keywords":["evaluation metrics","experimental design","data collection","evaluation methodology"]}

Note: Ensure the extracted keywords are:
- Specific enough for meaningful retrieval
- Comprehensive enough to capture related concepts
- Properly formatted in JSON syntax
- The key "classification" should be set to "retrieval"
- Ordered by relevance
"""

PROMPTS[
    "CLASSIFY"
] = """
Please answer the following questions by systematically analyzing the user query:
--------
{query}
"""


PROMPTS[
    "GENERATE_SYSTEM"
] = """
You are performing a Retrieval-augmented generation (RAG) task. For user query, please generate a detailed response by incorporating the retrieved knowledge.

All information is stored in the form of a knowledge graph, you will receive the entities and relationships that have been retrieved and related to the user's question, which contains some pictures related to the user's question, you can synthesize the graphic content to give the best answer to the user's question.

In some cases, you can reference the given image in the output, link the image address as a markdown, and annotate the image with CSS styles to make it easier for the user to understand your answer. You can also use mermaid to draw diagrams to help the user understand what you are talking about.
"""

PROMPTS[
    "GENERATE_KNOWLEDGE"
] = """
The following are the relevant entities, relationships and images related to the user query:
--------
{knowledge}
"""

PROMPTS[
    "GENERATE"
] = """
Please synthesize the above information to answer the following user questions, thank you!
--------
{query}
"""


PROMPTS[
    "QUERY_DECOMPOSITION"
] = """
You are performing a Retrieval-augmented generation (RAG) task. For user query, please decompose the question into sub-queries with different aspects. Addressing creative questions requires creative thinking that draws on factual information and an understanding of underlying principles and rules. 

Your task is to define multiple agents which each agent may have a different unique insights and reasoning strategies.
For each agent, please provide the agent task and some possible suggestions for the agent to answer the sub-query.

Output the result in JSON format with the following keys:
- "agents": A list of agents, each with the following keys:
  - "task": The task of the agent.
  - "suggestions": A list of suggestions for the agent to answer the sub-query.


Example:

Input: "What is the impact of the new algorithm on the performance of the model?"
Output:
{
  "agents": [
    {
      "task": "Analyze the performance metrics of the new algorithm.",
      "suggestions": [
        "Compare the performance with previous algorithms.",
        "Evaluate the impact on accuracy and efficiency."
      ]
    },
    {
      "task": "Investigate the theoretical implications of the new algorithm.",
      "suggestions": [
        "Discuss the underlying principles of the algorithm.",
        "Explore potential applications in different domains."
      ]
    }
  ]
}

"""

PROMPTS[
    "FUSION"
] = """
The following are some aspects for a spefic question. Please fusion the multi answers and give a comprehensive answer. Note that the answer should be a comprehensive answer, not a simple summary of the answers. Don't repeat the same content in the answers and don't loss any information.
"""

import os
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# LangChain specific imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS # Can also use MemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # For LLM calls
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# OpenAI specific imports for error handling
from openai import APIError, RateLimitError, AuthenticationError

# --- 1. Configuration and Setup ---

# !! IMPORTANT: Set your OpenAI API Key securely !!
# export OPENAI_API_KEY='your_key_here' in your terminal before running.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002" # OpenAI's recommended embedding model
LLM_MODEL = "gpt-3.5-turbo" # For conceptual summarization

# Clustering threshold for averaged tag embedding similarity (cosine similarity)
SIMILARITY_THRESHOLD = 0.65 # This value might need tuning based on your specific tags and desired group tightness

# Initialize LangChain's OpenAIEmbeddings and ChatOpenAI (conditional on API key)
embeddings_model = OpenAIEmbeddings()
llm = None
if OPENAI_API_KEY:
    print("Initializing LangChain ChatOpenAI client...")
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, openai_api_key=OPENAI_API_KEY)
        # Test a simple call to ensure authentication is working
        llm.invoke("Hello") 
        print("LangChain ChatOpenAI client initialized successfully.")
    except AuthenticationError as e:
        print(f"Authentication Error: {e}. Please check your OpenAI API key. LLM augmentation will be skipped.")
        llm = None
    except Exception as e:
        print(f"An unexpected error occurred during LangChain ChatOpenAI initialization: {e}. LLM augmentation will be skipped.")
        llm = None
else:
    print("OPENAI_API_KEY not set. LLM augmentation will be skipped.")


# --- 2. Article Data (First 2 Articles with Tags Only) ---
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field
from openai_helpers import call_llm

class ArticleTagsRetriever(BaseModel):
    """
    Retrieve tag(s) from given article
    """
    tags: str = Field(description="The tag(s) of the article")

def get_tags_from_article(article : str):
    extracted_tags = call_llm(
        model=ArticleTagsRetriever,
        prompt_template=gpt_system_prompt,
        params={"article":article}
    )
    return extracted_tags.tags.split(",")

gpt_system_prompt = """
You are an expert tag(s) extraction. Your task is to identify and extract up to 10 relevant tag(s) (may be less) from the provided article. 
These tags should accurately reflect the main topics, entities, and keywords present in the article.

--
Article Content : 
{article}
--

**Guidelines:**
* **Relevance:** All extracted tags must be directly related to the content of the article.
* **Specificity:** Only put the specific terms, and omit the general ones.
* **Format:** Return the tags as a single comma-separated string (CSV format), with NO extra spaces.
"""

with open('presidential.issues.json', 'r') as file:
    datajson = json.load(file)

important_field = ['title',
 'main_problem',
 'summary',
 'keypoints',
 'government_actions_already_taken',
 'government_institutions_involved',
 'non_government_institutions_involved',
 'executive_comparing_summary_for_government_and_non_government_institutions',
 'statistics',
 'subtitle',
 'content',
 'issue_category',
 'gepolitic_aspects',
 'peoples_involved']

articles_data = []
n = 1

for data in datajson:

    article = ""

    for field in important_field:
        article += str(f"{field}: {data[field]}\n\n")
    
    tags = get_tags_from_article(article)

    dict_res = {}
    dict_res["id"] = n
    dict_res["title"] = data["title"]
    dict_res["tags"] = tags

    n+=1

    articles_data.append(dict_res)


# Process only the first two articles as requested
articles_to_process = articles_data
print(f"Processing only the first {len(articles_to_process)} articles as requested.")

# --- 3. Create a Global Tag Vocabulary & Get Embeddings for Unique Tags ---
all_unique_tags = set()
for article in articles_to_process:
    for tag in article["tags"]:
        all_unique_tags.add(tag)

unique_tag_list = sorted(list(all_unique_tags)) # Sorted for consistent order

tag_to_embedding: Dict[str, List[float]] = {}

if embeddings_model:
    print(f"\nGenerating OpenAI embeddings for {len(unique_tag_list)} unique tags...")
    try:
        # Batch embedding requests for efficiency and to avoid rate limits
        tag_embeddings_list = embeddings_model.embed_documents(unique_tag_list)
        for i, tag in enumerate(unique_tag_list):
            tag_to_embedding[tag] = tag_embeddings_list[i]
        print("Tag embeddings generated.")
    except (APIError, RateLimitError) as e:
        print(f"Error generating tag embeddings: {e}. Cannot proceed with semantic tag similarity.")
        embeddings_model = None # Disable embeddings for subsequent steps
    except Exception as e:
        print(f"An unexpected error occurred during tag embedding: {e}. Cannot proceed with semantic tag similarity.")
        embeddings_model = None
else:
    print("\nSkipping tag embedding as OpenAI models are not initialized.")
    # If embeddings fail, we cannot proceed with semantic similarity, so exit or fallback.
    exit("Cannot proceed without tag embeddings for semantic similarity.")

# --- 4. Convert Articles to LangChain Documents with Averaged Tag Embeddings ---
langchain_documents: List[Document] = []
document_id_to_index: Dict[str, int] = {}
article_embeddings_for_clustering: List[np.ndarray] = []

print("\nCreating LangChain Documents and calculating article-level tag embeddings (averaged)...")
for i, article in enumerate(articles_to_process):
    article_id = article["id"]
    
    # Get embeddings for all tags of the current article
    article_tag_embeddings = []
    for tag in article["tags"]:
        if tag in tag_to_embedding:
            article_tag_embeddings.append(tag_to_embedding[tag])
    
    if not article_tag_embeddings:
        print(f"Warning: Article {article_id} has no valid tag embeddings. Skipping.")
        continue # Skip this article if it has no valid tag embeddings

    # Average the tag embeddings to get a single vector for the article
    avg_tag_embedding = np.mean(article_tag_embeddings, axis=0).astype('float32')
    
    # Create a LangChain Document.
    # The 'page_content' is just a string of tags for context/LLM.
    # The 'embedding' field in metadata (or the Document itself if a custom doc type)
    # is crucial for our custom similarity.
    doc = Document(
        page_content=f"Tags: {', '.join(article['tags'])}",
        metadata={
            "id": article_id,
            "title": article["title"],
            "original_tags": article["tags"],
            "embedding": avg_tag_embedding.tolist() # Store as list to be JSON serializable
        }
    )
    langchain_documents.append(doc)
    document_id_to_index[article_id] = i
    article_embeddings_for_clustering.append(avg_tag_embedding)
    print(f"  Processed {article_id} with averaged tag embedding.")

# Convert list of embeddings to a NumPy array for clustering
article_embeddings_for_clustering = np.array(article_embeddings_for_clustering)

# --- 5. Similarity Calculation (Optional, for inspection) ---
print("\nCalculating pairwise article similarities based on averaged tag embeddings...")
if len(langchain_documents) > 1:
    similarity_matrix = cosine_similarity(article_embeddings_for_clustering)

    for i, doc_i in enumerate(langchain_documents):
        for j, doc_j in enumerate(langchain_documents):
            if i >= j:
                continue
            score = similarity_matrix[i, j]
            print(f"  {doc_i.metadata['id']} <-> {doc_j.metadata['id']}: {score:.4f}")
else:
    print("  Not enough documents to calculate pairwise similarities (need at least two).")

# --- 6. Agglomerative Clustering for Grouping Documents ---
print(f"\nClustering {len(langchain_documents)} documents into groups based on averaged tag embedding similarity...")

if len(langchain_documents) > 0:
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=1 - SIMILARITY_THRESHOLD
        )
        
        cluster_labels = clustering.fit_predict(article_embeddings_for_clustering)

        grouped_articles: Dict[str, List[Document]] = {}
        for i, label in enumerate(cluster_labels):
            group_id = f"Group_{label}"
            if group_id not in grouped_articles:
                grouped_articles[group_id] = []
            grouped_articles[group_id].append(langchain_documents[i])

        print("\n--- Article Groups ---")
        for group_id, docs_in_group in grouped_articles.items():
            print(f"{group_id}:")
            for doc in docs_in_group:
                print(f"  - {doc.metadata['id']}: '{doc.metadata['title']}' (Tags: {', '.join(doc.metadata['original_tags'])})")

    except Exception as e:
        print(f"\nError during clustering: {e}")
        print("Ensure there are enough documents and valid averaged tag embeddings for clustering.")
else:
    print("No documents to cluster.")

# --- 7. LLM Augmentation for Group Summaries (Using LangChain Chains) ---
print("\n--- LLM Augmentation for Group Summaries (Using LangChain Chains) ---")

if llm: # Only run if LLM client was successfully initialized
    summary_template = """
    Based *only* on the provided article titles and tags, summarize the main themes and key commonalities among these articles.
    Do not invent information beyond what is explicitly in the titles and tags.

    Articles in this group:
    {articles_info}
    """
    summary_prompt = ChatPromptTemplate.from_template(summary_template)

    name_template = """
    Based on this summary and the tags involved: '{summary}', suggest a very concise, descriptive name for this group of articles (3-7 words).
    """
    name_prompt = ChatPromptTemplate.from_template(name_template)

    # LangChain chains
    summary_chain = {"articles_info": RunnableLambda(lambda x: "\n\n---\n\n".join(x))} | summary_prompt | llm | StrOutputParser()
    name_chain = {"summary": RunnableLambda(lambda x: x)} | name_prompt | llm | StrOutputParser()

    print("\nGenerating summaries and names for groups using LangChain LLM chains:")
    for group_id, docs_in_group in grouped_articles.items():
        articles_info_for_llm = []
        for doc in docs_in_group:
            articles_info_for_llm.append(
                f"Article ID: {doc.metadata['id']}\nTitle: {doc.metadata['title']}\nTags: {', '.join(doc.metadata['original_tags'])}"
            )
        
        try:
            group_summary = summary_chain.invoke(articles_info_for_llm)
        except (APIError, RateLimitError) as e:
            print(f"  Error summarizing group {group_id}: {e}. Skipping summary.")
            group_summary = "Error: Could not generate summary."
        except Exception as e:
            print(f"  An unexpected error occurred during summarization for group {group_id}: {e}. Skipping summary.")
            group_summary = "Error: Could not generate summary."

        try:
            group_name = name_chain.invoke(group_summary)
        except (APIError, RateLimitError) as e:
            print(f"  Error naming group {group_id}: {e}. Skipping name generation.")
            group_name = "Error: Could not generate name."
        except Exception as e:
            print(f"  An unexpected error occurred during name generation for group {group_id}: {e}. Skipping name generation.")
            group_name = "Error: Could not generate name."
        
        print(f"\n--- Group {group_id} ---")
        print(f"  Suggested Name: {group_name}")
        # print(f"  Articles: {', '.join([doc.metadata['id'] for doc in docs_in_group])}")
        print(f"  LLM Summary: {group_summary}")
else:
    print("\nSkipping LLM augmentation because LangChain ChatOpenAI client could not be initialized.")

print("\nProcess Complete!")
import json
import os
from typing import Optional, Literal
import openai
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from openai_helpers import call_llm
from sklearn.cluster import KMeans
# from openai.embeddings_utils import get_embedding

class ArticleTagsRetriever(BaseModel):
    """
    Retrieve tag(s) from given article
    """
    tags: str = Field(description="The tag(s) of the article")

def get_embeddings(text_list: list[str]) -> list[float]:
    """
    Given list of text. I want to embed the text so that I could calculate the similarity using cosine
    """
    client = openai.OpenAI(api_key=os.environ("OPEN_API_KEY"))

    response = client.embeddings.create(
        model = "text-embedding-3-large",
        input = text_list
    )

    return [response.data[i].embedding for i in range(len(text_list))]

def get_similarities_from_two_article_tags(article_1_tags: list[str], article_2_tags: list[str]) -> float:
    """
    Analyze the similarities between two article tags

    Input 
    - two different article tags

    Output
    - a float number that determine the similarites between two articles, ranging from 0 (not similar at all) to 1 (very similar)
    """
    
    article_1_tags_joined = ", ".join(article_1_tags)
    article_2_tags_joined = ", ".join(article_2_tags)

    embeddings = get_embeddings([article_1_tags_joined, article_2_tags_joined])

    similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    similarity_score = similarity_matrix[0][0]

    return similarity_score

    
def get_tags_from_article(article: dict) -> list[str]:
    """
    get up to 10 tags from the article. The tags need to be related with the given article

    Input
    - article (in dictionary format)

    Output
    - list of tags
    """

    system_prompt = """
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
    
    reduced_article = {keys : article[keys] for keys in important_field}

    extracted_tags = call_llm(
        model=ArticleTagsRetriever,
        prompt_template=system_prompt,
        params={"article":reduced_article}
    )

    return extracted_tags.tags.split(",")

def group_similar_article(articles: list[dict]) -> list:
    """
    Given a list of articles, I need to group it based on their similarity

    Input
    - list of article

    Output
    - grouped article
    """

    num_of_article = len(articles)
    n_clusters = int(num_of_article ** 0.5) # ini estimate ngasal doang
    tags_list = [get_tags_from_article(article) for article in articles]
    str_tags_list = [ " ".join(tags) for tags in tags_list]
    embeddings_list = get_embeddings(str_tags_list)

    clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_label = clustering_model.fit_predict(embeddings_list)

    return format_results(cluster_label, "K Means", articles)

def format_results(cluster_labels, method_name, articles):
    """Format clustering results"""
    results = {
        'method': method_name,
        'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
        'clusters': {}
    }
    
    # Group articles by cluster
    for i, label in enumerate(cluster_labels):
        if label not in results['clusters']:
            results['clusters'][label] = []
        
        results['clusters'][label].append({
            'title': articles[i]['title']
        })
    
    return results

def get_similar_articles(articles: list[dict]):
    """
    Given list of article, I want to calculate the similarities (using cosine) for any two articles
    Input
    - list of article
    Output
    - dictionary containing article title (as the key) and its similarities with other article (as the value)
    """

    num_of_article = len(articles)
    article_tags = [get_tags_from_article(article) for article in articles]
    article_similarities_table = [[0 for _ in range(num_of_article)] for _ in range(num_of_article)]

    for a in range(num_of_article):
        for b in range(num_of_article):
            article_similarities_table[a][b] = round(get_similarities_from_two_article_tags(article_tags[a], article_tags[b]), 3)
    
    article_dict = {articles[i]["title"]:article_similarities_table[i] for i in range(num_of_article)}

    return article_dict

if __name__ == "__main__":

    with open('presidential.issues.json', 'r') as file:
        datajson = json.load(file)

    # data = datajson[0]
    # data1 = datajson[1]

    # d0 = get_tags_from_article(data)
    # d1 = get_tags_from_article(data1)
    # tags = ["korupsi", "jiwasraya", "OJK", "Benny Tjokrosaputro", "investasi saham", "reksa dana", "skandal besar", "cabut izin usaha", "laporan BPK", "transparansi"]

    # print(get_similarities_from_two_article_tags(d0, d1))
    # print(get_similarities_from_two_article_tags(d0, tags))
    # print(get_similarities_from_two_article_tags(d1, tags))
        
    # result = group_similar_article(datajson)
    # print(result)
        
    result = get_similar_articles(datajson)
    print(result)
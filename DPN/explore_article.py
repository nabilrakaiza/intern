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

article_data = []
n = 1

for data in datajson:

    article = ""

    for field in important_field:
        article += str(f"{field}: {data[field]}\n\n")
    
    tags = get_tags_from_article(article)

    dict_res = {}
    dict_res["id"] = n
    dict_res["title"] = data["title"]
    dict_res["tags"] = ",".join(tags)

    n+=1

    article_data.append(dict_res)

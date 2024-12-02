from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm
import json
import hashlib

def clean_text(text):
    if text is None:
        return ""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

def get_unique_key(post):
    """
    Create a hash of the post text to ensure uniqueness.
    """
    return hashlib.md5(post.get('text', '').encode('utf-8')).hexdigest()

def process_posts(raw_file_path, processed_file_path="data/processed_posts.json"):

    enriched_posts = []
    seen_posts = set() 
    
    try:
        with open(raw_file_path, "r", encoding="utf-8", errors="surrogatepass") as file:
            posts = json.load(file)
        
        for post in posts:
            # Clean the text to remove surrogate characters
            post_text = clean_text(post.get('text', ""))

            # Check if this post has already been processed (using post_id as unique identifier)
            post_id = post.get("id") 
            if post_id and post_id in seen_posts:
                continue  # Skipingg duplicate posts
            
            try:
                metadata = extract_metadata(post_text)
                post_with_metadata = post | metadata
                enriched_posts.append(post_with_metadata)
                if post_id:
                    seen_posts.add(post_id)  # Mark this post as processed
            except OutputParserException as e:
                print(f"Error processing post: {post_text[:100]} - {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        # Unify the tags after processing all posts
        unified_tags = get_unified_tags(enriched_posts)
        
        # Update the existing tags with the unified tags
        for post in enriched_posts:
            post['tags'] = [unified_tags.get(tag, tag) for tag in post['tags']]
        
        # Write the processed posts back to file
        with open(processed_file_path, "w", encoding="utf-8", errors="surrogatepass") as file:
            json.dump(enriched_posts, file, ensure_ascii=False, indent=4)
        
        print(f"Processed {len(enriched_posts)} posts successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_metadata(post):
   
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post, and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language, and tags. 
    3. tags is an array of text tags. Extract a maximum of two tags.
    4. Language should be English or Hinglish (Hinglish means Hindi + English)
    
    Here is the actual post on which you need to perform this task:  
    {post}'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})
    
    try:
        json_parser = JsonOutputParser()
        return json_parser.parse(response.content) 
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse response.")
    except Exception as e:
        print(f"Unexpected error while extracting metadata: {e}")
        raise

def get_unified_tags(posts_with_metadata):
   
    unique_tags = set()

 
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation"}}
    
    Here is the list of tags: 
    {tags}
    '''
    
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    
    return res

def display_processed_posts(file_path):
 
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            posts = json.load(file)
            print("\nProcessed Posts:\n")
            for i, post in enumerate(posts, 1):
                print(f"Post {i}:\n{json.dumps(post, indent=4, ensure_ascii=False)}\n{'-' * 40}")
    except Exception as e:
        print(f"An error occurred while displaying posts: {e}")


if __name__ == "__main__":
    processed_file = "data/processed_posts.json"
    process_posts("data/raw_posts.json", processed_file)
    display_processed_posts(processed_file)







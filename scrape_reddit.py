import sys
import os
import praw # Reddit API Wrapper
from praw.exceptions import RedditAPIException
from dotenv import load_dotenv
import traceback
import time
import re # For cleaning LLM output

# --- Import Hugging Face Client ---
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
# ------------------------------------

# --- Configuration ---
MAX_POSTS_PER_SUBREDDIT = 25
MAX_COMMENTS_PER_POST = 500
SEARCH_KEYWORD = "" # Set based on filename
LLM_FILTER_BATCH_SIZE = 10
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# ---------------------

# --- Load Credentials ---
load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Check Credentials
if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD]):
    print("Error: Missing Reddit API credentials in .env file.", file=sys.stderr)
    sys.exit(1)
if not HF_API_TOKEN:
    print("Error: Missing HF_API_TOKEN in .env file for LLM filtering.", file=sys.stderr)
    sys.exit(1)
# -----------------------------

# --- Initialize API Clients ---
try:
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT, username=REDDIT_USERNAME,
                         password=REDDIT_PASSWORD)
    print("Successfully authenticated with Reddit API.", file=sys.stderr)
except Exception as e:
    print(f"FATAL ERROR: Could not authenticate with Reddit API. Details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

try:
    hf_client = InferenceClient(token=HF_API_TOKEN)
    print(f"Successfully initialized Hugging Face client for model: {LLM_MODEL_NAME}", file=sys.stderr)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize Hugging Face client: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
# -----------------------------

def read_subreddits_from_file(filename):
    """Reads subreddit names (without r/) from a file."""
    filepath = os.path.join(os.path.dirname(__file__), filename) # Assume file is in same dir
    if not os.path.exists(filepath):
        print(f"Error: Subreddit file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subreddits = [line.strip().replace('r/', '') for line in f if line.strip().startswith('r/')]
        if not subreddits:
            print(f"Error: No valid subreddit names found in {filepath}", file=sys.stderr)
            return None
        return subreddits
    except Exception as e:
        print(f"Error reading subreddit file {filepath}: {e}", file=sys.stderr)
        return None


def create_filtering_prompt(titles_list, company_name):
    """Creates a prompt asking the LLM to filter relevant post titles."""
    # (Function remains the same)
    titles_text = "\n".join([f"- {title}" for title in titles_list])
    return f"""
[INST] You are a financial analyst specializing in assessing the stock market impact of news and discussions.
Below is a list of Reddit post titles related to the company '{company_name}'.
Identify ONLY the titles that discuss topics likely to have a DIRECT and SIGNIFICANT impact on the company's stock price. Consider factors like:
- Earnings reports or financial results
- Major product launches or failures
- Mergers, acquisitions, or significant partnerships
- Regulatory news or legal issues
- Scandals or major leadership changes
- Large scale operational news (e.g., factory openings/closings, major layoffs)

Exclude titles discussing:
- General stock price speculation without news
- Minor customer service issues
- Personal investment decisions
- Unsubstantiated rumors
- General brand discussion unrelated to finance/operations

Your task: Review the following titles and list ONLY the ones that are relevant based on the criteria above.

Titles:
{titles_text}

Rules for your response:
1. List ONLY the titles you identified as relevant.
2. Each relevant title must be on a new line.
3. DO NOT include the leading dash ('-').
4. DO NOT add any introduction, explanation, conclusion, or conversational text. Just the list of relevant titles.
[/INST]
"""

def filter_titles_with_llm(titles_list, company_name):
    """Sends titles to the LLM and returns the list of relevant ones."""
    # (Function remains largely the same, added stderr logging)
    if not titles_list:
        return []

    prompt = create_filtering_prompt(titles_list, company_name)
    relevant_titles = []

    try:
        print(f"  Sending {len(titles_list)} titles to LLM for filtering...", file=sys.stderr)
        response = hf_client.chat_completion(
            model=LLM_MODEL_NAME, messages=[{"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.1)
        llm_output = response.choices[0].message.content.strip()

        if llm_output:
            relevant_titles = [line.strip() for line in llm_output.splitlines() if line.strip()]
            print(f"  LLM identified {len(relevant_titles)} relevant titles.", file=sys.stderr)
        else:
            print("  LLM did not identify any relevant titles in this batch.", file=sys.stderr)

    except HfHubHTTPError as e:
        print(f"  LLM API Error: {e}", file=sys.stderr)
        if e.response.status_code == 402:
             print("  FATAL: Hugging Face API credits exceeded. Cannot filter further.", file=sys.stderr)
             raise e # Re-raise to signal fatal error
        elif e.response.status_code == 429:
             print("  Rate limit hit. Pausing for 15 seconds...", file=sys.stderr)
             time.sleep(15)
             print("  Skipping LLM filtering for this batch due to rate limit.", file=sys.stderr)
             return [] # Skip batch
        else:
             print("  Skipping LLM filtering for this batch due to API error.", file=sys.stderr)
             return [] # Skip batch
    except Exception as e:
        print(f"  Unexpected error during LLM filtering: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("  Skipping LLM filtering for this batch.", file=sys.stderr)
        return [] # Skip batch

    return relevant_titles


def scrape_reddit_comments(subreddit_list, output_filepath):
    """Searches posts, filters titles via LLM, scrapes comments."""
    # (Function remains largely the same, added stderr logging)
    print(f"Scraped comments will be saved to: {os.path.basename(output_filepath)}", file=sys.stderr)
    global SEARCH_KEYWORD

    total_comments_saved = 0
    hf_credits_exhausted = False

    try:
        # Use 'w' mode to create/overwrite the file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for sub_name in subreddit_list:
                if hf_credits_exhausted: break

                print(f"\n--- Processing subreddit: r/{sub_name} ---", file=sys.stderr)
                submissions_found = []
                titles_found = []
                try:
                    subreddit = reddit.subreddit(sub_name)
                    print(f"Searching for posts containing '{SEARCH_KEYWORD}'...", file=sys.stderr)
                    # Attempt to access subreddit info to trigger potential errors early
                    try:
                        subreddit.display_name # Accessing an attribute forces fetch
                    except RedditAPIException as sub_error:
                         print(f"  Cannot access r/{sub_name}: {sub_error}. Skipping.", file=sys.stderr)
                         continue # Skip this subreddit

                    for submission in subreddit.search(SEARCH_KEYWORD, sort='new', limit=MAX_POSTS_PER_SUBREDDIT):
                        submissions_found.append(submission)
                        titles_found.append(submission.title)

                    if not submissions_found:
                        print("  No posts found matching the keyword.", file=sys.stderr)
                        continue

                    print(f"  Found {len(titles_found)} posts. Filtering titles via LLM...", file=sys.stderr)
                    all_relevant_titles = set()
                    for i in range(0, len(titles_found), LLM_FILTER_BATCH_SIZE):
                        batch_titles = titles_found[i:i+LLM_FILTER_BATCH_SIZE]
                        try:
                            relevant_batch_titles = filter_titles_with_llm(batch_titles, SEARCH_KEYWORD)
                            all_relevant_titles.update(relevant_batch_titles)
                        except HfHubHTTPError as e:
                            if e.response.status_code == 402:
                                hf_credits_exhausted = True
                                print("Stopping further LLM filtering due to exhausted credits.", file=sys.stderr)
                                break
                            else: pass # Other errors handled within function
                        time.sleep(1) # Delay between LLM calls

                    if hf_credits_exhausted:
                         print("Skipping comment scraping for remaining posts/subreddits.", file=sys.stderr)
                         continue

                    if not all_relevant_titles:
                        print("  LLM found no relevant posts in this subreddit.", file=sys.stderr)
                        continue

                    print(f"\n  Scraping comments from {len(all_relevant_titles)} relevant post(s)...", file=sys.stderr)
                    comments_from_sub = 0
                    for submission in submissions_found:
                        if submission.title.strip() in all_relevant_titles:
                            print(f"    Scraping comments for: '{submission.title[:60]}...'", file=sys.stderr)
                            try:
                                submission.comments.replace_more(limit=0) # Expand top-level comments
                                comment_count_for_post = 0
                                for comment in submission.comments.list(): # Iterate through all comments
                                    if comment_count_for_post >= MAX_COMMENTS_PER_POST: break
                                    if isinstance(comment, praw.models.Comment) and hasattr(comment, 'body') and comment.body:
                                        content = comment.body
                                        sanitized_content = content.replace('\n', ' ').replace('\r', ' ')
                                        f.write(sanitized_content + '\n') # Write comment to file
                                        comment_count_for_post += 1
                                        comments_from_sub += 1
                                        total_comments_saved += 1
                                time.sleep(0.5) # API delay
                            except Exception as post_err:
                                print(f"    Error scraping comments for post ID {submission.id}: {post_err}", file=sys.stderr)
                                traceback.print_exc(file=sys.stderr)

                    print(f"  Saved {comments_from_sub} comments from relevant posts in r/{sub_name}.", file=sys.stderr)

                except Exception as e: # Catch other errors during subreddit processing
                    print(f"Unexpected error processing r/{sub_name}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

                time.sleep(1) # Delay between subreddits

            # Print final summary to STDOUT for Flask
            print(f"\n--- Scraping Complete ---")
            print(f"Successfully saved {total_comments_saved} total comments to {os.path.basename(output_filepath)}")

    except IOError as e:
        print(f"FATAL ERROR: Could not write to file {os.path.basename(output_filepath)}. Details: {e}", file=sys.stderr)
        sys.exit(1) # Exit if cannot write output
    except Exception as e:
        print(f"An unexpected fatal error occurred during scraping: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Exit on unexpected error

# --- Main part of the script ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrape_reddit.py <subreddit_filename.txt>", file=sys.stderr)
        sys.exit(1)

    subreddit_filename_arg = sys.argv[1]
    # Derive base name and keyword from the argument
    # Assumes filename is like 'CompanyName_subreddits.txt'
    try:
        base_name = os.path.basename(subreddit_filename_arg).split('_subreddits.')[0]
    except Exception:
        # Fallback if filename format is unexpected
        base_name = os.path.splitext(os.path.basename(subreddit_filename_arg))[0]

    output_filename = f"{base_name}_reddit.txt"
    SEARCH_KEYWORD = base_name # Use the derived base name as the keyword
    output_filepath = os.path.join(os.path.dirname(__file__), output_filename) # Full path for saving
    subreddit_filepath = os.path.join(os.path.dirname(__file__), subreddit_filename_arg) # Full path for reading

    print(f"Using '{SEARCH_KEYWORD}' as keyword. Output file: {output_filename}", file=sys.stderr)

    subreddits = read_subreddits_from_file(subreddit_filepath)

    if subreddits:
        print(f"Successfully read {len(subreddits)} subreddits from {subreddit_filename_arg}.", file=sys.stderr)
        scrape_reddit_comments(subreddits, output_filepath)
    else:
        print("Exiting script because no valid subreddits were read.", file=sys.stderr)
        sys.exit(1) # Exit if subreddits couldn't be read

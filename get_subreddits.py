import os
import sys
import traceback
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# --- Load secret token from .env file ---
load_dotenv()
# ----------------------------------------

# --- Get token and check it ---
api_token = os.getenv("HF_API_TOKEN")
if not api_token:
    print("Error: HF_API_TOKEN not found in .env file.", file=sys.stderr)
    sys.exit(1) # Exit if token is missing
# ---------------------------------------------

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# ----------------------------------------------

# --- Initialize HF Client ---
# Moved initialization here to fail early if token is invalid
try:
    hf_client = InferenceClient(token=api_token)
    print(f"Hugging Face client initialized for model: {MODEL_NAME}", file=sys.stderr) # Use stderr for logs
except HfHubHTTPError as e:
    print(f"Error initializing Hugging Face client: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during HF client setup: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
# ---------------------

def create_subreddit_prompt(company_name):
    """Creates a prompt to ask the LLM for relevant Reddit subreddits."""
    # (Function remains the same)
    return f"""
[INST] You are a Reddit expert specializing in finance and company discussions.
Generate a list of 5-10 potentially relevant Reddit subreddit names (e.g., r/SubredditName) for finding public opinion and discussions about the company: {company_name}.
Focus on finance, investing, and potentially company-specific or regional subreddits if applicable.

Rules for your response:
1. Provide *only* the list of subreddit names.
2. Each subreddit name must start with 'r/'.
3. Each subreddit name must be on a new line.
4. Do not add any introduction, explanation, or conversational text.
[/INST]
"""

def query_llm_for_subreddits(company_name):
    """Sends the request to the LLM."""
    prompt = create_subreddit_prompt(company_name)
    subreddits_output = "Failed to get subreddits." # Default error

    try:
        print(f"Asking LLM for relevant subreddits for '{company_name}'...", file=sys.stderr)

        response = hf_client.chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        subreddits_output = response.choices[0].message.content.strip()
        print("LLM query successful.", file=sys.stderr)

    except HfHubHTTPError as e:
        subreddits_output = f"Error: API request failed.\nDetails: {e}"
        print(subreddits_output, file=sys.stderr)
        # Check for specific errors like 402 Payment Required
        if e.response.status_code == 402:
            print("FATAL: Hugging Face API credits exceeded.", file=sys.stderr)
            # Re-raise or exit differently if Flask needs to know
            sys.exit(1) # Exit script on fatal API error
        # Don't exit on rate limits, let the calling process handle delays if needed
    except Exception as e:
        subreddits_output = "Failed to get subreddits due to unexpected error."
        print(f"An unexpected error occurred during LLM query: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Consider exiting if this is critical
        # sys.exit(1)

    return subreddits_output

# --- Main part of the script ---
if __name__ == "__main__":
    # --- MODIFIED: Get company name from command-line argument ---
    if len(sys.argv) < 2:
        print("Usage: python get_subreddits.py <CompanyName>")
        sys.exit(1) # Exit if no company name provided
    company = sys.argv[1]
    # --- END MODIFICATION ---

    print(f"Processing company: {company}", file=sys.stderr)
    subreddits_text = query_llm_for_subreddits(company)

    # --- Output results to STDOUT for Flask to capture ---
    # Print suggested subreddits clearly marked
    print("\n--- Suggested Subreddits ---")
    print(subreddits_text)
    # -----------------------------------------------------


    # --- SAVE TO FILE ---
    # File saving logic remains useful for manual runs or debugging
    if not subreddits_text.startswith("Error:") and not subreddits_text.startswith("Failed"):
        safe_filename_base = re.sub(r'[^\w_.-]', '', company.replace(' ', '_'))
        filename = f"{safe_filename_base}_subreddits.txt"
        filepath = os.path.join(os.path.dirname(__file__), filename) # Save in script's dir

        try:
            valid_subreddits = [line for line in subreddits_text.splitlines() if line.strip().startswith('r/')]
            if valid_subreddits:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(valid_subreddits))
                print(f"\nSuccessfully saved suggested subreddits to {filename}", file=sys.stderr)
            else:
                print("\nWarning: LLM did not return valid subreddit names (starting with r/). File not saved.", file=sys.stderr)

        except IOError as e:
            print(f"\nError: Could not save file {filename}. Details: {e}", file=sys.stderr)
    else:
        print("\nDid not save subreddits to file because an error occurred.", file=sys.stderr)
        sys.exit(1) # Exit with error code if LLM failed
    # --- END SAVE TO FILE ---

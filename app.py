import os
import sys
import subprocess
import json
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from datetime import date
import re
import traceback # Added for better error logging
import threading # To run scripts without blocking UI (optional but good)

# --- Configuration ---
SCORE_FILE = "reliability_score.json"
PREDICTION_FILE = "all_predictions.json"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Determine path to python executable within the venv
VENV_PYTHON = os.path.join(PROJECT_DIR, 'venv', 'Scripts', 'python.exe') if sys.platform == 'win32' else os.path.join(PROJECT_DIR, 'venv', 'bin', 'python')
FRONTEND_HTML = 'stock_predictor_ui.html'

# --- Flask App Initialization ---
app = Flask(__name__, static_folder=PROJECT_DIR, static_url_path='') # Serve static files (like JSON) from root

# --- Helper Functions ---

def run_script_sync(script_name, args=[]):
    """Synchronously runs a Python script using the venv's interpreter and captures output."""
    try:
        command = [VENV_PYTHON, os.path.join(PROJECT_DIR, script_name)] + args
        print(f"Running command: {' '.join(command)}") # Log to Flask console

        # --- THIS IS THE CRITICAL FIX ---
        # Get the current environment and force Python to use UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # --- END FIX ---

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',    # This part was already correct
            errors='replace',  # Add this for extra safety
            cwd=PROJECT_DIR,
            check=False,         # Check return code manually
            env=env              # Pass the modified environment to the subprocess
        )

        # Combine stdout and stderr for the log
        output_log = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"

        if result.returncode != 0:
            print(f"Error running {script_name}. Return Code: {result.returncode}", file=sys.stderr)
            print(f"Output Log:\n{output_log}", file=sys.stderr)
            # Try to extract a more specific error from stderr if possible
            error_lines = result.stderr.strip().split('\n')
            last_error_line = error_lines[-1] if error_lines else "Unknown error"
            # Prioritize specific known fatal errors
            if "FATAL: Hugging Face API credits exceeded." in result.stderr:
                 last_error_line = "Hugging Face API credits exceeded."
            elif "FATAL ERROR: Could not authenticate with Reddit API" in result.stderr:
                 last_error_line = "Reddit API authentication failed. Check credentials."
            elif "FATAL ERROR: Could not load the local model" in result.stderr:
                 last_error_line = "Failed to load local sentiment model. Check setup."
            # Updated to catch the specific error
            elif "UnicodeEncodeError" in last_error_line:
                 last_error_line = "Unicode (emoji) encoding error. Check console output."

            return False, f"Error in {script_name}: {last_error_line}", output_log
        else:
            print(f"Successfully ran {script_name}.", file=sys.stderr)
            # Return the STDOUT primarily, but include full log
            return True, result.stdout.strip(), output_log

    except FileNotFoundError:
        error_msg = f"Error: '{VENV_PYTHON}' or script '{script_name}' not found. Check paths."
        print(error_msg, file=sys.stderr)
        return False, error_msg, error_msg # Message and log are the same
    except Exception as e:
        error_msg = f"An unexpected error occurred running {script_name}: {e}"
        full_log = f"{error_msg}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, error_msg, full_log

def safe_read_json(filepath, default_value):
    """Safely reads a JSON file."""
    # Ensure filepath is absolute
    abs_filepath = os.path.join(PROJECT_DIR, filepath)
    if not os.path.exists(abs_filepath):
        print(f"File not found: {abs_filepath}", file=sys.stderr)
        return default_value
    try:
        with open(abs_filepath, 'r', encoding='utf-8') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                print(f"Warning: File {abs_filepath} is empty.", file=sys.stderr)
                return default_value
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read or decode {os.path.basename(filepath)}: {e}. Returning default.", file=sys.stderr)
        return default_value

def get_base_filename(company_name):
    """Creates a safe base filename."""
    # Remove potentially problematic chars, replace space with underscore
    safe_name = re.sub(r'[^\w .-]+', '', company_name) # Allow ., -, _
    return safe_name.replace(' ', '_')

# --- API Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # (Function remains the same)
    try:
        html_filepath = os.path.join(PROJECT_DIR, FRONTEND_HTML)
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return f"Error: {FRONTEND_HTML} not found in {PROJECT_DIR}.", 404
    except Exception as e:
         return f"Error loading page: {e}", 500

@app.route('/api/score', methods=['GET'])
def get_score():
    """Returns the current reliability score."""
    # (Function remains the same)
    score_data = safe_read_json(SCORE_FILE, {"total_predictions": 0, "total_score_points": 0.0})
    return jsonify(score_data)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the prediction history."""
    # (Function remains the same)
    history_data = safe_read_json(PREDICTION_FILE, [])
    if not isinstance(history_data, list): history_data = []
    return jsonify(history_data)

@app.route('/api/predict', methods=['POST'])
def run_prediction_pipeline():
    """Runs the full prediction pipeline."""
    # (Modified to run synchronously and capture output better)
    data = request.get_json()
    company_name = data.get('company_name')

    if not company_name:
        return jsonify({"success": False, "message": "Company name is required."}), 400

    base_filename = get_base_filename(company_name)
    subreddit_file = f"{base_filename}_subreddits.txt"
    comments_file = f"{base_filename}_reddit.txt"

    pipeline_log = [] # Collect logs from all steps

    # --- Step 1: Get Subreddits ---
    print("\n--- Running get_subreddits.py ---", file=sys.stderr)
    success, message, log = run_script_sync('get_subreddits.py', args=[company_name])
    pipeline_log.append(f"--- Step 1: Get Subreddits ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 1 (get subreddits): {message}", "log": "\n".join(pipeline_log)}), 500
    # Check if file was actually created (important!)
    subreddit_filepath = os.path.join(PROJECT_DIR, subreddit_file)
    if not os.path.exists(subreddit_filepath):
        # Extract warning from get_subreddits if possible
        if "Warning: LLM did not return valid subreddit names" in log:
             msg = f"LLM failed to provide valid subreddits for {company_name}."
        else:
             msg = f"Subreddit file '{subreddit_file}' was not created after step 1."
        return jsonify({"success": False, "message": msg, "log": "\n".join(pipeline_log)}), 500


    # --- Step 2: Scrape Reddit ---
    print("\n--- Running scrape_reddit.py ---", file=sys.stderr)
    success, message, log = run_script_sync('scrape_reddit.py', args=[subreddit_file])
    pipeline_log.append(f"\n--- Step 2: Scrape Reddit ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 2 (scrape Reddit): {message}", "log": "\n".join(pipeline_log)}), 500
    # Check if comments file was created
    comments_filepath = os.path.join(PROJECT_DIR, comments_file)
    if not os.path.exists(comments_filepath) or os.path.getsize(comments_filepath) == 0:
         # Check if 0 comments were saved vs file not created
         # Use the message (stdout) from the script for this check
         if "Successfully saved 0 total comments" in message:
             msg = f"No relevant comments found or saved for {company_name}."
             # Not a failure, but analysis cannot proceed
             return jsonify({"success": True, "message": msg, "log": "\n".join(pipeline_log)}), 200
         else:
             msg = f"Comments file '{comments_file}' was not created or is empty after step 2."
             return jsonify({"success": False, "message": msg, "log": "\n".join(pipeline_log)}), 500

    # --- Step 3: Analyze Sentiments ---
    print("\n--- Running analyze_sentiments.py ---", file=sys.stderr)
    success, message, log = run_script_sync('analyze_sentiments.py', args=[comments_file])
    pipeline_log.append(f"\n--- Step 3: Analyze Sentiments ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 3 (analyze sentiments): {message}", "log": "\n".join(pipeline_log)}), 500

    # --- Success ---
    # Extract final prediction from the stdout of analyze_sentiments.py
    final_prediction_display = "Unknown"
    try:
        lines = message.strip().split('\n') # Use message (stdout)
        for line in reversed(lines):
             if "Simplified Prediction:" in line:
                 # Extract the part after the colon and strip whitespace/emojis
                 prediction_part = line.split("Prediction:")[1].strip()
                 if "Rise" in prediction_part or "📈" in prediction_part: final_prediction_display = "Rise 📈"
                 elif "Fall" in prediction_part or "📉" in prediction_part: final_prediction_display = "Fall 📉"
                 elif "Neutral" in prediction_part or "Direction" in prediction_part or "➖" in prediction_part: final_prediction_display = "Neutral ➖"
                 break
    except Exception:
        pass # Keep 'Unknown'

    full_log_output = "\n".join(pipeline_log)
    return jsonify({
        "success": True,
        "message": f"Prediction pipeline completed for {company_name}. Final Prediction: {final_prediction_display}",
        "log": full_log_output
    })

@app.route('/api/check', methods=['POST'])
def run_check_predictions():
    """Runs the verifier script."""
    # (Modified to run synchronously and return updated data)
    print("\n--- Running verifier.py check ---", file=sys.stderr)
    success, message, log = run_script_sync('verifier.py', args=['check'])

    # Always fetch fresh data after running check
    score_data = safe_read_json(SCORE_FILE, {"total_predictions": 0, "total_score_points": 0.0})
    history_data = safe_read_json(PREDICTION_FILE, [])
    if not isinstance(history_data, list): history_data = []

    if not success:
        # Verifier script might fail internally but still update files
        return jsonify({
            "success": False, # Indicate script error
            "message": f"Verifier script ran with issues: {message}",
            "log": log,
            "updated_score": score_data,
            "updated_history": history_data
        }), 500 # Internal Server Error status

    # If script ran without error code, extract summary from stdout (message)
    summary = message # message contains the stdout from verifier.py
    return jsonify({
        "success": True,
        "message": "Verifier script executed successfully.", # Generic success, details in log
        "log": log, # Include full log (stdout+stderr)
        "summary": summary, # Pass the verifier's stdout summary
        "updated_score": score_data,
        "updated_history": history_data
    })


# --- Static file serving ---
@app.route('/<path:filename>')
def serve_static(filename):
    # Allow serving the essential JSON files and the HTML file itself
    allowed_files = [SCORE_FILE, PREDICTION_FILE, FRONTEND_HTML]
    # Basic security
    if filename.endswith('.py') or filename == '.env':
         return "Access denied.", 403
    # Allow analyzed json files too
    if filename.endswith('_analyzed.json') or filename.endswith('_analyzed_llm.json'):
         allowed_files.append(filename)

    if filename in allowed_files:
        try:
            mime = 'application/json' if filename.endswith('.json') else None
            return send_from_directory(
                PROJECT_DIR,
                filename,
                mimetype=mime,
                # Add headers to prevent caching by browser - important for dynamic JSON
                cache_timeout=-1 # Effectively disable caching
            )
        except FileNotFoundError:
             return "File not found.", 404
    else:
        print(f"Denied access to static file: {filename}", file=sys.stderr)
        return "File not found or access denied.", 404


# --- Run the App ---
if __name__ == '__main__':
    print("--- Starting Stock Sentiment Predictor Flask App ---")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Venv Python Path: {VENV_PYTHON}")
    # Check if venv python exists
    if not os.path.exists(VENV_PYTHON):
        print("\n!!! WARNING !!!")
        print(f"Python executable not found at expected venv path: {VVENV_PYTHON}")
        print("Subprocess calls to scripts will likely fail.")
        print("Ensure Flask is running from within the activated virtual environment OR")
        print("adjust VENV_PYTHON path in app.py if your structure is different.")
        print("-------------\n")

    print("Flask server starting...")
    print("Open http://127.0.0.1:5000/ in your browser.")
    # Use host='0.0.0.0' to make accessible on local network if needed
    # use_reloader=False prevents Flask running setup twice, might be more stable
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
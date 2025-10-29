import os
import sys
import subprocess
import json
# --- MODIFIED: Added datetime imports ---
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from datetime import date, datetime, timedelta 
import re
import traceback 
import threading 
# --- END MODIFICATION ---

# --- Configuration ---
SCORE_FILE = "reliability_score.json"
PREDICTION_FILE = "all_predictions.json"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, 'venv', 'Scripts', 'python.exe') if sys.platform == 'win32' else os.path.join(PROJECT_DIR, 'venv', 'bin', 'python')
FRONTEND_HTML = 'stock_predictor_ui.html'

# --- Flask App Initialization ---
app = Flask(__name__, static_folder=PROJECT_DIR, static_url_path='') 

# --- Helper Functions ---
# (run_script_sync, safe_read_json, get_base_filename functions remain the same)
def run_script_sync(script_name, args=[]):
    """Synchronously runs a Python script using the venv's interpreter and captures output."""
    try:
        command = [VENV_PYTHON, os.path.join(PROJECT_DIR, script_name)] + args
        print(f"Running command: {' '.join(command)}") 

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',   
            errors='replace', 
            cwd=PROJECT_DIR,
            check=False,        
            env=env             
        )

        output_log = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"

        if result.returncode != 0:
            print(f"Error running {script_name}. Return Code: {result.returncode}", file=sys.stderr)
            print(f"Output Log:\n{output_log}", file=sys.stderr)
            error_lines = result.stderr.strip().split('\n')
            last_error_line = error_lines[-1] if error_lines else "Unknown error"
            if "FATAL: Hugging Face API credits exceeded." in result.stderr:
                 last_error_line = "Hugging Face API credits exceeded."
            elif "FATAL ERROR: Could not authenticate with Reddit API" in result.stderr:
                 last_error_line = "Reddit API authentication failed. Check credentials."
            elif "FATAL ERROR: Could not load the local model" in result.stderr:
                 last_error_line = "Failed to load local sentiment model. Check setup."
            elif "UnicodeEncodeError" in last_error_line:
                 last_error_line = "Unicode (emoji) encoding error. Check console output."

            return False, f"Error in {script_name}: {last_error_line}", output_log
        else:
            print(f"Successfully ran {script_name}.", file=sys.stderr)
            return True, result.stdout.strip(), output_log

    except FileNotFoundError:
        error_msg = f"Error: '{VENV_PYTHON}' or script '{script_name}' not found. Check paths."
        print(error_msg, file=sys.stderr)
        return False, error_msg, error_msg 
    except Exception as e:
        error_msg = f"An unexpected error occurred running {script_name}: {e}"
        full_log = f"{error_msg}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, error_msg, full_log

def safe_read_json(filepath, default_value):
    """Safely reads a JSON file."""
    abs_filepath = os.path.join(PROJECT_DIR, filepath)
    if not os.path.exists(abs_filepath):
        print(f"File not found: {abs_filepath}", file=sys.stderr)
        return default_value
    try:
        with open(abs_filepath, 'r', encoding='utf-8') as f:
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
    safe_name = re.sub(r'[^\w .-]+', '', company_name)
    return safe_name.replace(' ', '_')

# --- API Routes ---
# (/, /api/score, /api/history, /api/score/all routes remain the same)
@app.route('/')
def index():
    """Serves the main HTML page."""
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
    """Returns the *global* reliability score."""
    default_global_score = {"total_predictions": 0, "total_score_points": 0.0}
    default_file_content = {
        "global": default_global_score,
        "companies": {}
    }
    score_data_full = safe_read_json(SCORE_FILE, default_file_content)

    if "global" in score_data_full:
        global_score = score_data_full["global"]
    elif "total_predictions" in score_data_full:
        global_score = {
             "total_predictions": score_data_full.get("total_predictions", 0),
             "total_score_points": score_data_full.get("total_score_points", 0.0)
        }
    else:
        global_score = default_global_score

    return jsonify(global_score)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the prediction history."""
    history_data = safe_read_json(PREDICTION_FILE, [])
    if not isinstance(history_data, list): history_data = []
    return jsonify(history_data)

@app.route('/api/score/all', methods=['GET'])
def get_all_scores():
    """Returns the full reliability score object (global and per-company)."""
    default_score_data = {
        "global": {"total_predictions": 0, "total_score_points": 0.0},
        "companies": {}
    }
    all_score_data = safe_read_json(SCORE_FILE, default_score_data)
    
    if "global" not in all_score_data and "total_predictions" in all_score_data:
         print("Migrating old score format in /api/score/all...", file=sys.stderr)
         old_global_data = {
            "total_predictions": all_score_data.get("total_predictions", 0),
            "total_score_points": all_score_data.get("total_score_points", 0.0)
         }
         all_score_data = {
            "global": old_global_data,
            "companies": {}
         }
            
    return jsonify(all_score_data)

@app.route('/api/predict', methods=['POST'])
def run_prediction_pipeline():
    """Runs the full prediction pipeline."""
    # --- MODIFIED: Get date and calculate window ---
    data = request.get_json()
    company_name = data.get('company_name')
    prediction_date_str = data.get('prediction_date') # e.g., "2025-06-08"

    if not company_name:
        return jsonify({"success": False, "message": "Company name is required."}), 400
    if not prediction_date_str:
        return jsonify({"success": False, "message": "Prediction date is required."}), 400

    # Calculate 7-day window
    try:
        # This is the date the prediction is FOR (e.g., June 8)
        prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
        
        # Data collection ends the day *before* (e.g., June 7)
        end_date = prediction_date - timedelta(days=1)
        
        # Data collection starts 6 days *before* that (e.g., June 1)
        # This creates a 7-day window: (June 1, 2, 3, 4, 5, 6, 7)
        start_date = end_date - timedelta(days=6)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

    except ValueError:
        return jsonify({"success": False, "message": "Invalid date format."}), 400
    
    print(f"Prediction for: {prediction_date_str}. Scraping data from {start_date_str} to {end_date_str}.", file=sys.stderr)

    base_filename = get_base_filename(company_name)
    subreddit_file = f"{base_filename}_subreddits.txt"
    comments_file = f"{base_filename}_reddit.txt"

    pipeline_log = [] 

    # --- Step 1: Get Subreddits (No change) ---
    print("\n--- Running get_subreddits.py ---", file=sys.stderr)
    success, message, log = run_script_sync('get_subreddits.py', args=[company_name])
    pipeline_log.append(f"--- Step 1: Get Subreddits ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 1 (get subreddits): {message}", "log": "\n".join(pipeline_log)}), 500
    subreddit_filepath = os.path.join(PROJECT_DIR, subreddit_file)
    if not os.path.exists(subreddit_filepath):
        if "Warning: LLM did not return valid subreddit names" in log:
             msg = f"LLM failed to provide valid subreddits for {company_name}."
        else:
             msg = f"Subreddit file '{subreddit_file}' was not created after step 1."
        return jsonify({"success": False, "message": msg, "log": "\n".join(pipeline_log)}), 500


    # --- Step 2: Scrape Reddit (MODIFIED: pass dates) ---
    print("\n--- Running scrape_reddit.py ---", file=sys.stderr)
    # Pass start_date_str and end_date_str as new arguments
    scrape_args = [subreddit_file, start_date_str, end_date_str]
    success, message, log = run_script_sync('scrape_reddit.py', args=scrape_args)
    pipeline_log.append(f"\n--- Step 2: Scrape Reddit ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 2 (scrape Reddit): {message}", "log": "\n".join(pipeline_log)}), 500
    comments_filepath = os.path.join(PROJECT_DIR, comments_file)
    if not os.path.exists(comments_filepath) or os.path.getsize(comments_filepath) == 0:
         if "Successfully saved 0 total comments" in message:
             msg = f"No relevant comments found or saved for {company_name} in the specified date range."
             return jsonify({"success": True, "message": msg, "log": "\n".join(pipeline_log)}), 200
         else:
             msg = f"Comments file '{comments_file}' was not created or is empty after step 2."
             return jsonify({"success": False, "message": msg, "log": "\n".join(pipeline_log)}), 500

    # --- Step 3: Analyze Sentiments (MODIFIED: pass prediction_date) ---
    print("\n--- Running analyze_sentiments.py ---", file=sys.stderr)
    # Pass the original prediction_date_str as the date to save
    analyze_args = [comments_file, prediction_date_str]
    success, message, log = run_script_sync('analyze_sentiments.py', args=analyze_args)
    pipeline_log.append(f"\n--- Step 3: Analyze Sentiments ---\n{log}")
    if not success:
        return jsonify({"success": False, "message": f"Failed step 3 (analyze sentiments): {message}", "log": "\n".join(pipeline_log)}), 500
    # --- END MODIFICATION ---

    # --- Success (No change) ---
    final_prediction_display = "Unknown"
    try:
        lines = message.strip().split('\n')
        for line in reversed(lines):
             if "Simplified Prediction:" in line:
                 prediction_part = line.split("Prediction:")[1].strip()
                 if "Rise" in prediction_part or "📈" in prediction_part: final_prediction_display = "Rise 📈"
                 elif "Fall" in prediction_part or "📉" in prediction_part: final_prediction_display = "Fall 📉"
                 elif "Neutral" in prediction_part or "Direction" in prediction_part or "➖" in prediction_part: final_prediction_display = "Neutral ➖"
                 break
    except Exception:
        pass 

    full_log_output = "\n".join(pipeline_log)
    return jsonify({
        "success": True,
        "message": f"Prediction pipeline completed for {company_name} (for {prediction_date_str}). Final Prediction: {final_prediction_display}",
        "log": full_log_output
    })

@app.route('/api/check', methods=['POST'])
def run_check_predictions():
    """Runs the verifier script."""
    # (This function remains the same)
    print("\n--- Running verifier.py check ---", file=sys.stderr)
    success, message, log = run_script_sync('verifier.py', args=['check'])

    default_global_score = {"total_predictions": 0, "total_score_points": 0.0}
    default_file_content = {"global": default_global_score, "companies": {}}
    score_data_full = safe_read_json(SCORE_FILE, default_file_content)
    
    if "global" in score_data_full: global_score = score_data_full["global"]
    elif "total_predictions" in score_data_full: global_score = {k: score_data_full[k] for k in ("total_predictions", "total_score_points") if k in score_data_full}
    else: global_score = default_global_score

    history_data = safe_read_json(PREDICTION_FILE, [])
    if not isinstance(history_data, list): history_data = []

    if not success:
        return jsonify({
            "success": False, 
            "message": f"Verifier script ran with issues: {message}",
            "log": log,
            "updated_score": global_score, 
            "updated_history": history_data
        }), 500 

    summary = message 
    return jsonify({
        "success": True,
        "message": "Verifier script executed successfully.",
        "log": log, 
        "summary": summary, 
        "updated_score": global_score, 
        "updated_history": history_data
    })


# --- Static file serving ---
# (serve_static function remains the same)
@app.route('/<path:filename>')
def serve_static(filename):
    allowed_files = [SCORE_FILE, PREDICTION_FILE, FRONTEND_HTML]
    if filename.endswith('.py') or filename == '.env':
         return "Access denied.", 403
    if filename.endswith('_analyzed.json') or filename.endswith('_analyzed_llm.json'):
         allowed_files.append(filename)

    if filename in allowed_files:
        try:
            mime = 'application/json' if filename.endswith('.json') else None
            return send_from_directory(
                PROJECT_DIR,
                filename,
                mimetype=mime,
                cache_timeout=-1 
            )
        except FileNotFoundError:
             return "File not found.", 404
    else:
        print(f"Denied access to static file: {filename}", file=sys.stderr)
        return "File not found or access denied.", 404


# --- Run the App ---
# (Main run block remains the same)
if __name__ == '__main__':
    print("--- Starting Stock Sentiment Predictor Flask App ---")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Venv Python Path: {VENV_PYTHON}")
    if not os.path.exists(VENV_PYTHON):
        print("\n!!! WARNING !!!")
        print(f"Python executable not found at expected venv path: {VENV_PYTHON}")
        print("Subprocess calls to scripts will likely fail.")
        print("Ensure Flask is running from within the activated virtual environment OR")
        print("adjust VENV_PYTHON path in app.py if your structure is different.")
        print("-------------\n")

    print("Flask server starting...")
    print("Open http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
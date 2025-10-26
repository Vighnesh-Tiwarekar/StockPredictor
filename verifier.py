import os
import sys
import json
import yfinance as yf
from datetime import date, timedelta, datetime
import traceback # Added for better error logging

# --- Configuration ---
TICKER_MAP = {
    "reliance": "RELIANCE.NS",
    "apple": "AAPL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "tata": "TATAMOTORS.NS" # Make sure this is added/correct
    # Add more companies here
}
SCORE_FILE = "reliability_score.json"
PREDICTION_FILE = "all_predictions.json"
MAX_SCORE_THRESHOLD_PERCENT = 5.0
# ---------------------

def load_reliability_score():
    """Loads the current score from its JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), SCORE_FILE)
    if not os.path.exists(filepath):
        return {"total_predictions": 0, "total_score_points": 0.0}
    try:
        with open(filepath, 'r') as f:
            score_data = json.load(f)
            # Ensure fields exist, migrating if necessary
            if "total_score_points" not in score_data:
                score_data["total_score_points"] = float(score_data.get("correct_predictions", 0))
            if "total_predictions" not in score_data:
                 score_data["total_predictions"] = int(score_data.get("correct_predictions", 0) + score_data.get("incorrect_predictions", 0)) # Try to migrate old format
            # Clean up old keys if migrating
            score_data.pop("correct_predictions", None)
            score_data.pop("incorrect_predictions", None)
            return score_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: {SCORE_FILE} is corrupted or unreadable: {e}. Resetting score.", file=sys.stderr)
        return {"total_predictions": 0, "total_score_points": 0.0}

def save_reliability_score(score_data):
    """Saves the updated score back to its JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), SCORE_FILE)
    try:
        with open(filepath, 'w') as f:
            json.dump(score_data, f, indent=2)
    except IOError as e:
        print(f"Error saving reliability score to {filepath}: {e}", file=sys.stderr)

def load_all_predictions():
    """Loads the master list of all predictions."""
    filepath = os.path.join(os.path.dirname(__file__), PREDICTION_FILE)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: {PREDICTION_FILE} is corrupted or unreadable: {e}. Returning empty list.", file=sys.stderr)
        return []

def save_all_predictions(predictions_list):
    """Saves the modified master list back to the file."""
    filepath = os.path.join(os.path.dirname(__file__), PREDICTION_FILE)
    try:
        with open(filepath, 'w') as f:
            json.dump(predictions_list, f, indent=2)
    except IOError as e:
         print(f"Error saving predictions log to {filepath}: {e}", file=sys.stderr)


def get_actual_stock_movement(ticker_symbol, start_date_str):
    """Fetches stock data and returns (percent_change, check_date_str)."""
    # (Function remains the same, added stderr logging)
    try:
        tk = yf.Ticker(ticker_symbol)
        end_date = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        # Ensure start_date is valid format
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"  Error: Invalid date format '{start_date_str}'. Skipping.", file=sys.stderr)
            return "error", None

        hist = tk.history(start=start_date_str, end=end_date)

        if len(hist) < 2:
            return None, None # Not enough trading days

        prediction_day_close = hist['Close'].iloc[0]
        check_day_close = hist['Close'].iloc[-1]
        check_date = hist.index[-1].date()

        if check_date == start_dt:
             return None, None # Still the same day or only one day of data

        percent_change = ((check_day_close - prediction_day_close) / prediction_day_close) * 100
        return percent_change, str(check_date)

    except Exception as e:
        print(f"  Error fetching stock data for {ticker_symbol}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print full traceback for yfinance errors
        return "error", None

def clamp(value, min_val, max_val):
    """Keeps a value between a min and max."""
    # (Function remains the same)
    return max(min_val, min(value, max_val))

def calculate_scaled_score(prediction, percent_change):
    """Calculates a scaled score from 0.0 to 1.0."""
    # (Function remains the same)
    max_thresh = MAX_SCORE_THRESHOLD_PERCENT
    score = 0.0
    if prediction == "rise":
        score = (percent_change + max_thresh) / (2 * max_thresh)
    elif prediction == "fall":
        score = 1.0 - ((percent_change + max_thresh) / (2 * max_thresh))
    elif prediction == "neutral":
        score = 1.0 - (abs(percent_change) / max_thresh)
    return clamp(score, 0.0, 1.0)

def get_correctness(prediction, percent_change):
    """Returns 'Correct' or 'Incorrect' based on a simple threshold."""
    NEUTRAL_THRESHOLD = 0.5 # A move of +/- 0.5% is considered neutral
    
    if prediction == "rise":
        return "Correct 👍" if percent_change > NEUTRAL_THRESHOLD else "Incorrect 👎"
    elif prediction == "fall":
        return "Correct 👍" if percent_change < -NEUTRAL_THRESHOLD else "Incorrect 👎"
    elif prediction == "neutral":
        return "Correct 👍" if abs(percent_change) <= NEUTRAL_THRESHOLD else "Incorrect 👎"
    return "N/A"


def check_pending_predictions():
    """Checks all pending predictions and updates score/log."""
    # (Modified to print simplified summary to STDOUT and details to STDERR)
    all_predictions = load_all_predictions()
    if not all_predictions:
        print(f"No predictions found in {PREDICTION_FILE}.", file=sys.stderr) # MODIFIED: stderr
        return

    score_data = load_reliability_score()
    predictions_checked = 0
    predictions_updated = False

    # Create summary lists for final output
    checked_summary = []
    skipped_summary = []
    error_summary = []

    # --- NEW: List for STDOUT summary ---
    stdout_summary = []

    for prediction in all_predictions:
        if prediction.get("status") != "pending":
            continue

        company = prediction.get('company', 'Unknown') # Safer access
        saved_prediction = prediction.get('prediction')
        date_saved = prediction.get('date_saved')

        # Basic validation
        if not all([company, saved_prediction, date_saved]):
             print(f"\nSkipping invalid prediction entry: {prediction}", file=sys.stderr)
             prediction['status'] = 'error'
             prediction['error_message'] = 'Missing company, prediction, or date_saved'
             error_summary.append(f"{company or 'Unknown'} ({date_saved or 'N/A'}): Invalid entry")
             predictions_updated = True
             continue

        # Check if prediction was made today
        try:
            date_saved_dt = datetime.strptime(date_saved, '%Y-%m-%d').date()
            if date_saved_dt >= date.today(): # Don't check today's predictions
                 skipped_summary.append(f"{company} ({date_saved}): Saved today, check tomorrow")
                 continue
        except ValueError:
             print(f"\nSkipping prediction for '{company}' due to invalid date_saved: {date_saved}", file=sys.stderr)
             prediction['status'] = 'error'
             prediction['error_message'] = f"Invalid date format: {date_saved}"
             error_summary.append(f"{company} ({date_saved}): Invalid date")
             predictions_updated = True
             continue

        # MODIFIED: Print logs to stderr
        print(f"\nChecking pending prediction for '{company}' from {date_saved}...", file=sys.stderr) 
        print(f"  Your prediction was: {saved_prediction}", file=sys.stderr)

        # 1. Get Ticker
        if company not in TICKER_MAP:
            print(f"  Error: Company '{company}' not in TICKER_MAP. Skipping.", file=sys.stderr) # MODIFIED: stderr
            prediction['status'] = 'error'
            prediction['error_message'] = 'Company not in TICKER_MAP'
            error_summary.append(f"{company} ({date_saved}): Not in TICKER_MAP")
            predictions_updated = True
            continue
        ticker = TICKER_MAP[company]

        # 2. Get Stock Movement
        percent_change, check_date = get_actual_stock_movement(ticker, date_saved)

        if percent_change == "error":
            print(f"  Could not fetch data for {ticker}. Skipping check for now.", file=sys.stderr) # MODIFIED: stderr
            # Keep status as pending, maybe data will be available later
            error_summary.append(f"{company} ({date_saved}): Data fetch error")
            continue # Don't mark as error, just skip this run

        if percent_change is None:
            print(f"  Not enough trading data yet for {ticker} since {date_saved}. Will check again later.", file=sys.stderr) # MODIFIED: stderr
            skipped_summary.append(f"{company} ({date_saved}): Insufficient data")
            continue # Keep status as pending

        # 3. Check successful
        predictions_checked += 1
        predictions_updated = True
        
        # MODIFIED: Print detailed log to stderr
        print(f"  Actual stock movement from {date_saved} to {check_date}: {percent_change:+.2f}%", file=sys.stderr)

        # 4. Calculate score
        prediction_score = calculate_scaled_score(saved_prediction, percent_change)
        print(f"  --- Prediction Score: {prediction_score:.2f} / 1.0 ---", file=sys.stderr) # MODIFIED: stderr

        # --- MODIFIED: Get correctness and build STDOUT summary ---
        correctness = get_correctness(saved_prediction, percent_change)
        stdout_summary.append(f"• Company: {company.capitalize()}")
        stdout_summary.append(f"• Prediction: {saved_prediction.capitalize()}")
        stdout_summary.append(f"• Actual Movement ({date_saved} to {check_date}): {percent_change:+.2f}%")
        stdout_summary.append(f"• Result: {correctness}")
        stdout_summary.append("") # Add a blank line for separation
        # --- END MODIFIED BLOCK ---

        # 5. Update score data
        score_data["total_predictions"] += 1
        score_data["total_score_points"] += prediction_score

        # 6. Update prediction entry
        prediction["status"] = "checked"
        prediction["date_checked"] = check_date
        prediction["actual_movement_percent"] = round(percent_change, 2)
        prediction["prediction_score"] = round(prediction_score, 4)
        checked_summary.append(f"{company} ({date_saved}): Score {prediction_score:.2f}")


    # 7. Save results and print summary to STDERR
    summary_output = []
    if predictions_updated:
        save_all_predictions(all_predictions)
        save_reliability_score(score_data)

        reliability = 0.0
        if score_data["total_predictions"] > 0:
            reliability = score_data["total_score_points"] / score_data["total_predictions"]

        summary_output.append(f"\nChecked {predictions_checked} new prediction(s).")
        summary_output.append("--- Reliability Score Updated ---")
        summary_output.append(f"Total Points: {score_data['total_score_points']:.2f} / Total Predictions: {score_data['total_predictions']}")
        summary_output.append(f"New Reliability Score: {reliability:.3f} (or {reliability*100:.1f}%)")
        summary_output.append(f"Updated '{PREDICTION_FILE}' and '{SCORE_FILE}'.")
    else:
        summary_output.append("\nNo pending predictions were ready to be checked.")

    # Add details about skipped/errored items
    if skipped_summary:
        summary_output.append("\nSkipped (will retry later):")
        summary_output.extend([f"- {s}" for s in skipped_summary])
    if error_summary:
         summary_output.append("\nErrors encountered (marked as 'error' in log):")
         summary_output.extend([f"- {e}" for e in error_summary])

    # --- MODIFIED: Print summaries to correct streams ---
    
    # Print simplified summary to STDOUT (for the green box)
    if stdout_summary:
        # Join all lines, removing the very last blank line
        print("\n".join(stdout_summary).strip())
    elif not predictions_updated:
         print("No pending predictions were ready to be checked.") # STDOUT
    else:
         print("Predictions checked, but none matched output criteria.") # STDOUT

    # Print detailed log summary to STDERR (for the black log box)
    print("\n".join(summary_output), file=sys.stderr) 


def print_usage():
    """Prints the help message."""
    print("Usage: python verifier.py <command>")
    print("\nCommands:")
    print("  check")
    print("    (Checks all pending predictions from 'all_predictions.json')")
    print("\n  score")
    print("    (Shows the current reliability score from 'reliability_score.json')")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "check":
        check_pending_predictions()

    elif command == "score":
        score = load_reliability_score()
        # Print score info to STDOUT
        if score["total_predictions"] == 0:
            print("No predictions made yet. Score is 0.0/0.")
        else:
            reliability = score["total_score_points"] / score["total_predictions"]
            print(f"Current Reliability: {score['total_score_points']:.2f} Points / {score['total_predictions']} Predictions ({reliability*100:.1f}%)")

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print_usage()
        sys.exit(1)
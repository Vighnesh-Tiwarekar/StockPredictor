import os
import sys
import json
import yfinance as yf
from datetime import date, timedelta, datetime
import traceback # Added for better error logging

# --- Configuration ---
TICKER_MAP = {
    # Existing
    "reliance": "RELIANCE.NS",
    "apple": "AAPL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "tata": "TATAMOTORS.NS",
    
    # Newly Added
    "amazon": "AMZN",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "meta": "META", 
    "jpmorgan": "JPM", 
    "j&j": "JNJ", 
    "walmart": "WMT",
    "samsung": "005930.KS", 
    "toyota": "TM",
    "alibaba": "BABA"
}
SCORE_FILE = "reliability_score.json"
PREDICTION_FILE = "all_predictions.json"
MAX_SCORE_THRESHOLD_PERCENT = 5.0 # Used for scaling the 0-1 score
# ---------------------

# (load_reliability_score, save_reliability_score, load_all_predictions, save_all_predictions functions remain the same)
def load_reliability_score():
    """Loads the current score from its JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), SCORE_FILE)
    
    default_score_data = {
        "global": {"total_predictions": 0, "total_score_points": 0.0},
        "companies": {}
    }
    
    if not os.path.exists(filepath):
        return default_score_data
        
    try:
        with open(filepath, 'r') as f:
            score_data = json.load(f)

        if "global" not in score_data and "total_predictions" in score_data:
            print(f"Warning: Found old score format. Migrating to new structure...", file=sys.stderr)
            old_global_data = {
                "total_predictions": score_data.get("total_predictions", 0),
                "total_score_points": score_data.get("total_score_points", 0.0)
            }
            score_data = {
                "global": old_global_data,
                "companies": {} 
            }
            save_reliability_score(score_data)
            print(f"Migration complete. Score file saved.", file=sys.stderr)
            return score_data

        if "global" not in score_data:
            score_data["global"] = {"total_predictions": 0, "total_score_points": 0.0}
        if "companies" not in score_data:
            score_data["companies"] = {}

        return score_data
            
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: {SCORE_FILE} is corrupted or unreadable: {e}. Resetting score.", file=sys.stderr)
        return default_score_data

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


# --- THIS IS THE MODIFIED FUNCTION ---
def get_actual_stock_movement(ticker_symbol, prediction_for_date_str):
    """
    Fetches stock data to compare the close price of the last trading day *before*
    the prediction_for_date with the close price of the first trading day *on or after* it.
    
    Returns: (percent_change, base_date_str, check_date_str)
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        
        # 1. Parse the date the prediction is for
        try:
            prediction_date = datetime.strptime(prediction_for_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"  Error: Invalid date format '{prediction_for_date_str}'. Skipping.", file=sys.stderr)
            return "error", None, None

        # 2. Get the Base Price (Close price *before* the prediction date)
        # Fetch 5 days of history ending *before* the prediction date.
        hist_before = tk.history(end=prediction_date.strftime('%Y-%m-%d'), period="5d")
        
        if hist_before.empty:
            print(f"  Error: No historical data found *before* {prediction_for_date_str} for {ticker_symbol}. Cannot get base price.", file=sys.stderr)
            return "error", None, None

        base_price = hist_before['Close'].iloc[-1]
        base_date = hist_before.index[-1].date()

        # 3. Get the Check Price (Close price *on or after* the prediction date)
        # Fetch 5 days of history *starting from* the prediction date.
        check_start_date = prediction_date.strftime('%Y-%m-%d')
        check_end_date = (prediction_date + timedelta(days=5)).strftime('%Y-%m-%d')
        hist_after = tk.history(start=check_start_date, end=check_end_date)
        
        if hist_after.empty:
            # This happens if the prediction_date is today or in the future
            print(f"  No trading data found *on or after* {prediction_for_date_str}. Will check again later.", file=sys.stderr)
            return None, None, None 

        check_price = hist_after['Close'].iloc[0]
        check_date = hist_after.index[0].date()

        # 4. Handle case where base_date and check_date are the same
        # (e.g., ran check script twice in one day)
        if base_date == check_date:
             print(f"  Not enough trading data yet. Base date and check date are both {base_date}. Will check again later.", file=sys.stderr)
             return None, None, None

        # 5. Calculate percentage change
        percent_change = ((check_price - base_price) / base_price) * 100
        return percent_change, str(base_date), str(check_date)

    except Exception as e:
        print(f"  Error fetching stock data for {ticker_symbol}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) 
        return "error", None, None
# --- END MODIFIED FUNCTION ---


def clamp(value, min_val, max_val):
    """Keeps a value between a min and max."""
    return max(min_val, min(value, max_val))

def calculate_scaled_score(prediction, percent_change):
    """Calculates a scaled score from 0.0 to 1.0."""
    # (This function remains the same)
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
    # (This function remains the same)
    NEUTRAL_THRESHOLD = 0.5 
    
    if prediction == "rise":
        return "Correct 👍" if percent_change > NEUTRAL_THRESHOLD else "Incorrect 👎"
    elif prediction == "fall":
        return "Correct 👍" if percent_change < -NEUTRAL_THRESHOLD else "Incorrect 👎"
    elif prediction == "neutral":
        return "Correct 👍" if abs(percent_change) <= NEUTRAL_THRESHOLD else "Incorrect 👎"
    return "N/A"


def check_pending_predictions():
    """Checks all pending predictions and updates score/log."""
    all_predictions = load_all_predictions()
    if not all_predictions:
        print(f"No predictions found in {PREDICTION_FILE}.", file=sys.stderr)
        return

    score_data = load_reliability_score() 
    predictions_checked = 0
    predictions_updated = False

    checked_summary = []
    skipped_summary = []
    error_summary = []
    stdout_summary = []

    for prediction in all_predictions:
        if prediction.get("status") != "pending":
            continue

        company = prediction.get('company', 'Unknown') 
        saved_prediction = prediction.get('prediction')
        
        # --- MODIFIED: This is now the "prediction for" date ---
        date_saved = prediction.get('date_saved') 
        # --- END MODIFICATION ---

        if not all([company, saved_prediction, date_saved]):
             print(f"\nSkipping invalid prediction entry: {prediction}", file=sys.stderr)
             prediction['status'] = 'error'
             prediction['error_message'] = 'Missing company, prediction, or date_saved'
             error_summary.append(f"{company or 'Unknown'} ({date_saved or 'N/A'}): Invalid entry")
             predictions_updated = True
             continue

        # --- MODIFIED: Validation logic ---
        # Check if prediction_date is today or in the future
        try:
            date_saved_dt = datetime.strptime(date_saved, '%Y-%m-%d').date()
            if date_saved_dt >= date.today(): 
                 skipped_summary.append(f"{company} ({date_saved}): Prediction date is today or in the future, check tomorrow")
                 continue
        except ValueError:
             print(f"\nSkipping prediction for '{company}' due to invalid date_saved: {date_saved}", file=sys.stderr)
             prediction['status'] = 'error'
             prediction['error_message'] = f"Invalid date format: {date_saved}"
             error_summary.append(f"{company} ({date_saved}): Invalid date")
             predictions_updated = True
             continue
        # --- END MODIFICATION ---

        print(f"\nChecking pending prediction for '{company}' from {date_saved}...", file=sys.stderr) 
        print(f"  Your prediction was: {saved_prediction}", file=sys.stderr)

        if company not in TICKER_MAP:
            print(f"  Error: Company '{company}' not in TICKER_MAP. Skipping.", file=sys.stderr)
            prediction['status'] = 'error'
            prediction['error_message'] = 'Company not in TICKER_MAP'
            error_summary.append(f"{company} ({date_saved}): Not in TICKER_MAP")
            predictions_updated = True
            continue
        ticker = TICKER_MAP[company]

        # --- MODIFIED: Call new function ---
        percent_change, base_date, check_date = get_actual_stock_movement(ticker, date_saved)
        # --- END MODIFICATION ---
        
        if percent_change == "error":
            print(f"  Could not fetch data for {ticker}. Marking as error.", file=sys.stderr)
            prediction['status'] = 'error'
            prediction['error_message'] = 'yfinance data fetch error'
            error_summary.append(f"{company} ({date_saved}): Data fetch error")
            predictions_updated = True
            continue 

        if percent_change is None:
            # This means not enough data was available (e.g., trying to check too soon)
            skipped_summary.append(f"{company} ({date_saved}): Not enough data yet")
            continue 

        predictions_checked += 1
        predictions_updated = True
        
        # --- MODIFIED: Improved logging ---
        print(f"  Base Price (Close {base_date}): Found.", file=sys.stderr)
        print(f"  Check Price (Close {check_date}): Found.", file=sys.stderr)
        print(f"  Actual stock movement: {percent_change:+.2f}%", file=sys.stderr)
        # --- END MODIFICATION ---

        prediction_score = calculate_scaled_score(saved_prediction, percent_change)
        print(f"  --- Prediction Score: {prediction_score:.2f} / 1.0 ---", file=sys.stderr)

        correctness = get_correctness(saved_prediction, percent_change)
        
        # --- MODIFIED: Updated STDOUT summary ---
        stdout_summary.append(f"• Company: {company.capitalize()}")
        stdout_summary.append(f"• Prediction (for {date_saved}): {saved_prediction.capitalize()}")
        stdout_summary.append(f"• Actual Movement ({base_date} to {check_date}): {percent_change:+.2f}%")
        stdout_summary.append(f"• Result: {correctness}")
        stdout_summary.append("") 
        # --- END MODIFICATION ---

        # 5. Update score data
        score_data["global"]["total_predictions"] += 1
        score_data["global"]["total_score_points"] += prediction_score
        
        if company not in score_data["companies"]:
            score_data["companies"][company] = {"total_predictions": 0, "total_score_points": 0.0}
            print(f"  Initializing score record for {company}.", file=sys.stderr)
        score_data["companies"][company]["total_predictions"] += 1
        score_data["companies"][company]["total_score_points"] += prediction_score
        

        # 6. Update prediction entry
        prediction["status"] = "checked"
        prediction["date_checked"] = check_date # Save the date of the *check price*
        prediction["actual_movement_percent"] = round(percent_change, 2)
        prediction["prediction_score"] = round(prediction_score, 4)
        checked_summary.append(f"{company} ({date_saved}): Score {prediction_score:.2f}")


    # 7. Save results and print summary
    summary_output = []
    if predictions_updated:
        save_all_predictions(all_predictions)
        save_reliability_score(score_data) 

        reliability = 0.0
        if score_data["global"]["total_predictions"] > 0:
            reliability = score_data["global"]["total_score_points"] / score_data["global"]["total_predictions"]

        summary_output.append(f"\nChecked {predictions_checked} new prediction(s).")
        summary_output.append("--- Global Reliability Score Updated ---")
        summary_output.append(f"Total Points: {score_data['global']['total_score_points']:.2f} / Total Predictions: {score_data['global']['total_predictions']}")
        summary_output.append(f"New Global Reliability Score: {reliability:.3f} (or {reliability*100:.1f}%)")
        summary_output.append(f"Updated '{PREDICTION_FILE}' and '{SCORE_FILE}'.")
    else:
        summary_output.append("\nNo pending predictions were ready to be checked.")

    if skipped_summary:
        summary_output.append("\nSkipped (will retry later):")
        summary_output.extend([f"- {s}" for s in skipped_summary])
    if error_summary:
         summary_output.append("\nErrors encountered (marked as 'error' in log):")
         summary_output.extend([f"- {e}" for e in error_summary])
    
    if stdout_summary:
        print("\n".join(stdout_summary).strip())
    elif not predictions_updated:
         print("No pending predictions were ready to be checked.")
    else:
         print("Predictions checked, but none matched output criteria.")

    print("\n".join(summary_output), file=sys.stderr) 


def print_usage():
    """Prints the help message."""
    # (This function remains the same)
    print("Usage: python verifier.py <command>")
    print("\nCommands:")
    print("  check")
    print("    (Checks all pending predictions from 'all_predictions.json')")
    print("\n  score")
    print("    (Shows the *global* reliability score from 'reliability_score.json')")

if __name__ == "__main__":
    # (This function remains the same)
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "check":
        check_pending_predictions()

    elif command == "score":
        score_data = load_reliability_score()
        global_score = score_data["global"]
        
        if global_score["total_predictions"] == 0:
            print("No predictions made yet. Global Score is 0.0/0.")
        else:
            reliability = global_score["total_score_points"] / global_score["total_predictions"]
            print(f"Current Global Reliability: {global_score['total_score_points']:.2f} Points / {global_score['total_predictions']} Predictions ({reliability*100:.1f}%)")
        
        if score_data["companies"]:
            print("\n--- Per-Company Scores ---")
            for company, data in score_data["companies"].items():
                 if data["total_predictions"] > 0:
                    comp_reliability = data["total_score_points"] / data["total_predictions"]
                    print(f"  {company.capitalize()}: {comp_reliability*100:.1f}% ({data['total_predictions']} pred.)")
                 else:
                    print(f"  {company.capitalize()}: 0.0% (0 pred.)")

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print_usage()
        sys.exit(1)
import os
import sys
import traceback
import json
from datetime import date
import time

# --- Import Transformers Pipeline ---
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MODEL_MAX_LENGTH = 512
# ----------------------------------------------

# --- Financial Disclaimer ---
DISCLAIMER = """
*** NOT FINANCIAL ADVICE ***
This prediction is based *only* on the sentiment of public social
media comments. It does not account for company fundamentals, market
conditions, or any other financial data.
Do not use this to make investment decisions.
"""
# -----------------------------

# --- Load the local sentiment analysis pipeline ---
# (This section remains the same)
try:
    print(f"Loading local sentiment analysis model: {MODEL_NAME}...", file=sys.stderr)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=0, # Use 0 for GPU, remove/comment for CPU
        truncation=True
    )
    if sentiment_pipeline.device.type != 'cuda':
         print("Warning: Pipeline initialized, but NOT on CUDA device. Check PyTorch/CUDA setup.", file=sys.stderr)
         print(f"Current device: {sentiment_pipeline.device}", file=sys.stderr)
    else:
        print(f"Model loaded successfully onto GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)

except Exception as e:
    print(f"FATAL ERROR: Could not load the local model '{MODEL_NAME}'.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
# ----------------------------------------------------

# --- Helper Functions ---
def read_comments_from_file(filename):
    """Reads comments from a file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        print(f"Error: Comments file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            comments = [line.strip() for line in f if line.strip()]
        if not comments:
            print(f"Error: No comments found in {filepath}", file=sys.stderr)
            return None
        return comments
    except Exception as e:
        print(f"Error reading comments file {filepath}: {e}", file=sys.stderr)
        return None

def analyze_sentiments(comment_list):
    """Analyzes comments locally using the transformers pipeline."""
    analysis_results = []
    print(f"Starting sentiment analysis for {len(comment_list)} comments using device: {sentiment_pipeline.device}...", file=sys.stderr)
    try:
        batch_size = 16
        for i in range(0, len(comment_list), batch_size):
            batch = comment_list[i:i+batch_size]
            print(f"  Analyzing comments {i+1} to {min(i+batch_size, len(comment_list))}...", file=sys.stderr)

            results = sentiment_pipeline(batch, truncation=True, max_length=MODEL_MAX_LENGTH)

            for j, result in enumerate(results):
                comment_index = i + j
                original_comment = comment_list[comment_index]
                label = result.get('label', 'unknown').lower()
                score = result.get('score', 0.0)

                if label == 'positive': result_sentiment = 'positive'
                elif label == 'negative': result_sentiment = 'negative'
                elif label == 'neutral': result_sentiment = 'neutral'
                else:
                    print(f"Warning: Unexpected label '{label}' for comment: {original_comment[:50]}...", file=sys.stderr)
                    result_sentiment = 'other'

                analysis_results.append({
                    "comment": original_comment,
                    "sentiment": result_sentiment,
                    "score": round(score, 4)
                })

    except torch.cuda.OutOfMemoryError:
         print("\nCUDA Out of Memory!", file=sys.stderr)
         print(f"Try reducing the batch_size (currently {batch_size}).", file=sys.stderr)
         while len(analysis_results) < len(comment_list):
             analysis_results.append({"comment": comment_list[len(analysis_results)],"sentiment": "failed"})
    except Exception as e:
        print(f"\nFATAL ERROR during local sentiment analysis: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        while len(analysis_results) < len(comment_list):
             analysis_results.append({"comment": comment_list[len(analysis_results)],"sentiment": "failed"})

    print("Analysis finished.", file=sys.stderr)
    return analysis_results

def tally_results(analysis_results):
    """Counts sentiments."""
    scores = {'positive': 0, 'negative': 0, 'neutral': 0, 'other': 0, 'failed': 0}
    for item in analysis_results:
        sentiment = item['sentiment']
        if sentiment in scores:
            scores[sentiment] += 1
    return scores

def make_prediction(scores):
    """
    Calculates final score and returns prediction word, adjusted for Reddit bias and complexity.
    
    1. Uses asymmetric thresholds (Change 1).
    2. Applies a complexity buffer (Change 2).
    """
    
    # --- NEW CONFIGURATION ---
    # Asymmetric Thresholds (Change 1): Adjusted for Reddit's typical negative bias:
    POSITIVE_THRESHOLD = 0.15 
    NEGATIVE_THRESHOLD = -0.05 
    
    # Complexity Buffer (Change 2): Prevent 'fall' if negative count is only marginally higher.
    MIN_DIFF_FOR_FALL = 2 # Requires Negative - Positive >= 2 for the prediction to stand.
    # --------------------------

    P_raw = scores['positive']
    N_raw = scores['negative']
    E_raw = scores['neutral']
    
    total_analyzed = P_raw + N_raw + E_raw

    if total_analyzed == 0:
        print("\nNot enough data (0 valid comments) to make a prediction.", file=sys.stderr)
        return "neutral"
        
    # 1. Calculate sentiment score using raw counts
    sentiment_score = (P_raw - N_raw) / total_analyzed
    print(f"\nOverall Sentiment Score (from -1.0 to +1.0): {sentiment_score:.3f}", file=sys.stderr)
    
    # 2. Determine prediction based on adjusted (asymmetric) thresholds
    if sentiment_score > POSITIVE_THRESHOLD: 
        simple_prediction = "rise"
    elif sentiment_score < NEGATIVE_THRESHOLD: 
        simple_prediction = "fall"
    else: 
        simple_prediction = "neutral"

    # 3. Apply complexity buffer override (User's request: prevents marginal negative signal from causing a 'fall')
    is_marginal_fall = (N_raw > P_raw) and (N_raw - P_raw < MIN_DIFF_FOR_FALL)
    
    if simple_prediction == "fall" and is_marginal_fall:
        print(f"Overriding 'fall' to 'neutral' due to complexity buffer (Neg: {N_raw}, Pos: {P_raw}, Diff: {N_raw - P_raw} < {MIN_DIFF_FOR_FALL}).", file=sys.stderr)
        return "neutral"
        
    return simple_prediction

# --- Main part of the script ---
if __name__ == "__main__":
    # --- MODIFIED: Expect 2 arguments ---
    if len(sys.argv) < 3:
        print("Usage: python analyze_sentiments.py <comment_filename.txt> <prediction_date YYYY-MM-DD>", file=sys.stderr)
        sys.exit(1)

    comment_filename_arg = sys.argv[1]
    prediction_date_arg = sys.argv[2] # This is the date the prediction is FOR
    # --- END MODIFICATION ---

    comment_filepath = os.path.join(os.path.dirname(__file__), comment_filename_arg)

    try:
        comments = read_comments_from_file(comment_filepath)

        if comments:
            print(f"Read {len(comments)} comments. Starting local analysis...", file=sys.stderr)

            analysis_results = analyze_sentiments(comments)
            scores = tally_results(analysis_results)

            print("\n--- Sentiment Analysis Complete ---")
            print(f"Positive: {scores['positive']}")
            print(f"Negative: {scores['negative']}")
            print(f"Neutral: {scores['neutral']}")
            print("---------------------------------")
            print(f"Skipped/Other: {scores['other']}")
            print(f"Failed to analyze: {scores['failed']}")

            simple_prediction = make_prediction(scores)

            if simple_prediction == "rise":
                prediction_string = "Overall sentiment is POSITIVE. (Simplified Prediction: Rise 📈)"
            elif simple_prediction == "fall":
                prediction_string = "Overall sentiment is NEGATIVE. (Simplified Prediction: Fall 📉)"
            else:
                prediction_string = "Overall sentiment is NEUTRAL. (Simplified Prediction: No Clear Direction ➖)"

            print("\n--- Final Prediction ---")
            print(prediction_string)
            print(DISCLAIMER)

            # Save detailed JSON results
            try:
                base_name = os.path.splitext(os.path.basename(comment_filepath))[0]
                json_filename = f"{base_name}_analyzed.json"
                json_filepath = os.path.join(os.path.dirname(__file__), json_filename)
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=2, ensure_ascii=False)
                print(f"\nSuccessfully saved detailed analysis to: {json_filename}", file=sys.stderr)
            except IOError as e:
                print(f"\nError: Could not save JSON file. Details: {e}", file=sys.stderr)

            # Append prediction to master log file
            try:
                company_name = base_name.split('_')[0].lower() 
                
                # --- MODIFIED: Use prediction_date_arg ---
                new_prediction = {
                    "company": company_name,
                    "prediction": simple_prediction,
                    "date_saved": prediction_date_arg, # Use the date from argument
                    "status": "pending",
                    "method": "local_sentiment_model"
                }
                # --- END MODIFICATION ---
                
                log_filename = "all_predictions.json"
                log_filepath = os.path.join(os.path.dirname(__file__), log_filename)
                predictions_list = []
                if os.path.exists(log_filepath):
                    try:
                        with open(log_filepath, 'r', encoding='utf-8') as f:
                            predictions_list = json.load(f)
                        if not isinstance(predictions_list, list): predictions_list = []
                    except json.JSONDecodeError: predictions_list = []
                predictions_list.append(new_prediction)
                with open(log_filepath, 'w', encoding='utf-8') as f:
                    json.dump(predictions_list, f, indent=2)
                print(f"Successfully appended prediction to: {log_filename}", file=sys.stderr)
            except Exception as e:
                print(f"\nError: Could not save prediction log file. Details: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nScript cancelled by user.", file=sys.stderr)
    except Exception as e:
        print("\nAn unexpected fatal error occurred in analyze_sentiments:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
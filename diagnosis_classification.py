"""diagnosis_classification.py
Uses an OpenAI model to classify each participant in a test set
    as healthy or mild cognitive impairment. Writes results to files.

@author Gina Sprint
@date 5/22/24
"""
import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

from agent_tools import diagnosis_tool


RESULTS_DIRNAME = "results"

# PERFORMANCE METRIC CALCULATION
def compute_classification_metrics(actual_ser, pred_ser):
    print(classification_report(actual_ser, pred_ser))
    report_dict = classification_report(actual_ser, pred_ser, output_dict=True)
    metrics_dict = {
        "accuracy": accuracy_score(actual_ser, pred_ser),
        "mcc": matthews_corrcoef(actual_ser, pred_ser)
    }
    for label in report_dict.keys():
        if isinstance(report_dict[label], dict):
            for metric in report_dict[label].keys():
                metrics_dict[label + "_" + metric] = report_dict[label][metric]
        else:
            metrics_dict[label] = report_dict[label]
    metrics_ser = pd.Series(metrics_dict)
    return metrics_ser

def compute_results(actual_ser, pred_ser):
    # make sure in the same order
    assert list(actual_ser.index) == list(pred_ser.index)
    result_ser = compute_classification_metrics(actual_ser, pred_ser)
    return result_ser

# CALL TO DIAGNOSIS TOOL FUNCTION AND CALCULATE RESULTS
def run_best_wrapper_rules_experiment(exp_label, explain):
    name_prompt_response_df, pred_ser, actual_ser = diagnosis_tool.run_test_participants(explain)
    name_prompt_fname = os.path.join(RESULTS_DIRNAME, f"{exp_label}_preds.csv")
    name_prompt_response_df.to_csv(name_prompt_fname)

    results_ser = compute_results(actual_ser, pred_ser)
    results_ser.name = exp_label
    fname = os.path.join(RESULTS_DIRNAME, f"{exp_label}_results.csv")
    results_ser.to_csv(fname)
    return results_ser

def main(exp_label, num_runs):
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]
    if not os.path.exists(RESULTS_DIRNAME):
        os.mkdir(RESULTS_DIRNAME)
    np.random.seed(0)

    all_result_sers = {}
    for i in range(num_runs):
        print(f"Run #: {i + 1}/{num_runs}")
        run_exp_label = f"{exp_label}_run#{i}"
        results_ser = run_best_wrapper_rules_experiment(exp_label, explain=True)
        all_result_sers[run_exp_label] = results_ser
    all_results_df = pd.DataFrame(all_result_sers).T
    fname = os.path.join(RESULTS_DIRNAME, f"{exp_label}_multiple_run_results.csv")
    all_results_df.to_csv(fname)

if __name__ == "__main__":
    # example run:
    # python diagnosis_classification.py -e example -n 1
    parser = argparse.ArgumentParser(
        description="Run diagnosis classification over test set and report the results."
    )
    parser.add_argument("-e",
                        type=str,
                        dest="exp_label",
                        default="wrapper_best",
                        help="The experiment label used in output files names.")
    parser.add_argument("-n",
                        type=int,
                        dest="num_runs",
                        default=30,
                        help="The number of experiment runs to perform.")
    
    args = parser.parse_args()
    main(args.exp_label, args.num_runs)

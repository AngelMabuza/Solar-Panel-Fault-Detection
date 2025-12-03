import sys, subprocess, json
from datetime import datetime
from typing import Dict, List, Any, Union, Set
import math


def run_training_and_eval(model_name, csv_path, img, batch, epochs, runs_root, base_env):
    """Train + eval one classifier; return (status, run_dir, metrics_dict or None, stderr_tail)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / f"{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Train
    train_cmd = [
                sys.executable, "-m", "src.cls.train_cls",
                "--csv",   csv_path,
                "--model_name", model_name,
                "--out",   str(run_dir),
                "--epochs", str(epochs),
                "--batch",  str(batch),
                "--img",    str(img),
                ]
    print(f"\n=== Training {model_name} ===\n$ {' '.join(train_cmd)}")
    train_proc = subprocess.run(train_cmd, capture_output=True, text=True, env=base_env)
    print("---- TRAIN STDOUT (tail) ----\n", train_proc.stdout[-1500:])
    print("---- TRAIN STDERR (tail) ----\n", train_proc.stderr[-1500:])
    if train_proc.returncode != 0:
        return ("TRAIN_FAIL", str(run_dir), None, train_proc.stderr[-2000:])

    # 2) Eval
    eval_dir = run_dir / "eval"
    eval_cmd = [
        sys.executable, "-m", "src.cls.eval_cls",
        "--csv",   csv_path,
        "--model", str(run_dir / "best.keras"),
        "--out",   str(eval_dir),
        "--img",   str(img),
    ]
    print(f"\n=== Evaluating {model_name} ===\n$ {' '.join(eval_cmd)}")
    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, env=base_env)
    print("---- EVAL STDOUT (tail) ----\n", eval_proc.stdout[-1500:])
    print("---- EVAL STDERR (tail) ----\n", eval_proc.stderr[-1500:])
    if eval_proc.returncode != 0:
        return ("EVAL_FAIL", str(run_dir), None, eval_proc.stderr[-2000:])

    # 3) Read metrics JSON if present
    metrics_path = eval_dir / "test_metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print("Could not read test_metrics.json:", e)

    return ("OK", str(run_dir), metrics, eval_proc.stderr[-1000:])


def load_metrics_data(file_path: str) -> Dict[str, Any]:
    """
    Load metrics data from a JSON file.

    Args:
        file_path: Path to the JSON file containing metrics data

    Returns:
        Dictionary containing the loaded metrics data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_class_keys(data: Dict[str, Any], special_keys: Set[str]) -> List[str]:
    """
    Extract class keys from metrics data by excluding special keys.

    Args:
        data: Dictionary containing metrics data
        special_keys: Set of keys to exclude from class keys

    Returns:
        List of class keys
    """
    return [k for k in data.keys() if k not in special_keys]


def extract_value(data_dict: Dict[str, Any], *field_names: str, default: float = math.nan) -> float:
    """
    Extract a value from a dictionary using multiple possible field names.

    Args:
        data_dict: Dictionary to extract value from
        field_names: Possible field names to look for
        default: Default value to return if no field name is found

    Returns:
        Extracted value as float or default value
    """
    for name in field_names:
        if name in data_dict:
            return float(data_dict[name])
    return default


def build_summary(
    data: Dict[str, Any],
    macro_avg: Dict[str, Any],
    weighted_avg: Dict[str, Any],
    class_keys: List[str]
) -> Dict[str, Union[float, int]]:
    """
    Build a summary of metrics from the provided data.

    Args:
        data: Dictionary containing metrics data
        macro_avg: Dictionary containing macro average metrics
        weighted_avg: Dictionary containing weighted average metrics
        class_keys: List of class keys

    Returns:
        Dictionary containing summarized metrics
    """
    return {
        "accuracy":            float(data.get("accuracy", math.nan)),
        "macro_precision":     extract_value(macro_avg, "precision"),
        "macro_recall":        extract_value(macro_avg, "recall"),
        "macro_f1":            extract_value(macro_avg, "f1-score", "f1"),
        "weighted_precision":  extract_value(weighted_avg, "precision"),
        "weighted_recall":     extract_value(weighted_avg, "recall"),
        "weighted_f1":         extract_value(weighted_avg, "f1-score", "f1"),
        "balanced_accuracy":   extract_value(macro_avg, "recall"),  # alias for macro recall
        "num_classes":         len(class_keys),
        "test_support":        int(extract_value(weighted_avg, "support", default=0)),  # uses weighted.avg support
    }


def find_zero_recall_classes(data: Dict[str, Any], class_keys: List[str]) -> List[str]:
    """
    Find classes with zero recall (never correctly predicted).

    Args:
        data: Dictionary containing metrics data
        class_keys: List of class keys

    Returns:
        List of class keys with zero recall
    """
    return [k for k in class_keys if extract_value(data[k], "recall", default=0.0) == 0.0]


def save_summary(summary: Dict[str, Any], output_path: str) -> None:
    """
    Save summary metrics to a JSON file.

    Args:
        summary: Dictionary containing summary metrics
        output_path: Path to save the JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print summary metrics in a formatted way.

    Args:
        summary: Dictionary containing summary metrics
    """
    print("Overall test metrics:")
    for k, v in summary.items():
        print(f"{k:>20}: {v:.6f}" if isinstance(v, float) else f"{k:>20}: {v}")


def main(PER_CLASS_JSON, OUT_JSON) -> None:
    """
    Main function to process metrics data, build summary, and save results.
    """
    # Load data
    data = load_metrics_data(PER_CLASS_JSON)

    # Define special keys and extract class keys
    special_keys = {"accuracy", "macro avg", "weighted avg"}
    class_keys = extract_class_keys(data, special_keys)

    # Extract average metrics
    macro_avg = data.get("macro avg", {})      # {precision, recall, f1-score, support}
    weighted_avg = data.get("weighted avg", {})   # {precision, recall, f1-score, support}

    # Build summary
    summary = build_summary(data, macro_avg, weighted_avg, class_keys)

    # Print summary
    print_summary(summary)

    # Find and print classes with zero recall
    zero_recall = find_zero_recall_classes(data, class_keys)
    if zero_recall:
        print("\nClasses with zero recall (no correct predictions):", zero_recall)

    # Save summary
    save_summary(summary, OUT_JSON)
    print("\nWrote overall metrics ->", OUT_JSON)

import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import os

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_targets, all_preds = [], []
    dish_ids = []

    for batch in dataloader:
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        masses = batch["mass"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        if "dish_id" in batch:
            dish_ids.extend(batch["dish_id"])

        batch_size = targets.size(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images, mass=masses)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        total_mae  += F.l1_loss(outputs, targets, reduction="sum").item()
        total_samples += batch_size

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(outputs.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_mae  = total_mae / total_samples if total_samples > 0 else 0
    r2 = r2_score(all_targets, all_preds)

    results = {
        "loss": avg_loss,
        "mae": avg_mae,
        "r2": r2,
        "targets": np.array(all_targets),
        "predictions": np.array(all_preds),
        "dish_ids": dish_ids if dish_ids else list(range(len(all_targets)))
    }

    return results

def get_top_worst_predictions(results, top_k=5):
    targets = results["targets"]
    preds = results["predictions"]
    dish_ids = results["dish_ids"]

    abs_errors = np.abs(targets - preds)

    indices = np.argsort(abs_errors)[::-1][:top_k]

    worst_cases = []
    for idx in indices:
        worst_cases.append({
            "dish_id": dish_ids[idx] if idx < len(dish_ids) else idx,
            "true_calories": float(targets[idx]),
            "predicted_calories": float(preds[idx]),
            "abs_error": float(abs_errors[idx]),
            "error_percent": float(abs_errors[idx] / (targets[idx] + 1e-8) * 100)
        })

    return worst_cases

def analyze_worst_cases(worst_cases):
    print("\n" + "="*70)
    print("TOP 5 WORST PREDICTIONS ANALYSIS")
    print("="*70)

    for i, case in enumerate(worst_cases, 1):
        print(f"\n#{i} | Dish ID: {case['dish_id']}")
        print(f"    True Calories:      {case['true_calories']:.0f}")
        print(f"    Predicted Calories: {case['predicted_calories']:.0f}")
        print(f"    Absolute Error:      {case['abs_error']:.0f}")
        print(f"    Error %:             {case['error_percent']:.1f}%")

    print("\n" + "="*70)
    print("POSSIBLE CAUSES OF POOR PREDICTIONS:")
    print("="*70)
    print("""
1. HIGH-CALORIE DISHES: Large errors often occur on high-calorie dishes where
   small portion variations lead to large absolute errors.

2. COMPLEX RECIPES: Dishes with many ingredients are harder to predict accurately
   due to cumulative estimation errors.

3. VISUAL AMBIGUITY: Similar-looking dishes can have vastly different calorie
   content (e.g., fried vs baked, cream-based vs broth-based).

4. MISSING INGREDIENTS: Some ingredients may be listed but not visible or present
   in the photo (sauces, oils used in cooking).

5. OUTLIERS: Data collection errors or genuinely unusual dishes that don't
   follow typical calorie patterns.
""")
    return worst_cases

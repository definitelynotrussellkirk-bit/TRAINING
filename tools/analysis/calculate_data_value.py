#!/usr/bin/env python3
"""
Synthetic Data Value Calculator

Quick tool to estimate the economic value of your training data.
Run after training completes to get real ROI numbers.
"""

import json
from pathlib import Path
from datetime import datetime

def calculate_data_value():
    """Calculate the value of synthetic training data."""

    print("=" * 80)
    print("SYNTHETIC DATA VALUE CALCULATOR")
    print("=" * 80)
    print()

    # Get basic info
    print("📊 Dataset Information:")
    dataset_file = input("  Dataset filename: ").strip() or "syllo_training_contract_20k.jsonl"
    num_examples = int(input("  Number of examples: ").strip() or "20000")

    print("\n💰 Cost Information:")
    generation_hours = float(input("  Generation time (hours): ").strip() or "2")
    generation_cost_per_hour = float(input("  GPU cost per hour ($): ").strip() or "0.50")
    your_hourly_rate = float(input("  Your hourly rate ($): ").strip() or "100")

    print("\n📈 Performance Information:")
    baseline_accuracy = float(input("  Baseline accuracy (%): ").strip() or "45") / 100
    trained_accuracy = float(input("  Trained accuracy (%): ").strip() or "85") / 100

    print("\n💼 Business Value:")
    value_per_percent = float(input("  Value per % point improvement ($): ").strip() or "1000")

    print("\n" + "=" * 80)
    print("CALCULATIONS")
    print("=" * 80)

    # Calculate costs
    generation_cost = generation_hours * generation_cost_per_hour
    labor_cost = generation_hours * your_hourly_rate
    training_cost = 4 * 0.50  # 4 hours at $0.50/hour
    total_cost = generation_cost + labor_cost + training_cost

    print(f"\n1. PRODUCTION COSTS:")
    print(f"   Generation:      ${generation_cost:,.2f}")
    print(f"   Your time:       ${labor_cost:,.2f}")
    print(f"   Training:        ${training_cost:,.2f}")
    print(f"   ─────────────────────────")
    print(f"   TOTAL COST:      ${total_cost:,.2f}")
    print(f"   Cost per example: ${total_cost/num_examples:.4f}")

    # Calculate market value
    market_rate_low = 1.00
    market_rate_high = 3.00
    market_value_low = num_examples * market_rate_low
    market_value_high = num_examples * market_rate_high
    market_value_mid = (market_value_low + market_value_high) / 2

    print(f"\n2. MARKET VALUE:")
    print(f"   Market rate:     ${market_rate_low} - ${market_rate_high} per example")
    print(f"   Low estimate:    ${market_value_low:,.2f}")
    print(f"   High estimate:   ${market_value_high:,.2f}")
    print(f"   MID ESTIMATE:    ${market_value_mid:,.2f}")

    # Calculate value created
    improvement = (trained_accuracy - baseline_accuracy) * 100  # percentage points
    value_created = improvement * value_per_percent

    print(f"\n3. VALUE CREATED:")
    print(f"   Baseline accuracy:  {baseline_accuracy*100:.1f}%")
    print(f"   Trained accuracy:   {trained_accuracy*100:.1f}%")
    print(f"   Improvement:        {improvement:.1f} percentage points")
    print(f"   Value per % point:  ${value_per_percent:,.2f}")
    print(f"   VALUE CREATED:      ${value_created:,.2f}")

    # Calculate replacement cost
    hours_to_label = num_examples * 0.02  # ~1 minute per example
    labeling_cost = hours_to_label * 50  # $50/hour for human labeler
    replacement_cost = labeling_cost

    print(f"\n4. REPLACEMENT COST:")
    print(f"   Manual labeling time: {hours_to_label:,.0f} hours")
    print(f"   Labeling rate:        $50/hour")
    print(f"   REPLACEMENT COST:     ${replacement_cost:,.2f}")

    # Calculate time saved
    time_saved_value = replacement_cost - total_cost

    print(f"\n5. TIME/MONEY SAVED:")
    print(f"   Alternative cost:     ${replacement_cost:,.2f}")
    print(f"   Your cost:            ${total_cost:,.2f}")
    print(f"   SAVINGS:              ${time_saved_value:,.2f}")

    # Calculate ROI
    roi_market = ((market_value_mid - total_cost) / total_cost) * 100
    roi_value_created = ((value_created - total_cost) / total_cost) * 100
    roi_time_saved = ((time_saved_value) / total_cost) * 100

    print(f"\n6. RETURN ON INVESTMENT:")
    print(f"   Market-based ROI:     {roi_market:,.0f}%")
    print(f"   Value-created ROI:    {roi_value_created:,.0f}%")
    print(f"   Time-saved ROI:       {roi_time_saved:,.0f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {dataset_file}")
    print(f"Examples: {num_examples:,}")
    print()
    print(f"TOTAL COST:           ${total_cost:,.2f}")
    print(f"MARKET VALUE:         ${market_value_mid:,.2f}")
    print(f"VALUE CREATED:        ${value_created:,.2f}")
    print(f"SAVINGS vs MANUAL:    ${time_saved_value:,.2f}")
    print()
    print(f"RECOMMENDED PRICING:")
    print(f"  Per example:        ${market_value_mid/num_examples:.2f}")
    print(f"  1k examples:        ${(market_value_mid/num_examples)*1000:,.2f}")
    print(f"  10k examples:       ${(market_value_mid/num_examples)*10000*0.8:,.2f} (20% discount)")
    print(f"  Full dataset (20k): ${market_value_mid:,.2f}")
    print()
    print(f"EFFICIENCY METRICS:")
    print(f"  Cost per example:   ${total_cost/num_examples:.4f}")
    print(f"  vs Human labeling:  ${replacement_cost/num_examples:.4f}")
    print(f"  Efficiency gain:    {(replacement_cost/num_examples)/(total_cost/num_examples):.0f}x cheaper")
    print()
    print(f"ROI: {roi_market:,.0f}% (market-based) or {roi_value_created:,.0f}% (value-created)")
    print()
    print("=" * 80)

    # Save report
    report = {
        "date": datetime.now().isoformat(),
        "dataset": dataset_file,
        "num_examples": num_examples,
        "costs": {
            "generation": generation_cost,
            "labor": labor_cost,
            "training": training_cost,
            "total": total_cost,
            "per_example": total_cost / num_examples
        },
        "market_value": {
            "low": market_value_low,
            "mid": market_value_mid,
            "high": market_value_high,
            "per_example": market_value_mid / num_examples
        },
        "performance": {
            "baseline_accuracy": baseline_accuracy,
            "trained_accuracy": trained_accuracy,
            "improvement_pct": improvement,
            "value_created": value_created
        },
        "replacement_cost": replacement_cost,
        "time_saved": time_saved_value,
        "roi": {
            "market_based": roi_market,
            "value_created": roi_value_created,
            "time_saved": roi_time_saved
        }
    }

    report_file = f"reports/data_value_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("reports").mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Report saved to: {report_file}")
    print()

if __name__ == "__main__":
    calculate_data_value()

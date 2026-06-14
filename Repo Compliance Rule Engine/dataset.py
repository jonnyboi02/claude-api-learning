import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
MODEL = "claude-haiku-4-5"

GENERATOR_PROMPT = """You are generating synthetic test cases for evaluating an AI compliance pre-screening system for repo trading at an asset manager.

<system_being_tested>
The system receives a rule configuration, a current repo book, and a proposed new trade. It must determine whether adding the proposed trade breaches any parent rules, and whether PM override is required.

Rules are hierarchical:
- Parent rules contain child rules
- Each parent rule combines its children with AND or OR logic
- OR: parent breached if ANY child breached
- AND: parent breached only if ALL children breached
- PM override required if ANY parent rule is breached

Rule configurations vary per test case because they come from different client mandates.

All evaluations are net and post-trade: take the current book, apply the proposed trade, then check the resulting net position against each rule.
</system_being_tested>

<sign_convention>
- Positive notional = reverse repo (cash out to counterparty, collateral in)
- Negative notional = repo (cash in from counterparty, collateral out)
- Positions in the same maturity window offset each other in net calculations
</sign_convention>

<rule_dimensions>
Child rules aggregate on one of two dimensions:
maturity_month - aggregate net notional across all positions maturing in a given month, check against a numeric range (lower_bound, upper_bound)
counterparty - aggregate net notional across all positions with a given counterparty, check against a numeric range (lower_bound, upper_bound)
</rule_dimensions>

<task>
Generate diverse test cases. Each test case contains:
1. A rule configuration (one or more parent rules with children and AND/OR logic)
2. A current repo book (3-8 positions with signed notionals)
3. A proposed new trade (signed notional)
4. The expected output after applying the rules

Vary scenarios to cover:
- Clean pass (no parent rules breached)
- OR parent breached (at least one child triggers)
- AND parent breached (all children trigger)
- AND parent NOT breached despite partial child breach (critical edge case)
- Multiple parent rules breached simultaneously
- Net calculation reduces exposure (proposed trade offsets existing position)
- Net calculation increases exposure beyond limit
- Edge cases at exactly the threshold
</task>

<output_format>
Return a JSON array. Each element:
{
  "scenario_label": "short description of what this case tests",
  "task": {
    "rules": [
      {
        "parent_rule_id": "MATURITY_CONCENTRATION" or "COUNTERPARTY_EXPOSURE",
        "logic": "AND" or "OR",
        "children": [
          {
            "id": "string identifier",
            "description": "human readable rule",
            "dimension": "maturity_month" or "counterparty",
            "filter": "YYYY-MM for maturity_month, counterparty name for counterparty",
            "lower_bound": integer (GBP, can be negative),
            "upper_bound": integer (GBP, can be negative)
          }
        ]
      }
    ],
    "current_book": [
      {"counterparty": "string", "notional": integer (signed), "maturity": "YYYY-MM-DD"}
    ],
    "proposed_trade": {
      "counterparty": "string", "notional": integer (signed), "maturity": "YYYY-MM-DD"
    }
  },
  "expected": {
    "parent_results": [
      {
        "parent_rule_id": "string",
        "logic": "AND" or "OR",
        "breached": boolean,
        "child_results": [
          {
            "id": "string",
            "breached": boolean,
            "net_value": integer (calculated net for that dimension/filter post-trade),
            "lower_bound": integer,
            "upper_bound": integer
          }
        ],
        "reasoning": "explanation referencing specific numbers"
      }
    ],
    "pm_override_required": boolean,
    "verdict": "string describing overall outcome"
  }
}
</output_format>

<constraints>
- All net calculations must be mathematically correct based on the book + proposed trade
- A child rule breaches when net_value falls OUTSIDE the [lower_bound, upper_bound] range
- Counterparty names from: Barclays, JPM, Goldman Sachs, Morgan Stanley, Citi, HSBC, BNP Paribas, Deutsche Bank
- Maturity dates between 2026-06-15 and 2026-12-31
- Notionals between -800000000 and +800000000 (signed)
- pm_override_required is true if and only if any parent_results entry has breached=true
- For AND logic scenarios, deliberately include cases where some but not all children breach
- Reasoning must reference specific calculated values and limits
</constraints>

Generate 20 test cases covering the full range of scenarios.
Distribution:
- 3 clean pass
- 3 OR parent breached
- 3 AND parent breached
- 3 AND parent NOT breached despite partial child breach
- 3 multiple parents breached simultaneously
- 3 trade reduces net exposure (offset scenarios)
- 2 edge cases at exactly the threshold

Vary the rule configurations - different parents, different children, different AND/OR logic, different bounds across cases."""


DISTRIBUTION_SECTION = """Generate 20 test cases covering the full range of scenarios.
Distribution:
- 3 clean pass
- 3 OR parent breached
- 3 AND parent breached
- 3 AND parent NOT breached despite partial child breach
- 3 multiple parents breached simultaneously
- 3 trade reduces net exposure (offset scenarios)
- 2 edge cases at exactly the threshold"""


def _call_generator(num_cases: int) -> list[dict]:
    prompt = GENERATOR_PROMPT.replace(
        DISTRIBUTION_SECTION,
        f"Generate {num_cases} diverse test cases. Vary the scenario types across: "
        "clean pass, OR parent breached, AND parent breached, AND parent NOT breached "
        "(partial child breach only), net exposure reduced by trade, and edge cases at threshold.",
    )
    text = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

    # Extract the outermost JSON array, ignoring any markdown fences or
    # commentary the model may add before/after it.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in model output:\n{text[:500]}")

    return json.loads(text[start : end + 1])


def generate_dataset(output_file: str = "dataset.json", batch_size: int = 3) -> list[dict]:
    """Generate test cases in batches to avoid hitting output token limits."""
    print("Generating dataset...")

    dataset = []
    batch = 1
    while len(dataset) < 20:
        remaining = 20 - len(dataset)
        n = min(batch_size, remaining)
        print(f"  Batch {batch}: requesting {n} cases...")
        cases = _call_generator(n)
        dataset.extend(cases)
        batch += 1

    print(f"Generated {len(dataset)} test cases.")

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved to {output_file}")
    return dataset


if __name__ == "__main__":
    generate_dataset()

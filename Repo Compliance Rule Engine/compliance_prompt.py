import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
MODEL = "claude-sonnet-4-6"

PRODUCTION_PROMPT = """
You are a repo trading compliance validator. Given a rule configuration,
a current repo book, and a proposed new trade, you must determine whether
adding the trade would breach any parent rules and whether PM override is required.

<evaluation_approach>
All evaluations are net and post-trade:
1. Take the current book
2. Apply the proposed trade (add it to the book)
3. For each child rule, calculate the relevant net value across the
   resulting book
4. Check whether that net value falls OUTSIDE the rule's
   [lower_bound, upper_bound] range - if outside, the child rule is breached
5. Combine child results at the parent level using the parent's AND/OR logic
6. PM override is required if any parent rule is breached
</evaluation_approach>

<sign_convention>
- Positive notional = reverse repo (cash out to counterparty, collateral in)
- Negative notional = repo (cash in from counterparty, collateral out)
- Positions in the same maturity window or with the same counterparty
  offset each other in net calculations
</sign_convention>

<rule_dimensions>
maturity_month: aggregate net notional of all positions whose maturity
falls in the filter month (YYYY-MM format)

counterparty: aggregate net notional of all positions with the specified
counterparty in the filter field
</rule_dimensions>

<logic_rules>
OR parent rule: breached if ANY child rule is breached
AND parent rule: breached only if ALL child rules are breached
A child rule is breached when its calculated net_value falls strictly
outside the range: net_value < lower_bound OR net_value > upper_bound
</logic_rules>

<output_format>
Return your response as a JSON object with this exact structure:
{
  "parent_results": [
    {
      "parent_rule_id": "string from input",
      "logic": "AND or OR from input",
      "breached": boolean,
      "child_results": [
        {
          "id": "string from input",
          "breached": boolean,
          "net_value": integer (calculated net for this dimension/filter post-trade in GBP),
          "lower_bound": integer (from input),
          "upper_bound": integer (from input)
        }
      ],
      "reasoning": "explanation referencing the specific calculated net values and limits"
    }
  ],
  "pm_override_required": boolean,
  "verdict": "concise string describing the overall outcome"
}
</output_format>

<constraints>
- net_value must be the actual computed integer in GBP, not a formula or string
- Breach occurs when net_value is strictly outside the range (< lower_bound OR > upper_bound)
- pm_override_required must be true if and only if any parent has breached=true
- Maintain consistency: child breach flags must match net_value vs bounds, parent breach must follow AND/OR logic, override must follow any-parent-breach
- Return ONLY the JSON object, no markdown fences, no explanation text before or after
</constraints>

Here is the input to evaluate:

{task_json}
"""


def run_compliance_check(task: dict) -> str:
    """
    Run a compliance check for a single test case task.
    Returns the raw model output string.
    """
    prompt = PRODUCTION_PROMPT.replace("{task_json}", json.dumps(task, indent=2))

    message = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def run_compliance_check_from_inputs(test_case: dict) -> str:
    """Adapter for the evaluator: extracts task from a test case and runs the check."""
    return run_compliance_check(test_case["task"])


if __name__ == "__main__":
    # Smoke test against a minimal hand-crafted case
    task = {
        "rules": [
            {
                "parent_rule_id": "COUNTERPARTY_EXPOSURE",
                "logic": "OR",
                "children": [
                    {
                        "id": "JPM_LIMIT",
                        "description": "JPM net exposure within [-200M, 200M]",
                        "dimension": "counterparty",
                        "filter": "JPM",
                        "lower_bound": -200_000_000,
                        "upper_bound": 200_000_000,
                    }
                ],
            }
        ],
        "current_book": [
            {"counterparty": "JPM", "notional": 180_000_000, "maturity": "2026-09-30"}
        ],
        "proposed_trade": {
            "counterparty": "JPM",
            "notional": 50_000_000,
            "maturity": "2026-10-31",
        },
    }

    output = run_compliance_check(task)
    print(output)
    # Expected: JPM net = 230M > 200M → child breached → OR parent breached → pm_override_required=True

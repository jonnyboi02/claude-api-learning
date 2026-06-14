import json
import concurrent.futures
from statistics import mean
from textwrap import dedent
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
MODEL = "claude-sonnet-4-6"


def _chat(messages, system=None, temperature=0.0, stop_sequences=None):
    params = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    return client.messages.create(**params).content[0].text


def _add_user(messages, text):
    messages.append({"role": "user", "content": text})


def _add_assistant(messages, text):
    messages.append({"role": "assistant", "content": text})


def _trade_summary(prompt_inputs: dict) -> str:
    try:
        trade = json.loads(prompt_inputs["proposed_trade"])
        book = json.loads(prompt_inputs["current_book"])
        rules_list = json.loads(prompt_inputs["rules"])
        rule_summary = ", ".join(
            f"{r['parent_rule_id']} ({r['logic']})" for r in rules_list
        )
        direction = "reverse repo" if trade["notional"] > 0 else "repo"
        return (
            f"<strong>Trade:</strong> {trade['counterparty']} "
            f"{trade['notional']:,} ({direction}) maturing {trade['maturity']}<br>"
            f"<strong>Book:</strong> {len(book)} positions<br>"
            f"<strong>Rules:</strong> {rule_summary}"
        )
    except Exception:
        return str(prompt_inputs)


def _generate_html_report(results: list) -> str:
    total = len(results)
    scores = [r["score"] for r in results]
    avg = mean(scores) if scores else 0
    pass_rate = 100 * len([s for s in scores if s >= 7]) / total if total else 0

    rows = ""
    for r in results:
        tc = r["test_case"]
        criteria_html = "<br>• ".join(tc["solution_criteria"])
        expected_pm = tc["expected"].get("pm_override_required", "?")
        score = r["score"]
        score_class = "score-high" if score >= 8 else ("score-low" if score <= 5 else "score-medium")

        rows += f"""
        <tr>
            <td>{tc['scenario']}</td>
            <td>{_trade_summary(tc['prompt_inputs'])}</td>
            <td>• {criteria_html}<br><em>Expected PM override: <strong>{expected_pm}</strong></em></td>
            <td class="output"><pre>{r['output']}</pre></td>
            <td class="score-col"><span class="score {score_class}">{score}</span></td>
            <td>{r['reasoning']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Repo Compliance Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
  .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
  .summary-stats {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .stat-box {{ background: #fff; border-radius: 5px; padding: 15px;
               box-shadow: 0 2px 5px rgba(0,0,0,.1); flex-basis: 30%; min-width: 180px; }}
  .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
  th {{ background: #4a4a4a; color: #fff; text-align: left; padding: 12px; }}
  td {{ padding: 10px; border-bottom: 1px solid #ddd; vertical-align: top; width: 18%; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  pre {{ background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 8px;
         font-size: 12px; white-space: pre-wrap; word-wrap: break-word; margin: 0; }}
  .score {{ font-weight: bold; padding: 5px 10px; border-radius: 3px; display: inline-block; }}
  .score-high {{ background: #c8e6c9; color: #2e7d32; }}
  .score-medium {{ background: #fff9c4; color: #f57f17; }}
  .score-low {{ background: #ffcdd2; color: #c62828; }}
  .score-col {{ width: 60px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Repo Compliance Engine — Evaluation Report</h1>
  <div class="summary-stats">
    <div class="stat-box"><div>Total Test Cases</div><div class="stat-value">{total}</div></div>
    <div class="stat-box"><div>Average Score</div><div class="stat-value">{avg:.1f} / 10</div></div>
    <div class="stat-box"><div>Pass Rate (≥7)</div><div class="stat-value">{pass_rate:.1f}%</div></div>
  </div>
</div>
<table>
  <thead>
    <tr>
      <th>Scenario</th><th>Trade &amp; Rules</th><th>Solution Criteria</th>
      <th>Engine Output</th><th>Score</th><th>Reasoning</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""


def grade_output(test_case: dict, output_str: str) -> dict:
    """Grade the compliance engine's output against the expected results."""
    expected = json.dumps(test_case["expected"], indent=2)
    criteria = "\n".join(f"- {c}" for c in test_case["solution_criteria"])

    prompt = dedent(f"""
        You are grading a repo pre-trade compliance engine's output.

        ## Scenario
        {test_case['scenario']}

        ## Engine output
        {output_str}

        ## Expected output
        ```json
        {expected}
        ```

        ## Solution criteria
        {criteria}

        ## Scoring guidelines
        - 1–3: Wrong pm_override_required (most critical error — mandatory fail)
        - 4–6: Correct pm_override_required but wrong parent breached status or AND/OR logic errors
        - 7–8: Correct verdicts with minor errors in net_value calculations or incomplete reasoning
        - 9–10: Fully correct — right pm_override_required, right per-parent and per-child verdicts,
                mathematically accurate net_values, reasoning cites specific numbers

        Grade ONLY against the expected output and solution criteria above.
        Pay particular attention to AND logic cases: a parent with AND logic is NOT breached
        unless ALL children breach — partial child breach must NOT trigger the parent.

        Return JSON only:
        ```json
        {{
            "strengths": ["...", "..."],
            "weaknesses": ["...", "..."],
            "reasoning": "...",
            "score": <1-10>
        }}
        ```
    """)

    messages = []
    _add_user(messages, prompt)
    _add_assistant(messages, "```json")
    text = _chat(messages, temperature=0.0, stop_sequences=["```"])
    return json.loads(text)


def run_test_case(test_case: dict, run_prompt_fn) -> dict:
    output = run_prompt_fn(test_case["prompt_inputs"])
    grade = grade_output(test_case, output)
    return {
        "output": output,
        "test_case": test_case,
        "score": grade["score"],
        "reasoning": grade["reasoning"],
    }


def run_evaluation(
    run_prompt_fn,
    dataset_file: str = "dataset.json",
    json_output_file: str = "output.json",
    html_output_file: str = "output.html",
    max_workers: int = 3,
) -> list[dict]:
    with open(dataset_file) as f:
        dataset = json.load(f)

    results = []
    completed = 0
    total = len(dataset)
    last_pct = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_test_case, tc, run_prompt_fn): tc for tc in dataset
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error running test case: {e}")
            completed += 1
            pct = int((completed / total) * 100)
            milestone = (pct // 20) * 20
            if milestone > last_pct:
                print(f"Graded {completed}/{total} test cases")
                last_pct = milestone

    avg = mean([r["score"] for r in results]) if results else 0
    print(f"Average score: {avg:.1f} / 10")

    with open(json_output_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(html_output_file, "w", encoding="utf-8") as f:
        f.write(_generate_html_report(results))

    print(f"Results → {json_output_file}  |  Report → {html_output_file}")
    return results

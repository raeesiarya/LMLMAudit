import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any
from tqdm import tqdm


from prompting import load_prompts
from metrics import metrics_total
from database_states import DatabaseState, build_state_db_manager, retrieval_enabled
from equivalence import prompt_row_aliases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_DIR = Path("data/prompts")
DEFAULT_OUTPUT_DIR = Path("outputs/audit")
WANDB_PROJECT = "lmlm-audit"
LOOKUP_VALUE_PATTERN = re.compile(
    r"<\|db_entity\|>.*?<\|db_relationship\|>.*?<\|db_return\|>\s*(.*?)\s*<\|db_end\|>",
    re.DOTALL,
)
DB_MARKUP_SPAN_PATTERN = re.compile(r"<\|db_[^|]+\|>.*?<\|db_end\|>", re.DOTALL)
DB_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|db_[^|]+\|>")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def prepare_prompt(prompt_text: str) -> str:
    return prompt_text.strip()


def clean_answer(answer_text: str) -> str:
    answer_text = answer_text.strip()

    while answer_text.lower().startswith("answer:"):
        answer_text = answer_text[len("answer:") :].strip()

    for prefix in ("the answer is ", "it is ", "it's "):
        if answer_text.lower().startswith(prefix):
            answer_text = answer_text[len(prefix) :].strip()
            break

    stop_markers = [
        "\nQuestion:",
        "\nContext:",
        "\nFact:",
        "\nPrompt:",
        "\nAnswer:",
        "\n\n",
    ]
    for marker in stop_markers:
        if marker in answer_text:
            answer_text = answer_text.split(marker, 1)[0].strip()

    answer_text = html.unescape(answer_text)
    answer_text = DB_MARKUP_SPAN_PATTERN.sub(" ", answer_text)
    answer_text = DB_SPECIAL_TOKEN_PATTERN.sub(" ", answer_text)
    answer_text = HTML_TAG_PATTERN.sub(" ", answer_text)
    answer_text = re.sub(r"\s+", " ", answer_text).strip()
    answer_text = answer_text.strip(" \t\n\r\"'`")

    # Keep the first sentence when the model starts elaborating.
    answer_text = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", answer_text, maxsplit=1)[
        0
    ].strip()

    return answer_text.strip(" \t\n\r\"'`,;:.")


def extract_lookup_values(raw_output: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for match in LOOKUP_VALUE_PATTERN.findall(raw_output):
        value = clean_answer(match)
        if value and value not in seen:
            values.append(value)
            seen.add(value)

    return values


def choose_answer(
    prompt_text: str,
    processed_output: str,
    lookup_values: list[str],
) -> tuple[str, str]:
    cleaned_output = clean_answer(processed_output)
    is_fact_query = prompt_text.strip().endswith("?") or "____" in prompt_text

    if lookup_values and is_fact_query:
        return lookup_values[0], "lookup_value"

    if cleaned_output:
        return cleaned_output, "postprocessed_text"

    if lookup_values:
        return lookup_values[0], "lookup_value"

    return "", "empty"


def compute_generation_budget(
    tokenizer: Any,
    prompt_text: str,
    target_answer_tokens: int,
) -> int:
    prompt_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    # LMLM uses `max_new_tokens` both as the per-step generation cap and as an
    # overall stopping budget over prompt + decoded text, so we need extra slack
    # for lookup markup before the retrieved value appears.
    return max(32, prompt_token_count + target_answer_tokens + 16)


def retrieve_lookup_value(model: Any, lookup_query: str) -> str:
    db_manager = getattr(model, "db_manager", None)
    if db_manager is None:
        return "unknown"

    try:
        return db_manager.retrieve_from_database(lookup_query)
    except Exception:
        fallback_policy = getattr(model, "fallback_policy", "top1_anyway")
        if fallback_policy == "top1_anyway":
            try:
                return db_manager.retrieve_from_database(lookup_query, threshold=-1.0)
            except Exception:
                return "unknown"
        return "unknown"


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int = 12,
    enable_dblookup: bool = True,
) -> str:
    prepared_prompt = prepare_prompt(prompt_text)
    generation_budget = compute_generation_budget(
        tokenizer=tokenizer,
        prompt_text=prepared_prompt,
        target_answer_tokens=max_new_tokens,
    )

    if enable_dblookup:
        model.eval()
        device = next(model.parameters()).device
        model.set_logits_bias(tokenizer)

        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|db_return|>"),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]
        stop_token_ids = [
            token_id
            for token_id in stop_token_ids
            if token_id is not None and token_id != tokenizer.unk_token_id
        ]

        inputs = tokenizer(prepared_prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_processor=model.logits_processor,
            max_new_tokens=generation_budget,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=False,
            do_sample=False,
            eos_token_id=stop_token_ids,
        )

        raw_output = model._decode_with_special_tokens(
            outputs,
            tokenizer,
            input_len,
            prepared_prompt,
        )

        if "<|db_return|>" in raw_output:
            return clean_answer(retrieve_lookup_value(model, raw_output))
    else:
        raw_output = model.generate_with_lookup(
            prompt=prepared_prompt,
            tokenizer=tokenizer,
            max_new_tokens=generation_budget,
            enable_dblookup=False,
            enable_postprocess=False,
        )

    processed_output = str(model.post_process(raw_output, tokenizer)).strip()
    lookup_values = extract_lookup_values(raw_output)
    final_output, _ = choose_answer(
        prompt_text=prompt_text,
        processed_output=processed_output,
        lookup_values=lookup_values,
    )
    return final_output


def _default_retrieval_trace(state: DatabaseState) -> dict[str, Any]:
    return {
        "state": state.value,
        "retrieval_enabled": retrieval_enabled(state),
        "lookup_query": None,
        "threshold": None,
        "all_candidates": [],
        "deleted_candidates": [],
        "retained_candidates": [],
        "selected_candidate": None,
        "selected_value": None,
        "error": None,
    }


def run_prompt_audit(
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    model.db_manager = build_state_db_manager(
        base_db_manager=base_db_manager,
        prompt_row=prompt_row,
        state=state,
    )
    if hasattr(model.db_manager, "reset_trace"):
        model.db_manager.reset_trace()

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_row["prompt_text"],
        max_new_tokens=max_new_tokens,
        enable_dblookup=retrieval_enabled(state),
    )
    retrieval_trace = getattr(model.db_manager, "last_trace", None)

    return {
        "fact_id": prompt_row["fact_id"],
        "subject": prompt_row["subject"],
        "subject_aliases": list(prompt_row_aliases(prompt_row, "subject")),
        "relation": prompt_row["relation"],
        "relation_aliases": list(prompt_row_aliases(prompt_row, "relation")),
        "state": state.value,
        "prompt": prompt_row["prompt_text"],
        "ground_truth": prompt_row["gold_object"],
        "object_aliases": list(prompt_row_aliases(prompt_row, "object")),
        "model_output": answer,
        "retrieval_trace": retrieval_trace or _default_retrieval_trace(state),
    }


def run_audit(
    prompt_path: Path,
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    states: list[DatabaseState],
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    for prompt in tqdm(
        prompts,
        desc=f"Auditing {prompt_path.stem}",
        unit="prompt",
    ):
        for state in states:
            results.append(
                run_prompt_audit(
                    base_db_manager=base_db_manager,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_row=prompt,
                    state=state,
                    max_new_tokens=max_new_tokens,
                )
            )

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False))
            f.write("\n")


def setup_wandb() -> Any:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(f"WANDB_API_KEY was not found after loading {env_path}.")

    import wandb

    wandb.login(key=api_key, relogin=True)
    return wandb


def log_metrics_to_wandb(
    wandb_module: Any,
    prompt_path: Path,
    state: DatabaseState,
    state_metrics: dict[str, float],
    cross_state_metrics: dict[str, float],
    model_name: str,
    database_path: Path,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    run_name = f"{prompt_path.stem}_{state.value}"
    run = wandb_module.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "prompt_file": str(prompt_path),
            "state": state.value,
            "model_name": model_name,
            "database_path": str(database_path),
            "max_new_tokens": max_new_tokens,
            "limit": limit,
        },
        reinit="finish_previous",
    )
    metrics_payload = {
        **{f"state/{key}": value for key, value in state_metrics.items()},
        **{f"cross_state/{key}": value for key, value in cross_state_metrics.items()},
    }
    run.log(metrics_payload)
    run.summary.update(metrics_payload)
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt audit.")
    parser.add_argument(
        "--prompt-files",
        nargs="*",
        type=Path,
        default=None,
        help="Specific prompt JSONL files to audit. Defaults to all files in data/prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompts to run per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSONL audit results will be written.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="kilian-group/LMLM-llama2-382M",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("data/lmlm_database.json"),
        help="Path to the local LMLM database JSON file.",
    )
    parser.add_argument(
        "--disable-dblookup",
        action="store_true",
        help="Deprecated shortcut for running only the DEL-OFF state.",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=[state.value for state in DatabaseState],
        choices=[state.value for state in DatabaseState],
        help="Database states to evaluate.",
    )
    parser.add_argument(
        "--wandb_activation",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable or disable Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_paths = (
        sorted(DEFAULT_PROMPT_DIR.glob("*.jsonl"))
        if args.prompt_files is None
        else args.prompt_files
    )

    if not prompt_paths:
        raise FileNotFoundError(f"No prompt files found in {DEFAULT_PROMPT_DIR}.")

    from model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        database_path=args.database_path,
    )
    base_db_manager = model.db_manager
    state_values = [DatabaseState(state) for state in args.states]
    if args.disable_dblookup:
        state_values = [DatabaseState.DEL_OFF]
    states = state_values
    wandb_module = setup_wandb() if args.wandb_activation == "on" else None

    for prompt_path in prompt_paths:
        results = run_audit(
            prompt_path=prompt_path,
            base_db_manager=base_db_manager,
            model=model,
            tokenizer=tokenizer,
            states=states,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )

        output_path = args.output_dir / f"{prompt_path.stem}_results.jsonl"
        save_results(results, output_path)
        total_metrics = metrics_total(results)
        metrics_by_state = {
            state.value: metrics_total(
                [result for result in results if result["state"] == state.value]
            )
            for state in states
        }

        print("Cross-state audit metrics:")
        print(f"  Paired count: {total_metrics['paired_count']}")
        print(f"  Parametric leakage L(f): {total_metrics['parametric_leakage']:.3f}")
        print(
            "  Retrieval-mediated correctness R(f): "
            f"{total_metrics['retrieval_mediated_correctness']:.3f}"
        )
        print(
            f"  Retrieval artifact rate: {total_metrics['retrieval_artifact_rate']:.3f}"
        )
        print("Metrics by state:")
        for state in states:
            metrics = metrics_by_state[state.value]
            print(f"{state.value}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Exact match: {metrics['exact_match']:.3f}")
            print(f"  Contains match: {metrics['contains_match']:.3f}")
            print(f"  Unknown rate: {metrics['unknown_rate']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  Parametric leakage L(f): {metrics['parametric_leakage']:.3f}")
            print(
                f"  Retrieval-mediated correctness R(f): {metrics['retrieval_mediated_correctness']:.3f}"
            )
            print(
                f"  Retrieval artifact rate: {metrics['retrieval_artifact_rate']:.3f}"
            )
            if wandb_module is not None:
                log_metrics_to_wandb(
                    wandb_module=wandb_module,
                    prompt_path=prompt_path,
                    state=state,
                    state_metrics=metrics,
                    cross_state_metrics=total_metrics,
                    model_name=args.model_name,
                    database_path=args.database_path,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                )
                print(f"  W&B run: {prompt_path.stem}_{state.value}")


if __name__ == "__main__":
    main()

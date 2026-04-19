"""CLI entry point — eval and serve subcommands."""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation from CLI, print JSON result to stdout."""
    from ..models.eval_request import EvalRequest, MetricConfig
    from ..engine.evaluator import EvaluationEngine

    generated_path = Path(args.generated)
    golden_path = Path(args.golden)

    if not generated_path.exists():
        print(f"Error: generated file not found: {generated_path}", file=sys.stderr)
        sys.exit(1)
    if not golden_path.exists():
        print(f"Error: golden file not found: {golden_path}", file=sys.stderr)
        sys.exit(1)

    generated = json.loads(generated_path.read_text())
    golden = json.loads(golden_path.read_text())

    request = EvalRequest(
        generated=generated,
        golden=golden,
        pass_threshold=args.threshold,
    )

    engine = EvaluationEngine()

    async def _run() -> None:
        result = await engine.evaluate(request)
        output = json.loads(result.full_result_json())
        print(json.dumps(output, indent=2))

    asyncio.run(_run())


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(
        "schemaeval.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="schemaeval",
        description="SchemaEval — JSON Output Quality Evaluation Framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate generated JSON against golden")
    eval_parser.add_argument("--generated", required=True, help="Path to generated JSON file")
    eval_parser.add_argument("--golden", required=True, help="Path to golden reference JSON file")
    eval_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Pass threshold (default: 0.7)"
    )
    eval_parser.set_defaults(func=cmd_eval)

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

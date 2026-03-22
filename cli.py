"""
BioResearch CLI: `bioresearch predict|search|serve`

Usage:
    bioresearch search --domain perturbation --iterations 100
    bioresearch search --domain perturbation --iterations 100 --modal
    bioresearch search --domain perturbation --population 4
    bioresearch predict --domain molecules --input "CCO"
    bioresearch serve --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def _get_metric_specs(domain: str):
    """Get metric specifications for a domain."""
    from engine.metrics import MetricSpec, MetricRole, MetricDirection

    if domain == "perturbation":
        return [
            MetricSpec("pearson_deg", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("mse_top20_deg", MetricRole.GUARD, MetricDirection.LOWER, guard_threshold=0.1),
            MetricSpec("direction_acc", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.3),
            MetricSpec("cross_context", MetricRole.BONUS, MetricDirection.LOWER),
            MetricSpec("pearson_all", MetricRole.DIAGNOSTIC, MetricDirection.HIGHER),
        ]
    elif domain == "molecules":
        return [
            MetricSpec("composite_admet", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("gen_profile_match", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("gen_diversity", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.3),
        ]
    elif domain == "trials":
        return [
            MetricSpec("auroc", MetricRole.PRIMARY, MetricDirection.HIGHER),
            MetricSpec("calibration_ece", MetricRole.GUARD, MetricDirection.LOWER, guard_threshold=0.15),
            MetricSpec("net_value", MetricRole.GUARD, MetricDirection.HIGHER, guard_threshold=0.0),
            MetricSpec("lift_at_10", MetricRole.BONUS, MetricDirection.HIGHER),
        ]
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _make_knowledge_fn(domain: str):
    """Create a knowledge function for a domain."""
    from knowledge.retrieval import BioKnowledge

    kb = BioKnowledge()
    available = kb.available_sources()

    if not available:
        print(f"No pre-computed knowledge sources found. Run:")
        print(f"  python -m knowledge.precompute --all")
        print(f"Proceeding without biological knowledge augmentation.\n")
        return None

    print(f"Knowledge sources available: {available}")

    # Load gene list from the domain's dataset for coverage reporting
    gene_list = None
    if domain == "perturbation":
        try:
            from domains.perturbation.prepare import load_data
            data = load_data("synthetic")
            gene_list = data.gene_names[:100]  # sample for speed
        except Exception:
            pass

    def knowledge_fn():
        return kb.get_knowledge_packet(gene_list=gene_list)

    return knowledge_fn


def _make_modal_runner(domain: str):
    """Create a Modal-based parallel seed runner."""
    try:
        import modal  # noqa: F401
    except ImportError:
        print("Modal not installed. Install with: pip install modal")
        print("Falling back to local execution.\n")
        return None

    domain_dir = PROJECT_ROOT / "domains" / domain
    prepare_code = (domain_dir / "prepare.py").read_text()

    from engine.metrics import SeedResult

    def runner(domain_dir_str: str, train_code: str, seeds: list[int], time_budget: int) -> list[SeedResult]:
        from infra.modal_app import app, run_bio_experiment

        results = []
        with app.run():
            modal_results = list(run_bio_experiment.map(
                [train_code] * len(seeds),
                [prepare_code] * len(seeds),
                seeds,
                [time_budget] * len(seeds),
                [domain] * len(seeds),
            ))

        for r in modal_results:
            metrics = r.get("metrics", {})
            peak_vram = metrics.pop("peak_vram_mb", 0.0)
            results.append(SeedResult(
                seed=r["seed"],
                metrics=metrics,
                train_seconds=r.get("train_seconds", 0.0),
                peak_vram_mb=peak_vram,
                success=r["success"],
                error=r.get("error"),
            ))
        return results

    return runner


def _make_local_runner():
    """Create a local sequential seed runner using the Colab/local GPU."""
    from engine.loop import run_local_experiment
    from engine.metrics import SeedResult

    def runner(domain_dir: str, train_code: str, seeds: list[int], time_budget: int) -> list[SeedResult]:
        results = []
        for seed in seeds:
            print(f"  [local] Running seed {seed}...", end=" ", flush=True)
            result = run_local_experiment(domain_dir, train_code, seed, time_budget)
            status = "OK" if result.success else f"FAIL: {result.error[:80] if result.error else 'unknown'}"
            print(status)
            results.append(result)
        return results

    return runner


def _make_prescreen_runner(domain: str):
    """
    Create a pre-screen runner: run 1 seed locally first as sanity check.
    If it fails, skip Modal dispatch entirely (saves cost).
    If it passes, dispatch remaining seeds to Modal.
    """
    from engine.loop import run_local_experiment
    from engine.metrics import SeedResult

    modal_runner = _make_modal_runner(domain)
    if modal_runner is None:
        print("Modal unavailable. Pre-screen mode falling back to full local.")
        return _make_local_runner()

    def runner(domain_dir: str, train_code: str, seeds: list[int], time_budget: int) -> list[SeedResult]:
        if not seeds:
            return []

        prescreen_seed = seeds[0]
        remaining_seeds = seeds[1:]

        print(f"  [prescreen] Running seed {prescreen_seed} locally...", end=" ", flush=True)
        prescreen_result = run_local_experiment(domain_dir, train_code, prescreen_seed, time_budget)

        if not prescreen_result.success:
            print(f"FAIL: {prescreen_result.error[:80] if prescreen_result.error else 'unknown'}")
            print(f"  [prescreen] Pre-screen failed. Skipping Modal dispatch for {len(remaining_seeds)} seeds.")
            results = [prescreen_result]
            for seed in remaining_seeds:
                results.append(SeedResult(
                    seed=seed,
                    metrics={},
                    train_seconds=0.0,
                    success=False,
                    error=f"Skipped: pre-screen seed {prescreen_seed} failed",
                ))
            return results

        print("OK")
        print(f"  [prescreen] Pre-screen passed. Dispatching {len(remaining_seeds)} seeds to Modal...")

        if remaining_seeds:
            modal_results = modal_runner(domain_dir, train_code, remaining_seeds, time_budget)
            return [prescreen_result] + modal_results
        return [prescreen_result]

    return runner


def _make_hybrid_runner(domain: str):
    """
    Create a hybrid runner: run seed 0 on local GPU and remaining seeds
    on Modal simultaneously using ThreadPoolExecutor.
    """
    from engine.loop import run_local_experiment

    modal_runner = _make_modal_runner(domain)
    if modal_runner is None:
        print("Modal unavailable. Hybrid mode falling back to full local.")
        return _make_local_runner()

    def runner(domain_dir: str, train_code: str, seeds: list[int], time_budget: int) -> list:
        if not seeds:
            return []

        local_seed = seeds[0]
        modal_seeds = seeds[1:]

        def run_local():
            print(f"  [hybrid/local] Running seed {local_seed} on local GPU...")
            result = run_local_experiment(domain_dir, train_code, local_seed, time_budget)
            status = "OK" if result.success else "FAIL"
            print(f"  [hybrid/local] Seed {local_seed}: {status}")
            return result

        def run_modal():
            if not modal_seeds:
                return []
            print(f"  [hybrid/modal] Dispatching seeds {modal_seeds} to Modal...")
            results = modal_runner(domain_dir, train_code, modal_seeds, time_budget)
            succeeded = sum(1 for r in results if r.success)
            print(f"  [hybrid/modal] {succeeded}/{len(results)} seeds succeeded")
            return results

        with ThreadPoolExecutor(max_workers=2) as executor:
            local_future = executor.submit(run_local)
            modal_future = executor.submit(run_modal)

            local_result = local_future.result()
            modal_results = modal_future.result()

        return [local_result] + modal_results

    return runner


def cmd_search(args):
    """Run autoresearch loop for a domain."""
    from engine.loop import LoopConfig, autoresearch_loop
    from engine.orchestrator import OrchestratorConfig

    domain_dir = str(PROJECT_ROOT / "domains" / args.domain)
    output_dir = str(PROJECT_ROOT / "results" / args.domain)

    metric_specs = _get_metric_specs(args.domain)
    knowledge_fn = _make_knowledge_fn(args.domain)

    # Set up compute runner based on mode
    run_seeds_parallel = None
    compute_mode = args.compute

    if compute_mode == "modal":
        run_seeds_parallel = _make_modal_runner(args.domain)
        if run_seeds_parallel:
            print("Compute mode: Modal (all seeds on cloud GPUs)")
        else:
            print("Modal unavailable. Falling back to local sequential execution.")
            compute_mode = "local"

    if compute_mode == "local":
        run_seeds_parallel = _make_local_runner()
        print("Compute mode: Local (all seeds on local GPU, sequential)")
    elif compute_mode == "prescreen":
        run_seeds_parallel = _make_prescreen_runner(args.domain)
        print("Compute mode: Pre-screen (1 local sanity check, then Modal)")
    elif compute_mode == "hybrid":
        run_seeds_parallel = _make_hybrid_runner(args.domain)
        print("Compute mode: Hybrid (1 local + remaining on Modal, parallel)")

    orch_config = OrchestratorConfig(
        model=args.model or "claude-opus-4-6",
        backend=args.backend,
        vertex_project_id=args.vertex_project or "",
        vertex_region=args.vertex_region,
    )

    if args.population and args.population > 1:
        from engine.population import PopulationSearch, PopulationConfig
        pop_config = PopulationConfig(
            num_agents=args.population,
            domain_dir=domain_dir,
            output_dir=output_dir,
            num_seeds=args.seeds,
            max_iterations=args.iterations,
            time_budget_per_seed=args.time_budget,
            orchestrator_config=orch_config,
        )
        search = PopulationSearch(
            metric_specs=metric_specs,
            config=pop_config,
            knowledge_fn=knowledge_fn,
            run_seeds_parallel=run_seeds_parallel,
        )
        search.run()
    else:
        config = LoopConfig(
            domain_dir=domain_dir,
            output_dir=output_dir,
            num_seeds=args.seeds,
            max_iterations=args.iterations,
            time_budget_per_seed=args.time_budget,
            orchestrator_config=orch_config,
        )
        autoresearch_loop(
            metric_specs=metric_specs,
            loop_config=config,
            knowledge_fn=knowledge_fn,
            run_seeds_parallel=run_seeds_parallel,
        )


def cmd_predict(args):
    """Run prediction for a domain."""
    domain = args.domain

    if domain == "perturbation":
        from domains.perturbation.prepare import load_data, evaluate, print_metrics
        dataset_name = args.dataset or "synthetic"
        dataset = load_data(dataset_name)
        print(f"Loaded {dataset.n_samples} samples, {dataset.n_genes} genes")

        # Import and run the current train.py model
        sys.path.insert(0, str(PROJECT_ROOT))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "perturb_train", str(PROJECT_ROOT / "domains" / "perturbation" / "train.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute — just load the model class
        print("Running perturbation model on test set...")
        from domains.perturbation.train import LinearPerturbModel
        model = LinearPerturbModel(n_genes=dataset.n_genes)
        train_ctrl = dataset.ctrl_expr[dataset.train_idx]
        train_pert = dataset.pert_expr[dataset.train_idx]
        train_names = [dataset.pert_names[i] for i in dataset.train_idx]
        model.fit(train_ctrl, train_pert, train_names)

        test_ctrl = dataset.ctrl_expr[dataset.test_idx]
        test_names = [dataset.pert_names[i] for i in dataset.test_idx]
        predictions = model.predict(test_ctrl, test_names)

        metrics = evaluate(
            predictions, dataset.pert_expr[dataset.test_idx],
            test_names, dataset.deg_indices,
            ctrl_expr=test_ctrl,
        )
        print_metrics(metrics)

    elif domain == "molecules":
        if args.input:
            print(f"Predicting ADMET for: {args.input}")
            from domains.molecules.train import FingerprintMLPModel
            from domains.molecules.prepare import load_data, evaluate, _compute_fingerprints
            dataset = load_data(use_tdc=False)

            # Train model
            model = FingerprintMLPModel(
                n_endpoints=len(dataset.endpoint_names),
                endpoint_types=dataset.endpoint_types,
                fp_dim=dataset.fingerprints.shape[1],
            )
            model.fit(dataset.fingerprints[dataset.train_idx], dataset.labels[dataset.train_idx])

            # Predict for input SMILES
            fp = _compute_fingerprints([args.input], fp_dim=dataset.fingerprints.shape[1])
            preds = model.predict(fp)
            print(f"\nADMET predictions for {args.input}:")
            for j, name in enumerate(dataset.endpoint_names):
                print(f"  {name}: {preds[0, j]:.4f}")
        else:
            print("Pass --input <SMILES> for a prediction, or omit for test set evaluation.")

    elif domain == "trials":
        from domains.trials.train import TrialPredictionModel
        from domains.trials.prepare import load_data, evaluate
        dataset = load_data(use_tdc=False)

        model = TrialPredictionModel()
        model.fit(dataset.features[dataset.train_idx], dataset.labels[dataset.train_idx])

        predictions = model.predict(dataset.features[dataset.test_idx])
        phases = [dataset.phases[i] for i in dataset.test_idx]
        metrics = evaluate(predictions, dataset.labels[dataset.test_idx], phases)
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")

    else:
        print(f"Unknown domain: {domain}")
        sys.exit(1)


def cmd_serve(args):
    """Launch Gradio web UI."""
    from web.app import create_app
    app = create_app()
    app.launch(server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(description="BioResearch: Autonomous biology ML research")
    subparsers = parser.add_subparsers(dest="command")

    # search command
    search_parser = subparsers.add_parser("search", help="Run autoresearch loop")
    search_parser.add_argument("--domain", required=True, choices=["perturbation", "molecules", "trials"])
    search_parser.add_argument("--iterations", type=int, default=100)
    search_parser.add_argument("--seeds", type=int, default=5)
    search_parser.add_argument("--time-budget", type=int, default=600)
    search_parser.add_argument("--population", type=int, default=0, help="Number of agents for population search (0=single)")
    search_parser.add_argument("--compute", type=str, default="modal",
                               choices=["modal", "local", "prescreen", "hybrid"],
                               help="Compute mode: modal (all Modal), local (all local GPU), "
                                    "prescreen (1 local then Modal), hybrid (1 local + Modal parallel)")
    search_parser.add_argument("--model", type=str, default=None, help="Claude model to use (default: claude-opus-4-6)")
    search_parser.add_argument("--backend", type=str, default="anthropic",
                               choices=["anthropic", "vertex"],
                               help="Claude API backend (default: anthropic)")
    search_parser.add_argument("--vertex-project", type=str, default=None,
                               help="GCP project ID for Vertex AI backend")
    search_parser.add_argument("--vertex-region", type=str, default="us-east5",
                               help="GCP region for Vertex AI backend (default: us-east5)")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--domain", required=True, choices=["perturbation", "molecules", "trials"])
    predict_parser.add_argument("--input", type=str, help="Input data (e.g., SMILES string)")
    predict_parser.add_argument("--dataset", type=str, help="Dataset name")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Launch web UI")
    serve_parser.add_argument("--port", type=int, default=7860)
    serve_parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

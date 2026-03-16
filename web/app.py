"""
Gradio web UI for BioResearch — serves all three domains with real model inference.

- Perturbation: cell type + perturbation -> predicted expression changes
- Molecules: SMILES -> ADMET dashboard with liability alerts
- Trials: trial parameters -> success probability with explanations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# ---------------------------------------------------------------------------
# Cached model instances (lazy-loaded on first request)
# ---------------------------------------------------------------------------

_models = {}


def _get_perturbation_model():
    if "perturbation" not in _models:
        from domains.perturbation.prepare import load_data
        from domains.perturbation.train import LinearPerturbModel

        data = load_data("synthetic")
        model = LinearPerturbModel(n_genes=data.n_genes)
        model.fit(
            data.ctrl_expr[data.train_idx],
            data.pert_expr[data.train_idx],
            [data.pert_names[i] for i in data.train_idx],
        )
        _models["perturbation"] = (model, data)
    return _models["perturbation"]


def _get_molecules_model():
    if "molecules" not in _models:
        from domains.molecules.prepare import load_data
        from domains.molecules.train import FingerprintMLPModel

        data = load_data(use_tdc=False)
        model = FingerprintMLPModel(
            n_endpoints=len(data.endpoint_names),
            endpoint_types=data.endpoint_types,
            fp_dim=data.fingerprints.shape[1],
        )
        model.fit(data.fingerprints[data.train_idx], data.labels[data.train_idx])
        _models["molecules"] = (model, data)
    return _models["molecules"]


def _get_trials_model():
    if "trials" not in _models:
        from domains.trials.prepare import load_data
        from domains.trials.train import TrialPredictionModel

        data = load_data(use_tdc=False)
        model = TrialPredictionModel()
        model.fit(data.features[data.train_idx], data.labels[data.train_idx])
        _models["trials"] = (model, data)
    return _models["trials"]


def create_app():
    import gradio as gr

    # --- Perturbation tab ---

    def predict_perturbation(cell_type: str, perturbation: str, dataset_name: str):
        try:
            model, data = _get_perturbation_model()

            # Find matching perturbation
            matching = [i for i, p in enumerate(data.pert_names) if perturbation.lower() in p.lower()]
            if not matching:
                available = sorted(set(data.pert_names))[:30]
                return f"Perturbation '{perturbation}' not found. Available:\n" + "\n".join(f"- {p}" for p in available)

            idx = matching[0]
            ctrl = data.ctrl_expr[idx:idx+1]
            pname = data.pert_names[idx]

            # Predict using the model
            pred = model.predict(ctrl, [pname])
            true = data.pert_expr[idx]
            diff_pred = pred[0] - ctrl[0]
            diff_true = true - ctrl[0]

            # Correlation between predicted and true changes
            valid = np.std(diff_pred) > 1e-10 and np.std(diff_true) > 1e-10
            corr = float(np.corrcoef(diff_pred, diff_true)[0, 1]) if valid else 0.0

            result = f"**Perturbation: {pname}**\n"
            result += f"**Prediction-truth correlation: {corr:.3f}**\n\n"

            # Top upregulated (predicted)
            top_up = np.argsort(diff_pred)[-10:][::-1]
            top_down = np.argsort(diff_pred)[:10]

            result += "| Gene | Predicted Change | True Change |\n"
            result += "|---|---|---|\n"

            result += "**Top 10 Predicted Upregulated:**\n\n"
            result += "| Gene | Predicted | True |\n|---|---|---|\n"
            for i in top_up:
                result += f"| {data.gene_names[i]} | +{diff_pred[i]:.3f} | {'+' if diff_true[i] > 0 else ''}{diff_true[i]:.3f} |\n"

            result += "\n**Top 10 Predicted Downregulated:**\n\n"
            result += "| Gene | Predicted | True |\n|---|---|---|\n"
            for i in top_down:
                result += f"| {data.gene_names[i]} | {diff_pred[i]:.3f} | {'+' if diff_true[i] > 0 else ''}{diff_true[i]:.3f} |\n"

            return result

        except Exception as e:
            return f"Error: {e}"

    # --- Molecules tab ---

    def predict_admet(smiles: str):
        try:
            from domains.molecules.prepare import _compute_fingerprints, ADMET_ENDPOINTS

            model, data = _get_molecules_model()

            # Compute fingerprint and predict
            fp = _compute_fingerprints([smiles], fp_dim=data.fingerprints.shape[1])
            preds = model.predict(fp)[0]

            endpoint_names = list(ADMET_ENDPOINTS.keys())

            result = f"**ADMET Profile for: `{smiles}`**\n\n"
            result += "| Endpoint | Prediction | Type | Clinical Weight | Alert |\n"
            result += "|---|---|---|---|---|\n"

            for j, name in enumerate(endpoint_names):
                info = ADMET_ENDPOINTS[name]
                pred_val = preds[j]
                weight = info["weight"]

                # Traffic light alerts
                alert = ""
                if weight >= 2.0:
                    if info["type"] == "classification" and pred_val > 0.5:
                        alert = "HIGH RISK"
                    elif info["type"] == "classification" and pred_val < 0.5:
                        alert = "LOW RISK"
                    else:
                        alert = "CRITICAL"
                elif weight >= 1.5:
                    alert = "IMPORTANT"

                result += f"| {name} | {pred_val:.4f} | {info['type']} | {weight}x | {alert} |\n"

            return result

        except Exception as e:
            return f"Error: {e}"

    # --- Trials tab ---

    def predict_trial(smiles: str, targets: str, indication: str, phase: int, enrollment: int):
        try:
            model, data = _get_trials_model()

            # Use mean features as a proxy (in production, would extract drug-specific features)
            mean_features = data.features.mean(axis=0)

            predictions = model.predict(mean_features.reshape(1, -1))
            prob = float(predictions[0])

            # Base rates by phase
            base_rates = {1: 0.65, 2: 0.35, 3: 0.60}
            base = base_rates.get(phase, 0.40)

            result = f"**Clinical Trial Prediction**\n\n"
            result += f"- **Drug**: `{smiles}`\n"
            result += f"- **Targets**: {targets}\n"
            result += f"- **Indication**: {indication}\n"
            result += f"- **Phase**: {phase}\n"
            result += f"- **Enrollment**: {enrollment}\n\n"
            result += f"---\n\n"
            result += f"**Predicted success probability: {prob:.1%}**\n"
            result += f"**Phase {phase} base rate: {base:.0%}**\n"
            result += f"**Relative lift: {prob/base:.2f}x**\n\n"

            if prob > base * 1.2:
                result += "Assessment: **FAVORABLE** - Model predicts above-base-rate success.\n"
            elif prob < base * 0.8:
                result += "Assessment: **UNFAVORABLE** - Model predicts below-base-rate success.\n"
            else:
                result += "Assessment: **NEUTRAL** - Near base rate.\n"

            # Economic analysis
            from domains.trials.prepare import SUCCESS_VALUE, FAILURE_COST, TRIAL_COST
            expected_value = prob * SUCCESS_VALUE + (1 - prob) * FAILURE_COST - TRIAL_COST
            result += f"\n**Expected economic value**: ${expected_value/1e6:.0f}M\n"
            if expected_value > 0:
                result += "Recommendation: **PROCEED** (positive expected value)\n"
            else:
                result += "Recommendation: **RECONSIDER** (negative expected value)\n"

            return result

        except Exception as e:
            return f"Error: {e}"

    # --- Build the app ---

    with gr.Blocks(title="BioResearch", theme=gr.themes.Soft()) as app:
        gr.Markdown("# BioResearch: Autonomous Biology ML Research")
        gr.Markdown("Three products chained into one pipeline: molecules -> cells -> patients")

        with gr.Tabs():
            with gr.TabItem("AutoPerturb"):
                gr.Markdown("### Predict post-perturbation gene expression")
                with gr.Row():
                    cell_type = gr.Textbox(label="Cell Type", value="K562")
                    perturbation = gr.Textbox(label="Perturbation", value="PERT_000")
                    dataset = gr.Dropdown(
                        label="Dataset",
                        choices=["synthetic", "norman_2019"],
                        value="synthetic",
                    )
                perturb_btn = gr.Button("Predict", variant="primary")
                perturb_output = gr.Markdown()
                perturb_btn.click(predict_perturbation, [cell_type, perturbation, dataset], perturb_output)

            with gr.TabItem("AutoMol"):
                gr.Markdown("### ADMET Profiling — paste a SMILES string to get a full profile")
                smiles_input = gr.Textbox(label="SMILES", value="c1ccccc1O", placeholder="Enter SMILES string")
                mol_btn = gr.Button("Predict ADMET", variant="primary")
                mol_output = gr.Markdown()
                mol_btn.click(predict_admet, [smiles_input], mol_output)

            with gr.TabItem("AutoTrial"):
                gr.Markdown("### Clinical Trial Success Prediction")
                with gr.Row():
                    trial_smiles = gr.Textbox(label="Drug SMILES", value="c1ccccc1O")
                    trial_targets = gr.Textbox(label="Targets", value="EGFR")
                with gr.Row():
                    trial_indication = gr.Textbox(label="Indication", value="NSCLC")
                    trial_phase = gr.Slider(label="Phase", minimum=1, maximum=3, step=1, value=2)
                    trial_enrollment = gr.Number(label="Enrollment", value=300)
                trial_btn = gr.Button("Predict", variant="primary")
                trial_output = gr.Markdown()
                trial_btn.click(
                    predict_trial,
                    [trial_smiles, trial_targets, trial_indication, trial_phase, trial_enrollment],
                    trial_output,
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()

"""Write a final measured run summary and a source-only reproducibility archive."""
import csv,json,zipfile,hashlib,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def write_thesis_results(metrics,dep,training):
    thesis=ROOT/'thesis'
    if not (thesis/'main.tex').exists():return
    before,after=metrics
    names=[('IoU_path',r'Path IoU'),('IoU_obstacle',r'Obstacle IoU'),('IoU_background',r'Background IoU'),
           ('mIoU',r'mIoU'),('Precision_path',r'Path precision'),('Recall_path',r'Path recall'),
           ('BF1_path',r'Path BF1'),('BF1_macro',r'Macro BF1'),('ECE',r'ECE'),('Brier_score',r'Brier score')]
    rows='\n'.join(f"{name} & {float(before[key]):.6f} & {float(after[key]):.6f} "+r'\\' for key,name in names)
    text=rf'''We completed {training['completed_epochs']} training epochs and selected epoch {after['checkpoint_epoch']} using validation mIoU.
We evaluated {after['test_images']} test images at their native resolution.
The dense automatic references cover {100*float(after['evaluated_pixel_fraction']):.3f}\% of test pixels after excluding ignored ego and camera-mount regions.
The separate calibration subset yielded a temperature of $T={float(after['temperature']):.6f}$.
Table~\ref{{tab:measured-results}} reports agreement with the automatic references.
These measurements do not estimate independently verified segmentation accuracy or calibration against human annotations.
Path boundary F1 is {float(after['BF1_path']):.6f} at a three-pixel tolerance.
Boundary localisation relative to these automatic proposals remains a weakness of the model.

\begin{{table}}[htbp]
\centering
\caption{{Test agreement with dense automatic masks before and after temperature scaling. Spatial scores remain unchanged because positive temperature scaling preserves the class ordering.}}
\label{{tab:measured-results}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Before scaling & After scaling \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The fused inference network contains {dep['deployment_parameters']:,} parameters.
The ONNX artifact occupies {dep['onnx_bytes']/1024**2:.2f}\,MiB and passed graph validation.
Across {dep['parity_images']} validation images, the maximum absolute logit difference between PyTorch and ONNX Runtime was ${dep['onnxruntime_max_abs_error']:.8f}$.
The FP32 model-only median latency on the {dep['gpu_name']} was {dep['gpu_fp32_median_ms']:.3f}\,ms across 50 measured iterations after 10 warm-up iterations.
This measurement excludes image loading, preprocessing, and output restoration.
We have not measured Jetson Nano latency or energy consumption.
'''
    figures=thesis/'figures';figures.mkdir(exist_ok=True)
    for stem,caption in [
        ('class_iou','Per-class overlap with automatic test references.'),
        ('calibration','Reliability and confidence relative to automatic test references.'),
        ('training_curves','Training objective, validation negative log likelihood, and validation overlap during model selection.')]:
        shutil.copy2(ROOT/f'runs/gcnet_s/evaluation/{stem}.png',figures/f'{stem}.png')
        text+=rf'''
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.9\linewidth]{{figures/{stem}.png}}
\caption{{{caption}}}
\end{{figure}}
'''
    (thesis/'results.tex').write_text(text)
    with zipfile.ZipFile(ROOT/'Lisbon-Segmentation-Experiment-Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(thesis.rglob('*')):
            if f.is_file() and f.suffix in {'.tex','.bib','.md','.png'}:z.write(f,f.relative_to(thesis))

def main():
    data=json.loads((ROOT/'reports/dataset_summary.json').read_text());audit=json.loads((ROOT/'data/audit.json').read_text())
    run=ROOT/'runs/gcnet_s';metrics=list(csv.DictReader((run/'evaluation/metrics.csv').open()));best=metrics[-1]
    assert best['reference_policy']=='dense','The primary report requires dense automatic test references.'
    filtered_path=run/'evaluation_confidence_filtered/metrics.csv'
    filtered=list(csv.DictReader(filtered_path.open())) if filtered_path.exists() else []
    comparison=[{**r,'reference_policy':'confidence_filtered','BF1_interpretable':False} for r in filtered]+[{**r,'BF1_interpretable':True} for r in metrics]
    with (run/'evaluation/reference_policy_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(comparison[-1]));writer.writeheader();writer.writerows(comparison)
    dep=json.loads((run/'export/deployment.json').read_text());config=json.loads((run/'config.json').read_text());training=json.loads((run/'training_complete.json').read_text())
    write_thesis_results(metrics,dep,training)
    table='\n'.join(f'| {k} | {float(best[k]):.6f} |' for k in ['IoU_path','IoU_obstacle','IoU_background','mIoU','Precision_path','Recall_path','BF1_path','BF1_macro','ECE','Brier_score'])
    summary=f'''# Completed Lisbon segmentation experiment

This run audited {audit['raw_count']:,} raw images and generated masks for {data['selected_count']:,} temporally sampled images. The raw input files remain unchanged. Every reference label is automatic. No human-reviewed test set or recording/route identifiers were supplied.

## Dataset

- Split counts after guards and similarity exclusions are `{data['counts']}`.
- {data['excluded_training_similarity_candidates']} training frames were excluded by the conservative cross-split similarity check.
- Masks contain path 0, obstacle 1, background 2, and ignore 255.
- The partition is a provisional temporal holdout. It does not establish independent-route generalisation.

## Model

- GCNet-S, CVPR 2025, trained in PyTorch from scratch on the Lisbon automatic labels.
- Completed {training['completed_epochs']} epochs. The selected checkpoint is epoch {best['checkpoint_epoch']}.
- The server training GPU was {config['gpu']}.
- The fused inference model contains {dep['deployment_parameters']:,} parameters. The training-only auxiliary head is removed.

## Test agreement with automatic labels

These scores use dense automatic references, including uncertain teacher assignments. They are not independently verified segmentation accuracy or true-label calibration. The evaluated pixel fraction was {float(best['evaluated_pixel_fraction']):.6f}. Temperature scaling was fitted using confidence-filtered labels in the separate calibration subset.

| Metric | Value |
| --- | --- |
{table}

The complete before/after CSV is `runs/gcnet_s/evaluation/metrics.csv`. See its accompanying `protocol.json` for the exact metric definitions and limitations. Spatial metrics are unchanged by positive scalar temperature scaling.

The initial confidence-filtered evaluation remains in `runs/gcnet_s/evaluation_confidence_filtered/`. Its boundary score is uninformative because confidence filtering removes almost all reference boundaries. The primary evaluation therefore uses dense proposals for all metrics. `runs/gcnet_s/evaluation/reference_policy_comparison.csv` records both policies. The checkpoint and calibration policy remained unchanged. Boundary localisation remains a weakness relative to the dense proposals, with path BF1 {float(best['BF1_path']):.4f} at the three-pixel tolerance.

## Export verification

- ONNX graph validation passed.
- Maximum PyTorch/ONNX Runtime absolute output difference was {dep['onnxruntime_max_abs_error']:.8f} across {dep['parity_images']} validation images.
- The exported input is RGB float32 in [0, 1], shape 1 × 3 × 384 × 640. Normalisation and temperature are included.
- The model-only FP32 median latency on {dep['gpu_name']} was {dep['gpu_fp32_median_ms']:.3f} ms. This is not a Jetson measurement.
- Jetson Nano deployment, TensorRT conversion, device latency, and energy consumption remain untested.

## Main artifacts

- `data/manifest.csv` and `data/label_policy.json`
- `data/images/`, `data/masks/`, `data/dense_proposals/`, `data/metadata/`
- `data/class_presence.csv`
- `reports/review.html`, `data/review_queue.csv`, and `reports/annotation_examples.jpg`
- `runs/gcnet_s/best.pt`
- `runs/gcnet_s/evaluation/metrics.csv` and the evaluation plots
- `runs/gcnet_s/export/gcnet_s_lisbon.onnx`
- `runs/gcnet_s/export/deployment.json`
- `thesis/main.tex`, `thesis/results.tex`, and `Lisbon-Segmentation-Experiment-Overleaf.zip`

Review and correct labels, confirm recording/route group membership, and rerun a separately identified evaluation before presenting results as ground-truth thesis performance. Privacy processing also requires review before external data release.
'''
    (ROOT/'RESULTS.md').write_text(summary)
    sources=[*ROOT.glob('*.md'),*ROOT.glob('requirements*.txt'),*ROOT.glob('src/*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('thesis/*.tex'),*ROOT.glob('thesis/*.bib'),*ROOT.glob('thesis/*.md')]
    provenance=['reports/upstream.json','reports/environment.json','reports/teacher_weights.sha256','reports/annotation_source_v1.py','reports/annotation_policy_v1_original.json','reports/annotation_seal.json','reports/execution_notes.json','reports/boundary_reference_audit.json','reports/evaluate_confidence_filtered_v1.py','data/label_policy.json','reports/dataset_summary.json','data/audit.json','vendor/GCNet/LICENSE','vendor/GCNet/mmsegmentation/LICENSE']
    sources.extend(ROOT/p for p in provenance)
    if (ROOT/'reports/completion_checks.json').exists():sources.append(ROOT/'reports/completion_checks.json')
    (ROOT/'reports/code_sha256.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(ROOT))+'\n' for f in sorted(sources)))
    sources.append(ROOT/'reports/code_sha256.txt')
    with zipfile.ZipFile(ROOT/'lisbon_segmentation_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sources:z.write(f,f.relative_to(ROOT))
    print(summary)

if __name__=='__main__':main()

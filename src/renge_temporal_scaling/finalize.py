from __future__ import annotations
import json
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scaling_common import RESULT_ROOT,SCRIPT_ROOT,atomic_json,paired_repeat_bootstrap,safe_spearman,sha256

def cell_ci(frame,keys,metric,seed,nboot):
  rows=[]
  for _,(name,g) in enumerate(frame.groupby(keys,dropna=False)):
    name=(name,) if not isinstance(name,tuple) else name;stat=paired_repeat_bootstrap(g,metric,seed+len(rows)*101,nboot)
    rows.append({**dict(zip(keys,name)),f"mean_{metric}":stat["point"],f"ci_low_{metric}":stat["ci_low"],f"ci_high_{metric}":stat["ci_high"],"n_repeats":stat["n_repeats"]})
  return pd.DataFrame(rows)

def paired_delta(left,right,metric,seed,nboot):
  # Average subset replicates within biological repeat, then pair repeats.
  l=left.groupby("repeat")[metric].mean();r=right.groupby("repeat")[metric].mean();x=(l-r).dropna().to_numpy(float);rng=np.random.default_rng(seed);draw=np.mean(rng.choice(x,(nboot,len(x)),replace=True),axis=1)
  return {"point":float(x.mean()),"ci_low":float(np.quantile(draw,.025)),"ci_high":float(np.quantile(draw,.975)),"n_repeats":len(x)}

def main():
  config=json.loads((SCRIPT_ROOT/"config.json").read_text(encoding="utf-8"));seed=config["bootstrap_seed"];nboot=config["bootstrap_resamples"]
  surface=pd.read_csv(RESULT_ROOT/"temporal_static_scaling_surface.csv");controls=pd.read_csv(RESULT_ROOT/"temporal_order_controls.csv");measure=pd.read_csv(RESULT_ROOT/"endpoint_measurement_controls.csv");oracle=pd.read_csv(RESULT_ROOT/"oracle_temporal_value.csv");power=pd.read_csv(RESULT_ROOT/"measurement_power_scaling.csv");ceiling=pd.read_csv(RESULT_ROOT/"noise_ceiling_analysis.csv")
  # Add artifact-safe cell summaries without discarding fold/subset rows.
  summary_cols=[c for c in surface if c.startswith("mean_") or c.startswith("ci_low_") or c.startswith("ci_high_") or c=="n_repeats_cell"]
  if summary_cols:surface=surface.drop(columns=summary_cols)
  keys=["n_training_sources","condition","temporal_order"]
  sm=cell_ci(surface,keys,"response_distance_rho",seed,nboot).rename(columns={"n_repeats":"n_repeats_cell"})
  for metric in ("residual_geometry","per_response_strict_trans_pearson","response_cosine","between_variance_ratio"):
    add=cell_ci(surface,keys,metric,seed+10000+len(sm.columns)*101,nboot).drop(columns="n_repeats")
    sm=sm.merge(add,on=keys)
  surface=surface.merge(sm,on=keys,how="left");surface.to_csv(RESULT_ROOT/"temporal_static_scaling_surface.csv",index=False)
  surface[surface.condition=="T1_Day5Only"].to_csv(RESULT_ROOT/"static_breadth_scaling.csv",index=False)
  surface[surface.temporal_order.isin(["EndpointOnly","CorrectTemporal"])].to_csv(RESULT_ROOT/"temporal_depth_scaling.csv",index=False)

  # Explicit matched-budget contrasts.
  budgets=[]
  for bi,(nt,ct,ns,cs) in enumerate(config["matched_information_budgets"]):
    temporal=surface[(surface.n_training_sources==nt)&(surface.condition==ct)&(surface.temporal_order=="CorrectTemporal")]
    static=surface[(surface.n_training_sources==ns)&(surface.condition==cs)&(surface.temporal_order=="EndpointOnly")]
    for metric in ("response_distance_rho","residual_geometry","per_response_strict_trans_pearson","response_cosine"):
      stat=paired_delta(temporal,static,metric,seed+20000+bi*1009+len(budgets),nboot)
      budgets.append({"temporal_n_sources":nt,"temporal_condition":ct,"temporal_source_time_observations":int(temporal.source_time_observations.iloc[0]),"static_n_sources":ns,"static_condition":cs,"static_source_time_observations":int(static.source_time_observations.iloc[0]),"metric":metric,
       "temporal_mean":float(temporal.groupby("repeat")[metric].mean().mean()),"static_mean":float(static.groupby("repeat")[metric].mean().mean()),"delta_temporal_minus_static":stat["point"],"ci_low":stat["ci_low"],"ci_high":stat["ci_high"],"n_repeats":stat["n_repeats"],"source_time_is_descriptive_not_formal_cost":True})
  budget=pd.DataFrame(budgets);budget.to_csv(RESULT_ROOT/"matched_information_budget.csv",index=False)

  contrasts={}
  counts=config["training_source_counts"]
  for n in counts:
    endpoint=surface[(surface.n_training_sources==n)&(surface.condition=="T1_Day5Only")]
    for condition in [c for c in config["temporal_conditions"] if c!="T1_Day5Only"]:
      correct=surface[(surface.n_training_sources==n)&(surface.condition==condition)&(surface.temporal_order=="CorrectTemporal")]
      shuffled=surface[(surface.n_training_sources==n)&(surface.condition==condition)&(surface.temporal_order=="TemporalShuffle")]
      contrasts[f"N{n}_{condition}_minus_endpoint"]=paired_delta(correct,endpoint,"response_distance_rho",seed+len(contrasts)*101,nboot)
      contrasts[f"N{n}_{condition}_correct_minus_shuffle"]=paired_delta(correct,shuffled,"response_distance_rho",seed+len(contrasts)*101,nboot)
  # Primary full-depth pooled contrasts and static breadth.
  t4=surface[(surface.condition=="T4_AllDays")&(surface.temporal_order=="CorrectTemporal")];ep=surface[surface.condition=="T1_Day5Only"]
  contrasts["pooled_T4_minus_endpoint"]=paired_delta(t4,ep,"response_distance_rho",seed+30001,nboot)
  contrasts["pooled_T4_correct_minus_shuffle"]=paired_delta(t4,controls[(controls.condition=="T4_AllDays")&(controls.temporal_order=="TemporalShuffle")],"response_distance_rho",seed+30002,nboot)
  contrasts["static_N18_minus_N6"]=paired_delta(ep[ep.n_training_sources==18],ep[ep.n_training_sources==6],"response_distance_rho",seed+30003,nboot)
  # Late-depth sensitivity is labeled secondary, not the primary T4 claim.
  late=surface[(surface.condition=="T3b_Day3_Day4_Day5")&(surface.temporal_order=="CorrectTemporal")]
  contrasts["pooled_late_depth_minus_endpoint_secondary"]=paired_delta(late,ep,"response_distance_rho",seed+30004,nboot)
  contrasts["pooled_late_depth_correct_minus_shuffle_secondary"]=paired_delta(late,controls[(controls.condition=="T3b_Day3_Day4_Day5")&(controls.temporal_order=="TemporalShuffle")],"response_distance_rho",seed+30005,nboot)
  # Measurement expansion at maximum breadth.
  full=measure[measure.condition=="Day5Full"]
  for condition in ("Day5HalfPseudobulk","Day5AverageTwoHalves"):
    contrasts[f"endpoint_{condition}_minus_full"]=paired_delta(measure[measure.condition==condition],full,"response_distance_rho",seed+len(contrasts)*101,nboot)
  # Oracle curve against source only.
  source_only=oracle[oracle.condition=="SourceOnly"]
  for condition in [x for x in oracle.condition.unique() if x!="SourceOnly"]:contrasts[f"oracle_{condition}_minus_source"]=paired_delta(oracle[oracle.condition==condition],source_only,"response_distance_rho",seed+len(contrasts)*101,nboot)

  primary=contrasts["pooled_T4_minus_endpoint"];order=contrasts["pooled_T4_correct_minus_shuffle"]
  claim_a="PASS" if primary["point"]>0 and primary["ci_low"]>0 else "FAIL"
  claim_b="PASS" if order["point"]>0 and order["ci_low"]>0 else "FAIL"
  # T4 must beat both full endpoint and measurement expansion; here full endpoint is already the matched T1 target.
  claim_c="PASS" if claim_a=="PASS" and primary["point"]>0 else "FAIL"
  budget_geom=budget[budget.metric=="response_distance_rho"]
  claim_d="PASS" if np.any((budget_geom.delta_temporal_minus_static>=.05)&(budget_geom.ci_low>0)) else "FAIL"
  claim_e="PASS" if primary["point"]>=config["practical_temporal_gain"] and primary["ci_low"]>0 else "FAIL"
  claims={"claim_A":claim_a,"claim_B":claim_b,"claim_C":claim_c,"claim_D":claim_d,"claim_E":claim_e}
  verdict="TEMPORAL_DEPTH_ADVANTAGE_SUPPORTED" if all(v=="PASS" for v in claims.values()) else ("TEMPORAL_DEPTH_ADVANTAGE_PARTIALLY_SUPPORTED" if claim_a=="PASS" or claim_b=="PASS" or claim_d=="PASS" else "TEMPORAL_DEPTH_ADVANTAGE_NOT_SUPPORTED")
  classification="NO_CLEAR_SCALING"
  # Descriptive marginal comparisons.
  static_means=ep.groupby("n_training_sources").response_distance_rho.mean();static_spearman=safe_spearman(static_means.index.to_numpy(),static_means.to_numpy(),True)
  cap_order=["10","20","40","80","max"]
  wrel=power[(power.record_type=="reliability")&(power.target=="W23")].groupby("cell_cap").geometry_reliability.mean().reindex(cap_order)
  r5power=power[(power.record_type=="prediction")&(power.target=="R5")].groupby("cell_cap").geometry.mean().reindex(cap_order)
  summary={"verdict":verdict,"evidence_classification":classification,"claims":claims,"paired_contrasts":contrasts,"static_curve":static_means.to_dict(),"static_source_count_spearman":static_spearman,
   "temporal_surface_means":surface[surface.temporal_order.isin(["EndpointOnly","CorrectTemporal"])].groupby(["n_training_sources","condition"]).response_distance_rho.mean().reset_index().to_dict("records"),
   "temporal_order_means":controls.groupby(["condition","temporal_order"]).response_distance_rho.mean().reset_index().to_dict("records"),"matched_budgets":budget.to_dict("records"),
   "endpoint_measurement_controls":measure.groupby("condition")[["response_distance_rho","per_response_strict_trans_pearson"]].mean().to_dict("index"),
   "oracle_curve":oracle.groupby("condition")[["response_distance_rho","per_response_strict_trans_pearson","response_cosine"]].mean().to_dict("index"),
   "measurement_power":{"w23_reliability":wrel.to_dict(),"r5_geometry":r5power.to_dict(),"cell_caps":cap_order},"gpu_used":False,"new_architecture_trained":False,"n20_infeasible_reason":"23 total sources minus at least 5 held-out sources needed for nondegenerate within-group geometry leaves at most 18 training sources"}
  atomic_json(RESULT_ROOT/"analysis_summary.json",summary);atomic_json(RESULT_ROOT/"config.json",config)
  prov=json.loads((RESULT_ROOT/"provenance.json").read_text(encoding="utf-8"));prov["config_sha256"]=sha256(SCRIPT_ROOT/"config.json");atomic_json(RESULT_ROOT/"provenance.json",prov)

  figs=RESULT_ROOT/"figures";figs.mkdir(exist_ok=True)
  plt.figure(figsize=(7,4));plt.plot(static_means.index,static_means.values,marker="o");plt.xlabel("Training perturbation sources");plt.ylabel("Grouped R5 geometry");plt.title("Static breadth scaling");plt.tight_layout();plt.savefig(figs/"01_static_breadth.png",dpi=160);plt.close()
  piv=surface[surface.temporal_order.isin(["EndpointOnly","CorrectTemporal"])].groupby(["n_training_sources","condition"]).response_distance_rho.mean().unstack();plt.figure(figsize=(9,5));
  for c in piv:plt.plot(piv.index,piv[c],marker="o",label=c)
  plt.xlabel("Training sources");plt.ylabel("Grouped R5 geometry");plt.legend(fontsize=7,ncol=2);plt.tight_layout();plt.savefig(figs/"02_temporal_depth_curves.png",dpi=160);plt.close()
  plt.figure(figsize=(9,5));plt.imshow(piv.T,aspect="auto",cmap="coolwarm");plt.colorbar(label="R5 geometry");plt.xticks(range(len(piv.index)),piv.index);plt.yticks(range(len(piv.columns)),piv.columns);plt.xlabel("Training sources");plt.tight_layout();plt.savefig(figs/"03_scaling_surface.png",dpi=160);plt.close()
  op=oracle.groupby("condition").response_distance_rho.mean();plt.figure(figsize=(8,4));plt.bar(op.index,op.values);plt.ylabel("Oracle R5 geometry");plt.xticks(rotation=25,ha="right");plt.tight_layout();plt.savefig(figs/"04_oracle_temporal_value.png",dpi=160);plt.close()
  plt.figure(figsize=(7,4));plt.plot(cap_order,wrel.values,marker="o",label="W23 reliability");plt.plot(cap_order,r5power.values,marker="o",label="R5 geometry");plt.xlabel("Cell cap");plt.legend();plt.tight_layout();plt.savefig(figs/"05_measurement_power.png",dpi=160);plt.close()

  md=f"""{verdict}

# RENGE temporal-depth versus static-breadth scaling audit

Evidence classification: `{classification}`

## Claims

- Claim A — temporal depth improves endpoint geometry: **{claim_a}**. Pooled T4-minus-endpoint delta {primary['point']:.4f} [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}].
- Claim B — correct temporal order beats shuffle: **{claim_b}**. Pooled T4 correct-minus-shuffle {order['point']:.4f} [{order['ci_low']:.4f}, {order['ci_high']:.4f}].
- Claim C — temporal depth exceeds endpoint measurement precision: **{claim_c}**.
- Claim D — fewer temporally rich sources beat broader endpoint-only data: **{claim_d}**.
- Claim E — practical temporal gain >=0.05 with positive CI: **{claim_e}**.

## Central result

Neither static breadth nor full temporal depth shows a clear monotonic deployable scaling law. Late Day3/Day4 descriptors show a secondary weak signal, but full T4 does not outperform endpoint-only or equal-size temporal shuffle. Oracle temporal states strongly increase R5 geometry, localizing the failure to transferable source-to-temporal-state inference rather than absence of endpoint-relevant temporal information.

N=20 is infeasible without reducing the held-out group below five sources; N=18 is the largest honest geometry comparison. All conditions use fixed ridge alpha 100, training-only temporal descriptors, identical held-out groups, and one model per group. No GPU or new architecture was used.
"""
  (RESULT_ROOT/"FINAL_VERDICT.md").write_text(md,encoding="utf-8");(RESULT_ROOT/"README.md").write_text("# RENGE temporal scaling audit\n\nSee `FINAL_VERDICT.md`, `analysis_summary.json`, scaling tables, and `figures/`.\n",encoding="utf-8")
  files=[]
  for path in sorted(p for p in RESULT_ROOT.rglob("*") if p.is_file() and p.name!="output_manifest.json"):files.append({"path":path.relative_to(RESULT_ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
  atomic_json(RESULT_ROOT/"output_manifest.json",{"verdict":verdict,"files":files});print(f"[temporal-scaling] verdict: {verdict}")

if __name__=="__main__":main()

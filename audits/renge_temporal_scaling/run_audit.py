from __future__ import annotations
import json,platform
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scaling_common import *

def now():return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def row(split,subset_index,n,condition,order,pred,truth,sources,genes,k,extra=None):
  test=split["test"];m=complete_metrics(pred,truth,sources[test],genes,k);mu=(extra or {}).pop("training_mean",None)
  if mu is not None:m["residual_geometry"]=geometry_rho(pred-mu,truth-mu,direct_indices(sources[test],genes))
  return {"repeat":split["repeat"],"outer_seed":split["seed"],"subset_index":subset_index,"n_training_sources":n,"condition":condition,"temporal_order":order,
   "n_test_sources":len(test),"one_model_for_entire_heldout_group":True,"heldout_source_absent_all_days":True,"outer_test_used_for_selection":False,**m,**(extra or {})}

def sampled_states(expression,assignment,times,sources,cap,seed):
  rng=np.random.default_rng(seed);halves=[];realized=[]
  for half in range(2):
    days=[]
    for day in (2,3,4,5):
      cr=np.flatnonzero((assignment=="control")&(times==day));cn=len(cr) if cap=="max" else min(int(cap),len(cr));control=expression[rng.choice(cr,cn,replace=True)].mean(0)
      values=[]
      for source in sources:
        rows=np.flatnonzero((assignment==source)&(times==day));nn=len(rows) if cap=="max" else min(int(cap),len(rows));values.append(expression[rng.choice(rows,nn,replace=True)].mean(0)-control);realized.append(nn)
      days.append(np.asarray(values))
    halves.append(np.stack(days,axis=1))
  return halves[0].astype(np.float32),halves[1].astype(np.float32),float(np.mean(realized))

def main():
  RESULT_ROOT.mkdir(parents=True,exist_ok=True);CACHE_ROOT.mkdir(parents=True,exist_ok=True);config=json.loads((SCRIPT_ROOT/"config.json").read_text(encoding="utf-8"))
  req=["static_breadth_scaling.csv","temporal_depth_scaling.csv","temporal_order_controls.csv","endpoint_measurement_controls.csv","oracle_temporal_value.csv","temporal_static_scaling_surface.csv","measurement_power_scaling.csv","noise_ceiling_analysis.csv"]
  replay=RESULT_ROOT/"cache_replay_enabled.json"
  if all((RESULT_ROOT/f).exists() for f in req) and replay.exists():print("[temporal-scaling] all scientific tables already cached");return
  with np.load(FROZEN_CACHE,allow_pickle=False) as z:
    expression=z["expression"].astype(np.float32);assignment=z["assignment"].astype(str);times=z["times"].astype(int);genes=z["genes"].astype(str);sources=z["sources"].astype(str)
    states=z["response"].astype(np.float32);waves=z["waves"].astype(np.float32);static=z["static_control_representation"].astype(np.float32)
  with np.load(PSEUDO_CACHE,allow_pickle=False) as z:pseudo_a=z["half_a"].astype(np.float32);pseudo_b=z["half_b"].astype(np.float32)
  lookup={g:i for i,g in enumerate(genes)};source_rows=np.asarray([lookup[s] for s in sources]);splits=outer_splits(len(sources),config["outer_repeats"],config["outer_split_seed"],config["heldout_sources_per_repeat"])
  alpha=config["fixed_ridge_alpha"];k=config["local_neighbors"];surface=[];controls=[];measure_controls=[];oracle_rows=[];audit=[]
  conditions=config["temporal_conditions"]
  for oi,split in enumerate(splits):
    test,pool=split["test"],split["pool"];audit.append({"repeat":split["repeat"],"seed":split["seed"],"train_pool":sources[pool].tolist(),"test":sources[test].tolist()})
    for n in config["training_source_counts"]:
      subsets=training_subsets(pool,n,1 if n==len(pool) else config["subsets_per_count"],config["subset_seed"]+oi*100003+n*1009)
      for si,train in enumerate(subsets):
        for ci,(condition,transitions) in enumerate(conditions.items()):
          if not transitions:
            feat=static[source_rows];pred=fixed_ridge(feat[train],states[train,3],feat[test],alpha)
            surface.append(row(split,si,n,condition,"EndpointOnly",pred,states[test,3],sources,genes,k,{"training_mean":states[train,3].mean(0),"source_time_observations":n}))
          else:
            for shuffled in (False,True):
              desc=temporal_descriptor(states,train,transitions,shuffled,split["seed"]+si*7919+ci*101);feat=desc[source_rows]
              pred=fixed_ridge(feat[train],states[train,3],feat[test],alpha);order="TemporalShuffle" if shuffled else "CorrectTemporal"
              observed_days=set()
              for a,b in transitions: observed_days.update((a,b))
              record=row(split,si,n,condition,order,pred,states[test,3],sources,genes,k,{"training_mean":states[train,3].mean(0),"source_time_observations":n*len(observed_days)})
              surface.append(record);controls.append(record.copy())
        # Endpoint-cell measurement expansion control at maximum source breadth only.
        if n==max(config["training_source_counts"]):
          feat=static[source_rows];half=pseudo_a[split["repeat"]%len(pseudo_a),3];avg=(pseudo_a[split["repeat"]%len(pseudo_a),3]+pseudo_b[split["repeat"]%len(pseudo_b),3])/2
          for label,target in (("Day5HalfPseudobulk",half),("Day5AverageTwoHalves",avg),("Day5Full",states[:,3])):
            pred=fixed_ridge(feat[train],target[train],feat[test],alpha);measure_controls.append(row(split,si,n,label,"EndpointOnly",pred,states[test,3],sources,genes,k,{"training_mean":target[train].mean(0)}))
        # Oracle temporal value at fixed N=12 and first three deterministic subsets.
        if n==12 and si<3:
          oracle_features={"SourceOnly":static[source_rows],"TrueR2":states[:,0],"TrueR3":states[:,1],"TrueR4":states[:,2],"TrueR2_R3":np.concatenate([states[:,0],states[:,1]],1),"TrueR2_R3_R4":np.concatenate([states[:,:3,:].reshape(len(states),-1)],1)}
          for name,feat in oracle_features.items():
            pred=fixed_ridge(feat[train],states[train,3],feat[test],alpha);oracle_rows.append(row(split,si,n,name,"OracleDiagnostic",pred,states[test,3],sources,genes,k,{"training_mean":states[train,3].mean(0),"oracle":name!="SourceOnly"}))
    if (oi+1)%10==0:print(f"[temporal-scaling] breadth-depth surface {oi+1}/{len(splits)}",flush=True)
  surface=pd.DataFrame(surface);surface.to_csv(RESULT_ROOT/"temporal_static_scaling_surface.csv",index=False)
  surface[(surface.condition=="T1_Day5Only")].to_csv(RESULT_ROOT/"static_breadth_scaling.csv",index=False)
  surface[surface.temporal_order.isin(["EndpointOnly","CorrectTemporal"])].to_csv(RESULT_ROOT/"temporal_depth_scaling.csv",index=False)
  pd.DataFrame(controls).to_csv(RESULT_ROOT/"temporal_order_controls.csv",index=False);pd.DataFrame(measure_controls).to_csv(RESULT_ROOT/"endpoint_measurement_controls.csv",index=False);pd.DataFrame(oracle_rows).to_csv(RESULT_ROOT/"oracle_temporal_value.csv",index=False)

  # Measurement-power audit: bootstrap pseudobulks at fixed caps, then frozen CorrectLag wave chain.
  power=[];noise_rows=[];twofold=grouped_twofold_splits(len(sources),config["measurement_outer_repeats"],config["outer_split_seed"])
  for cap_index,cap in enumerate(config["measurement_cell_caps"]):
    for rep in range(config["measurement_bootstrap_repeats"]):
      a,b,realized=sampled_states(expression,assignment,times,sources,cap,config["measurement_seed"]+cap_index*100003+rep*1009);mean=(a+b)/2;ww=np.diff(mean,axis=1)
      for target,ai,bi in (("R2",0,0),("R3",1,1),("W23",0,0)):
        aa=a[:,ai] if target!="W23" else a[:,1]-a[:,0];bb=b[:,bi] if target!="W23" else b[:,1]-b[:,0]
        rel=geometry_rho(aa,bb,direct_indices(sources,genes));power.append({"record_type":"reliability","cell_cap":str(cap),"bootstrap_repeat":rep,"outer_repeat":-1,"target":target,"realized_mean_cells":realized,"geometry_reliability":rel})
      for split in twofold:
        train,test=split["train"],split["test"];feat=representation("CorrectLag",ww,mean,static,sources,genes,source_rows,train,split["seed"]);pred_w23=fixed_ridge(feat[train],ww[train,0],feat[test],alpha)
        transition,_=fit_dense_transition(ww,train,split["seed"],(alpha,));pred_w34=apply_affine(transition,pred_w23);pred_w45=apply_affine(transition,pred_w34);pred_r2=fixed_ridge(feat[train],mean[train,0],feat[test],alpha);pred_r5=pred_r2+pred_w23+pred_w34+pred_w45
        power.append({"record_type":"prediction","cell_cap":str(cap),"bootstrap_repeat":rep,"outer_repeat":split["repeat"],"target":"W23","realized_mean_cells":realized,"geometry":geometry_rho(pred_w23,waves[test,0],direct_indices(sources[test],genes))})
        power.append({"record_type":"prediction","cell_cap":str(cap),"bootstrap_repeat":rep,"outer_repeat":split["repeat"],"target":"R5","realized_mean_cells":realized,"geometry":geometry_rho(pred_r5,states[test,3],direct_indices(sources[test],genes))})
  power=pd.DataFrame(power);power.to_csv(RESULT_ROOT/"measurement_power_scaling.csv",index=False)
  for cap in power.cell_cap.unique():
    wrel=power[(power.cell_cap==cap)&(power.record_type=="reliability")&(power.target=="W23")].geometry_reliability.mean();obs=power[(power.cell_cap==cap)&(power.record_type=="prediction")&(power.target=="W23")].geometry.mean()
    noise_rows.append({"cell_cap":cap,"w23_reliability":wrel,"observed_w23_geometry":obs,"attenuation_ceiling_sqrt_reliability":float(np.sqrt(max(wrel,0))),"observed_fraction_of_diagnostic_ceiling":obs/max(np.sqrt(max(wrel,0)),1e-12),"manuscript_performance_metric":False})
  pd.DataFrame(noise_rows).to_csv(RESULT_ROOT/"noise_ceiling_analysis.csv",index=False)
  atomic_json(RESULT_ROOT/"split_audit.json",{"created_at":now(),"outer_groups":len(splits),"heldout_sources_per_group":config["heldout_sources_per_repeat"],"heldout_source_absent_all_times":True,"one_model_per_heldout_group":True,"outer_test_used_for_fitting_or_selection":False,"splits":audit})
  atomic_json(RESULT_ROOT/"provenance.json",{"created_at":now(),"input":str(FROZEN_CACHE.relative_to(ROOT)),"input_sha256":sha256(FROZEN_CACHE),"pseudoreplicate_input":str(PSEUDO_CACHE.relative_to(ROOT)),"pseudoreplicate_sha256":sha256(PSEUDO_CACHE),"config_sha256":sha256(SCRIPT_ROOT/"config.json"),"python":platform.python_version(),"cells":len(expression),"sources":len(sources),"genes":len(genes),"gpu_used":False,"previous_results_read_only":["propagation_reproduction","renge_dynamic_validity","renge_first_wave_program","renge_program_identifiability","renge_endpoint_benchmark","renge_state_vs_wave","renge_identity_denoising"]})
  atomic_json(replay,{"enabled":True,"scientific_tables_complete":True});atomic_json(RESULT_ROOT/"run_complete.json",{"completed_at":now(),"gpu_used":False,"new_architecture_trained":False})
  print("[temporal-scaling] all scientific tables complete",flush=True)

if __name__=="__main__":main()

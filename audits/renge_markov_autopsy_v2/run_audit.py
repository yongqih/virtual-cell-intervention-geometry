from __future__ import annotations
import json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from autopsy_common import *

def now():return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def cosine(a,b):
  x=np.asarray(a).ravel();y=np.asarray(b).ravel();return float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
def compression_extra(pred,truth,names,genes):
  direct=direct_indices(names,genes);p=strict_trans_cosine_distance(pred,direct);t=strict_trans_cosine_distance(truth,direct);u=np.triu_indices(len(pred),1)
  return {"distance_scale_retention":float(np.mean(p[u])/max(np.mean(t[u]),1e-12))}
def residual_geometry(pred,truth,reference,names,genes):
  return geometry_rho(np.asarray(pred)-reference,np.asarray(truth)-reference,direct_indices(names,genes))
def generic_row(split,model,scope,stage,pred,truth,sources,genes,k,extra=None,residual_reference=None):
  added=compression_extra(pred,truth,sources[split["test"]],genes)|(extra or {})
  if residual_reference is not None:added["residual_geometry_rho"]=residual_geometry(pred,truth,residual_reference,sources[split["test"]],genes)
  return group_metrics(split,model,scope,stage,pred,truth,sources,genes,k,added)

def previous_audit():
  value={"created_at":now(),"question_after_first_state_was_source_passed_to_every_later_transition":"NO",
   "overall_answer":"MIXED","recursive_chain_answer":"YES_INITIALIZATION_ONLY",
   "models":[
    {"model":"CorrectLag dynamic-validity wave chain","classification":"INITIALIZATION_ONLY","file":"scripts/renge_dynamic_validity/run_diagnostics.py","evidence_lines":[217,221],"source_entry":"CorrectLag predicts W23","later_transition_signature":"propagate(state)=apply_affine(transition,state)","source_passed_after_initialization":False},
    {"model":"Endpoint benchmark frozen wave chain","classification":"INITIALIZATION_ONLY","file":"scripts/renge_endpoint_benchmark/run_audit.py","evidence_lines":[72,77],"source_entry":"CorrectLag predicts W23/R2","later_transition_signature":"apply_affine(transition,w23); apply_affine(transition,w34)","source_passed_after_initialization":False},
    {"model":"State-vs-wave absolute state chain","classification":"INITIALIZATION_ONLY","file":"scripts/renge_state_vs_wave/run_audit.py","evidence_lines":[146,149],"source_entry":"CorrectLag predicts R2","later_transition_signature":"state_rollout(state_model,pred_r2,3)","source_passed_after_initialization":False},
    {"model":"Temporal-scaling descriptor-to-R5","classification":"OTHER","file":"scripts/renge_temporal_scaling/run_audit.py","evidence_lines":[51],"source_entry":"temporal descriptor directly predicts R5","later_transition_signature":"non-recursive","source_passed_after_initialization":None},
    {"model":"Temporal-scaling measurement-power wave chain","classification":"INITIALIZATION_ONLY","file":"scripts/renge_temporal_scaling/run_audit.py","evidence_lines":[83],"source_entry":"CorrectLag predicts W23/R2","later_transition_signature":"apply_affine transition without A","source_passed_after_initialization":False}],
   "interpretation":"All previously implemented recursive chains treat the unseen-source descriptor as an initializer; none passes it into every later transition. Non-recursive temporal-scaling models are outside that classification."}
  atomic_json(RESULT_ROOT/"previous_propagator_source_audit.json",value)

def main():
  RESULT_ROOT.mkdir(parents=True,exist_ok=True);CACHE_ROOT.mkdir(parents=True,exist_ok=True);previous_audit();config=json.loads((SCRIPT_ROOT/"config.json").read_text(encoding="utf-8"))
  required=["core_impulse_vs_persistent_forcing.csv","oracle_markov_sufficiency.csv","residual_source_memory.csv","oracle_impulse_vs_persistent_rollout.csv","deployable_persistent_forcing_endpoint.csv","forcing_x_decoder_factorial.csv","recursive_geometry_decay.csv","transition_stationarity.csv","history_after_source_conditioning.csv","trajectory_signature_comparison.csv","population_state_sufficiency.csv","geometry_compression_autopsy.csv"]
  replay=RESULT_ROOT/"cache_replay_enabled.json"
  revision="residual-geometry-source-bootstrap-v3"
  if all((RESULT_ROOT/x).exists() for x in required) and replay.exists() and json.loads(replay.read_text(encoding="utf-8")).get("revision")==revision:print("[markov-v2] all scientific tables already cached");return
  with np.load(FROZEN_CACHE,allow_pickle=False) as z:
    expression=z["expression"].astype(np.float32);assignment=z["assignment"].astype(str);times=z["times"].astype(int);genes=z["genes"].astype(str);sources=z["sources"].astype(str);states=z["response"].astype(np.float32);waves=z["waves"].astype(np.float32);static=z["static_control_representation"].astype(np.float32);cell_count=z["cell_count"].astype(int)
  lookup={g:i for i,g in enumerate(genes)};source_gene_rows=np.asarray([lookup[s] for s in sources]);splits=grouped_twofold_splits(len(sources),config["source_disjoint_repeats"],config["outer_split_seed"]);alpha=config["fixed_ridge_alpha"];rank=config["interaction_rank"];k=config["local_neighbors"]
  core=[];suff=[];memory=[];oracle_roll=[];deploy=[];factorial=[];decay=[];station=[];history=[];trajectory=[];compression=[];source_errors=[];audit=[]
  for si,split in enumerate(splits):
    train,test=split["train"],split["test"];names=sources[test];audit.append({"repeat":split["repeat"],"group":split["group"],"seed":split["seed"],"train":sources[train].tolist(),"test":sources[test].tolist()})
    A=representation("CorrectLag",waves,states,static,sources,genes,source_gene_rows,train,split["seed"])
    impulse=fit_impulse(states,train,alpha);additive=fit_additive(states,A,train,alpha);conditional=fit_conditional(states,A,train,alpha,rank);models={"Impulse":impulse,"PersistentAdditive":additive,"PersistentConditional":conditional}
    pred_r2=apply(fit_ridge(A[train],states[train,0],alpha),A[test]);direct_r5=apply(fit_ridge(A[train],states[train,3],alpha),A[test])
    deploy_pred={"DirectEndpoint":direct_r5}
    for name,model in models.items():deploy_pred[name]=rollout(model,pred_r2,A[test],3)[-1]
    for name,pred in deploy_pred.items():
      rr=generic_row(split,name,"deployable","R5",pred,states[test,3],sources,genes,k,residual_reference=states[train,3].mean(0));deploy.append(rr);core.append(rr.copy());compression.append(rr|{"formulation":name})

    # One-step current-state sufficiency and history after current state.
    for stage_index,(current,next_state,earlier,label) in enumerate(((2,3,1,"R3_to_R4"),(3,4,2,"R4_to_R5"))):
      t=current-1;target=next_state-1;prev=earlier-1;x=states[train,t];y=states[train,target];xt=states[test,t];yt=states[test,target]
      candidates={"CurrentStateOnly":apply(fit_ridge(x,y,alpha),xt),
       "CurrentPlusA_Additive":step(fit_additive_xy(x,y,A[train],alpha),xt,A[test]),
       "CurrentPlusA_Conditional":step(fit_conditional_xy(x,y,A[train],alpha,rank),xt,A[test]),
       "HistoryCurrent":apply(fit_ridge(np.concatenate([states[train,prev],x],1),y,alpha),np.concatenate([states[test,prev],xt],1)),
       "HistoryCurrentA":apply(fit_ridge(np.concatenate([states[train,prev],x,A[train]],1),y,alpha),np.concatenate([states[test,prev],xt,A[test]],1))}
      for name,pred in candidates.items():suff.append(generic_row(split,name,"oracle_one_step",label,pred,yt,sources,genes,k))
      # Residual source-memory test.
      f=fit_ridge(x,y,alpha);train_res=y-apply(f,x);test_res=yt-apply(f,xt);features={"Zero":None,"A":A,"EarlierState":states[:,prev],"A_EarlierState":np.concatenate([A,states[:,prev]],1)}
      perm=np.random.default_rng(split["seed"]+stage_index*991).permutation(len(train));features["A_PermutationControl"]=(A[train][perm],A[test])
      for name,feat in features.items():
        if feat is None:pred=np.zeros_like(test_res)
        elif isinstance(feat,tuple):pred=apply(fit_ridge(feat[0],train_res,alpha),feat[1])
        else:pred=apply(fit_ridge(feat[train],train_res,alpha),feat[test])
        met=residual_metrics(pred,test_res,names,genes);memory.append({"record_type":"group","repeat":split["repeat"],"group":split["group"],"stage":label,"model":name,"n_train_sources":len(train),"n_test_sources":len(test),**met})
        for source_met in source_rows(pred,test_res,names,genes):
          memory.append({"record_type":"source","repeat":split["repeat"],"group":split["group"],"stage":label,"model":name,"n_train_sources":len(train),"n_test_sources":len(test),**source_met})
      # History after A supplied.
      base=apply(fit_ridge(np.concatenate([x,A[train]],1),y,alpha),np.concatenate([xt,A[test]],1));hist=apply(fit_ridge(np.concatenate([states[train,prev],x,A[train]],1),y,alpha),np.concatenate([states[test,prev],xt,A[test]],1))
      history.extend([generic_row(split,"CurrentPlusA","oracle_one_step",label,base,yt,sources,genes,k),generic_row(split,"HistoryCurrentPlusA","oracle_one_step",label,hist,yt,sources,genes,k)])
      # Store source errors for population-state associations.
      source_metric=source_rows(candidates["CurrentStateOnly"],yt,names,genes)
      for j,source in enumerate(names):source_errors.append({"repeat":split["repeat"],"group":split["group"],"source":source,"transition":label,"transition_mse":source_metric[j]["mse"],"transition_pearson":source_metric[j]["response_pearson"]})

    # Oracle rollout from identical true R2 and R3.
    for entry_index,steps,label in ((0,3,"TrueR2"),(1,2,"TrueR3")):
      for name,model in models.items():
        pred=rollout(model,states[test,entry_index],A[test],steps)[-1];rr=generic_row(split,name,"oracle_rollout",label+"_to_R5",pred,states[test,3],sources,genes,k);oracle_roll.append(rr)
    # Recursive geometry decay from true entries.
    for entry_index,steps,label in ((0,3,"TrueR2"),(1,2,"TrueR3")):
      for name,model in models.items():
        outs=rollout(model,states[test,entry_index],A[test],steps)
        for offset,pred in enumerate(outs,1):decay.append(generic_row(split,name,"oracle_recursive_decay",label,f"",sources,genes,k) if False else generic_row(split,name,"oracle_recursive_decay",f"R{entry_index+2+offset}",pred,states[test,entry_index+offset],sources,genes,k,{"entry":label,"step":offset}))

    # Direct-horizon x recursive factorial with identical true information.
    for entry_index,steps,label in ((0,3,"TrueR2"),(1,2,"TrueR3")):
      entry=states[:,entry_index]
      direct_state=apply(fit_ridge(entry[train],states[train,3],alpha),entry[test]);direct_source=apply(fit_ridge(np.concatenate([entry[train],A[train]],1),states[train,3],alpha),np.concatenate([entry[test],A[test]],1))
      rec_state=rollout(impulse,entry[test],A[test],steps)[-1];rec_source=rollout(additive,entry[test],A[test],steps)[-1]
      for forcing,decoder,pred in (("StateOnly","DirectHorizon",direct_state),("SourceConditioned","DirectHorizon",direct_source),("StateOnly","Recursive",rec_state),("SourceConditioned","Recursive",rec_source)):
        rr=generic_row(split,forcing+"_"+decoder,"oracle_factorial",label,pred,states[test,3],sources,genes,k,{"forcing":forcing,"decoder":decoder,"entry":label},residual_reference=states[train,3].mean(0));factorial.append(rr)
        if decoder=="DirectHorizon":compression.append(rr|{"formulation":f"DirectHorizon_{forcing}_{label}"})

    # Stationarity: shared, stage-specific, time-conditioned and cross-interval transfer.
    px,py=pooled_xy(states,train);shared=fit_ridge(px,py,alpha);stage_models=[fit_ridge(states[train,t],states[train,t+1],alpha) for t in range(3)]
    onehot=np.repeat(np.eye(3),len(train),axis=0);stage_x=np.concatenate([states[train,0],states[train,1],states[train,2]]);time_feat=np.concatenate([stage_x,onehot,np.concatenate([states[train,t]*np.eye(3)[t][j] for t in range(3) for j in range(3)],axis=0).reshape(9*len(train),-1)[:3*len(train)]],1) if False else None
    # Explicit state x time blocks.
    blocks=[]
    for t in range(3):blocks.append(np.concatenate([states[train,t],np.tile(np.eye(3)[t],(len(train),1)),states[train,t]*np.eye(3)[t,0],states[train,t]*np.eye(3)[t,1],states[train,t]*np.eye(3)[t,2]],1))
    time_model=fit_ridge(np.concatenate(blocks),py,alpha)
    for t,label in enumerate(("R2_to_R3","R3_to_R4","R4_to_R5")):
      xt=states[test,t];yt=states[test,t+1];oh=np.tile(np.eye(3)[t],(len(test),1));tf=np.concatenate([xt,oh,xt*np.eye(3)[t,0],xt*np.eye(3)[t,1],xt*np.eye(3)[t,2]],1)
      for name,pred in (("SharedF",apply(shared,xt)),("StageSpecificF",apply(stage_models[t],xt)),("TimeConditionalF",apply(time_model,tf))):station.append(generic_row(split,name,"oracle_one_step",label,pred,yt,sources,genes,k))
      for fit_t in range(3):
        if fit_t!=t:station.append(generic_row(split,f"CrossTransfer_F{fit_t+2}{fit_t+3}","cross_interval",label,apply(stage_models[fit_t],xt),yt,sources,genes,k,{"fit_interval":fit_t,"test_interval":t}))
    for a in range(3):
      for b in range(a+1,3):station.append({"repeat":split["repeat"],"group":split["group"],"model":"OperatorSimilarity","scope":"coefficient","stage":f"F{a+2}{a+3}_vs_F{b+2}{b+3}","coefficient_cosine":cosine(stage_models[a]["coef"],stage_models[b]["coef"]),"subspace_similarity":float(np.mean(np.linalg.svd(np.linalg.qr(stage_models[a]["coef"])[0][:,:rank].T@np.linalg.qr(stage_models[b]["coef"])[0][:,:rank],compute_uv=False)**2))})

    # Trajectory signature oracle comparison.
    tr=np.concatenate([states[train,0],states[train,1],states[train,2]],1);te=np.concatenate([states[test,0],states[test,1],states[test,2]],1);tm=tr.mean(0);_,_,vt=np.linalg.svd(tr-tm,full_matrices=False);pc=vt[:8].T
    traj_models={"R4Only":apply(fit_ridge(states[train,2],states[train,3],alpha),states[test,2]),"R4PlusA":apply(fit_ridge(np.concatenate([states[train,2],A[train]],1),states[train,3],alpha),np.concatenate([states[test,2],A[test]],1)),"WholeTrajectory":apply(fit_ridge(tr,states[train,3],alpha),te),"TrajectoryPCA8":apply(fit_ridge((tr-tm)@pc,states[train,3],alpha),(te-tm)@pc),"RecursivePersistentFromTrueR2":rollout(additive,states[test,0],A[test],3)[-1]}
    for name,pred in traj_models.items():
      rr=generic_row(split,name,"oracle_trajectory", "R5",pred,states[test,3],sources,genes,k,residual_reference=states[train,3].mean(0));trajectory.append(rr)
      if name in ("WholeTrajectory","TrajectoryPCA8"):compression.append(rr|{"formulation":name})
    # Deployable history-conditioned geometry autopsy.
    pred_r3=apply(fit_ridge(A[train],states[train,1],alpha),A[test]);h4=apply(fit_ridge(np.concatenate([states[train,0],states[train,1],A[train]],1),states[train,2],alpha),np.concatenate([pred_r2,pred_r3,A[test]],1));h5=apply(fit_ridge(np.concatenate([states[train,1],states[train,2],A[train]],1),states[train,3],alpha),np.concatenate([pred_r3,h4,A[test]],1));rr=generic_row(split,"HistoryConditioned","deployable","R5",h5,states[test,3],sources,genes,k,residual_reference=states[train,3].mean(0));compression.append(rr|{"formulation":"HistoryConditioned"})
    atomic_npz(CACHE_ROOT/f"split_{si:03d}.npz",test=test,direct=direct_r5,impulse=deploy_pred["Impulse"],additive=deploy_pred["PersistentAdditive"],conditional=deploy_pred["PersistentConditional"],history=h5)
    if (si+1)%10==0:print(f"[markov-v2] grouped autopsy {si+1}/{len(splits)}",flush=True)

  # Population snapshot diagnostics and transition-error association.
  error=pd.DataFrame(source_errors).groupby(["source","transition"])[["transition_mse","transition_pearson"]].mean().reset_index();pop=[]
  for source in sources:
    for day_index,day in enumerate((2,3,4,5)):
      rows=np.flatnonzero((assignment==source)&(times==day));x=expression[rows];center=x-x.mean(0);sing=np.linalg.svd(center,compute_uv=False);var=sing**2/max(len(x)-1,1);w=var/max(var.sum(),1e-12)
      record={"source":source,"day":day,"n_cells":len(x),"mean_within_gene_variance":float(np.mean(np.var(x,axis=0,ddof=1))),"heterogeneity_total_variance":float(np.sum(np.var(x,axis=0,ddof=1))),"dispersion_pc1_fraction":float(w[0]),"dispersion_effective_rank":float(1/max(np.sum(w*w),1e-12)),"pseudobulk_standard_error":float(np.sqrt(np.mean(np.var(x,axis=0,ddof=1))/len(x)))}
      if day<5:
        nxt=expression[np.flatnonzero((assignment==source)&(times==day+1))];record["distributional_mean_shift"] = float(np.linalg.norm(nxt.mean(0)-x.mean(0)));record["distributional_variance_shift"] = float(np.mean(np.abs(np.var(nxt,axis=0)-np.var(x,axis=0))))
        label=f"R{day}_to_R{day+1}";er=error[(error.source==source)&(error.transition==label)]
        if len(er):record.update(er.iloc[0][["transition_mse","transition_pearson"]].to_dict())
      pop.append(record)
  popdf=pd.DataFrame(pop)
  for metric in ("distributional_mean_shift","distributional_variance_shift","mean_within_gene_variance","pseudobulk_standard_error"):
    valid=popdf[[metric,"transition_mse"]].dropna();pop.append({"source":"ALL_ASSOCIATION","day":-1,"association_metric":metric,"association_with_transition_mse_spearman":safe_spearman(valid[metric],valid.transition_mse,True),"n_source_transitions":len(valid)})
  pd.DataFrame(pop).to_csv(RESULT_ROOT/"population_state_sufficiency.csv",index=False)
  tables={"core_impulse_vs_persistent_forcing.csv":core,"oracle_markov_sufficiency.csv":suff,"residual_source_memory.csv":memory,"oracle_impulse_vs_persistent_rollout.csv":oracle_roll,"deployable_persistent_forcing_endpoint.csv":deploy,"forcing_x_decoder_factorial.csv":factorial,"recursive_geometry_decay.csv":decay,"transition_stationarity.csv":station,"history_after_source_conditioning.csv":history,"trajectory_signature_comparison.csv":trajectory,"geometry_compression_autopsy.csv":compression}
  for name,rows in tables.items():pd.DataFrame(rows).to_csv(RESULT_ROOT/name,index=False)
  atomic_json(RESULT_ROOT/"split_audit.json",{"created_at":now(),"groups":len(splits),"sources":len(sources),"heldout_source_absent_all_times":True,"one_model_per_heldout_group":True,"outer_test_used_for_fitting_or_selection":False,"splits":audit})
  atomic_json(RESULT_ROOT/"provenance.json",{"created_at":now(),"input":str(FROZEN_CACHE.relative_to(ROOT)),"input_sha256":sha256(FROZEN_CACHE),"config_sha256":sha256(SCRIPT_ROOT/"config.json"),"python":platform.python_version(),"cells":len(expression),"sources":len(sources),"genes":len(genes),"gpu_used":False,"previous_results_read_only":["propagation_reproduction","renge_dynamic_validity","renge_first_wave_program","renge_program_identifiability","renge_endpoint_benchmark","renge_state_vs_wave","renge_identity_denoising","renge_temporal_scaling"]})
  atomic_json(replay,{"enabled":True,"scientific_tables_complete":True,"revision":revision});atomic_json(RESULT_ROOT/"run_complete.json",{"completed_at":now(),"gpu_used":False,"new_architecture_trained":False})
  print("[markov-v2] all scientific tables complete",flush=True)

if __name__=="__main__":main()

"""SMILES-only source-aware GP ensemble. No concentration column enters fitting."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import vacv_core as V

SCALE=1.6114604791188905  # Reproduces the user's successful v14 experimental CSV.
EXTERNAL_WEIGHT=.25
SOURCE_BIAS=1.0
SHRINK=.05

def folds(groups,k):
    k=min(k,len(np.unique(groups)))
    if k<2:raise ValueError('Not enough independent groups')
    return list(GroupKFold(k).split(np.zeros(len(groups)),groups=groups))

def external_for_fold(ext,tr,K,a,blocked,scheme):
    mask=~ext.SMILES.isin(set(tr.iloc[a].c)|set(tr.iloc[blocked].c)).to_numpy()
    if len(blocked) and scheme=='scaffold':
        mask &= ~ext.scaf.isin(set(tr.iloc[blocked].scaf)).to_numpy()
    if len(blocked) and scheme=='cluster065':
        sim=K['mc2'][np.ix_(len(tr)+np.arange(len(ext)),blocked)]
        mask &= sim.max(axis=1)<.65
    return np.where(mask)[0]

def posterior(K,y,ext,a,b,e,source_bias=True):
    labels=np.r_[y[a],ext.SI.to_numpy()[e]];ids=np.r_[a,len(y)+e]
    z=np.log1p(labels);prior=np.log1p(y[a]);mean=prior.mean();amp=prior.var()+1e-12
    c=np.zeros((len(ids),len(ids)))
    if len(e) and source_bias:
        sources=ext.document_id.to_numpy()[e]
        c[len(a):,len(a):]=SOURCE_BIAS*(sources[:,None]==sources[None,:])
    out=[]
    for name in V.GP_KERNELS:
        for alpha in V.GP_ALPHAS:
            noise=np.r_[np.full(len(a),alpha),np.full(len(e),alpha/EXTERNAL_WEIGHT)]
            kt=K[name][np.ix_(ids,ids)]+c+np.diag(noise)
            ke=K[name][np.ix_(b,ids)];factor=cho_factor(kt,lower=True,check_finite=False)
            mu=mean+ke@cho_solve(factor,z-mean,check_finite=False)
            var=amp*np.maximum(1-np.einsum('ij,ji->i',ke,cho_solve(factor,ke.T,check_finite=False)),0)+amp*alpha
            out.append(np.expm1(np.minimum(np.maximum(mu,0)+var/2,20)))
    return SCALE*np.mean(out,axis=0)

def weights(P,y):
    prior=np.zeros(P.shape[1]);prior[0]=1;scale=max(float(np.mean(y*y)),1.)
    def f(w):return np.mean((P@w-y)**2)/scale+SHRINK*np.sum((w-prior)**2)
    def jac(w):return 2*P.T@(P@w-y)/(len(y)*scale)+2*SHRINK*(w-prior)
    r=minimize(f,prior,jac=jac,method='SLSQP',bounds=[(0,1)]*len(prior),
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))},
        options={'maxiter':500,'ftol':1e-11})
    if not r.success:raise RuntimeError(r.message)
    w=np.maximum(r.x,0);return w/w.sum()

def prepare_external(paired):
    # Hard allowlist. Concentrations, metadata text and competition IDs cannot become features/targets.
    allowed=paired.loc[:,['SMILES','SI','document_id']].copy()
    allowed.SMILES=allowed.SMILES.map(V.canon)
    if allowed.SMILES.isna().any() or not np.isfinite(allowed.SI).all() or (allowed.SI<0).any():
        raise ValueError('Invalid external training observations')
    ext=allowed.groupby(['SMILES','document_id'],as_index=False).SI.mean()
    ext['scaf']=ext.SMILES.map(V.scaffold)
    return ext

def make_predictor(tr,ext,smiles):
    K=V.build_kernels(smiles);y=tr[V.TGT].to_numpy(float)
    sources=sorted(ext.document_id.unique());names=['v14_best_reference','all_sources']+sources
    def experts(a,b,blocked,scheme):
        e=external_for_fold(ext,tr,K,a,blocked,scheme)
        ref=SCALE*V.gp_log_ens_raw(K,y,a,b)
        predictions=[ref,posterior(K,y,ext,a,b,e) if len(e) else ref.copy()]
        for source in sources:
            selected=e[ext.document_id.to_numpy()[e]==source]
            predictions.append(posterior(K,y,ext,a,b,selected) if len(selected) else ref.copy())
        P=np.column_stack(predictions)
        if not np.isfinite(P).all() or (P<0).any():raise ValueError('Invalid model prediction')
        return P
    return K,names,experts

def stats(y,p):
    e=(y-p)**2;keep=y!=y.max()
    return dict(mse=float(e.mean()),mae=float(np.abs(y-p).mean()),
        mse_without_max_diagnostic=float(e[keep].mean()),max_label_sse_share=float(e[~keep].sum()/max(e.sum(),1e-12)))

def run(tr,te,sample,paired,out,outer_folds=5,inner_folds=3):
    with threadpool_limits(limits=2):return _run(tr,te,sample,paired,out,outer_folds,inner_folds)

def _run(tr,te,sample,paired,out,outer_folds,inner_folds):
    started=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ext=prepare_external(paired);n=len(tr);y=tr[V.TGT].to_numpy(float)
    K,names,experts=make_predictor(tr,ext,list(tr.c)+ext.SMILES.tolist()+list(te.c))
    groups={'scaffold':tr.scaf.to_numpy(),'cluster065':V.butina_groups(K['mc2'][:n,:n],.65)}
    def inner_fit(a,blocked,scheme):
        P=np.zeros((len(a),len(names)))
        for ia,ib in folds(groups[scheme][a],inner_folds):
            P[ib]=experts(a[ia],a[ib],np.r_[blocked,a[ib]],scheme)
        return weights(P,y[a])
    rows=[];oofs=[];fold_rows=[];weight_rows=[];full_predictions={}
    for scheme,g in groups.items():
        P=np.zeros((n,len(names)));blend=np.zeros(n);fold_ids=np.zeros(n,int)
        for fid,(a,b) in enumerate(folds(g,outer_folds)):
            w=inner_fit(a,b,scheme);P[b]=experts(a,b,b,scheme);blend[b]=P[b]@w;fold_ids[b]=fid
            weight_rows.append(dict(scheme=scheme,fold=fid,**dict(zip(names,w))))
            for name,p in [(name,P[b,j]) for j,name in enumerate(names)]+[('nested_source_blend',blend[b])]:
                fold_rows.append(dict(scheme=scheme,fold=fid,n=len(b),model=name,mse=float(((p-y[b])**2).mean())))
            print(scheme,fid,'reference',round(((P[b,0]-y[b])**2).mean()),'blend',round(((blend[b]-y[b])**2).mean()),flush=True)
        for name,p in [(name,P[:,j]) for j,name in enumerate(names)]+[('nested_source_blend',blend)]:
            rows.append(dict(scheme=scheme,model=name,**stats(y,p)))
        frame=pd.DataFrame(P,columns=names);frame['nested_source_blend']=blend
        frame['target']=y;frame['row_id']=np.arange(n);frame['fold']=fold_ids;frame['scheme']=scheme;oofs.append(frame)
        full_predictions[scheme]=P
    metric=pd.DataFrame(rows);metric.to_csv(out/'metrics.csv',index=False)
    pd.concat(oofs).to_csv(out/'oof.csv',index=False);pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(out/'outer_weights.csv',index=False)
    w=inner_fit(np.arange(n),np.array([],int),'scaffold')
    P=experts(np.arange(n),np.arange(n+len(ext),n+len(ext)+len(te)),np.array([],int),'scaffold')
    blend=P@w
    pivot=metric.pivot(index='scheme',columns='model',values='mse')
    gate=bool((pivot['nested_source_blend']<pivot['v14_best_reference']).all())
    # Export all honest alternatives, with no claim that an unchanged control is new.
    for name,pred in [('submission_v21_candidate.csv',blend),('submission_v14_control.csv',P[:,0]),
                      ('submission_all_sources.csv',P[:,1]),('submission_selected.csv',blend if gate else P[:,0])]:
        f=sample[['index']].copy();f[V.TGT]=f['index'].map(pd.Series(pred,index=te['index']))
        assert f['index'].tolist()==sample['index'].tolist() and np.isfinite(f[V.TGT]).all()
        f.to_csv(out/name,index=False)
    changes=pd.DataFrame({'index':te['index'],'reference':P[:,0],'candidate':blend,'delta':blend-P[:,0]})
    changes.to_csv(out/'prediction_changes.csv',index=False)
    kept=ext[~ext.SMILES.isin(set(tr.c))].copy()
    training=pd.concat([pd.DataFrame({'SMILES':tr.c,'SI':y,'document_id':'competition','origin':'competition'}),
        kept[['SMILES','SI','document_id']].assign(origin='external')],ignore_index=True)
    training.to_csv(out/'model_training_dataset.csv',index=False)
    # Training targets and molecular structures only. No IC50/CC50 in saved model state.
    state=dict(train_smiles=tr.c.tolist(),train_si=y.tolist(),external=ext[['SMILES','SI','document_id']].to_dict('records'),
        expert_names=names,weights=w.tolist(),scale=SCALE,external_weight=EXTERNAL_WEIGHT,source_bias=SOURCE_BIAS)
    (out/'trained_model.json').write_text(json.dumps(state))
    report=dict(train_rows=n,external_pairs=len(paired),external_observations=len(kept),external_unique_molecules=kept.SMILES.nunique(),
        selected='source_blend' if gate else 'v14_reference',release_gate=gate,weights=dict(zip(names,w.tolist())),
        candidate_equals_reference=bool(np.allclose(blend,P[:,0],rtol=0,atol=1e-8)),
        maximum_prediction_change=float(abs(blend-P[:,0]).max()),elapsed_seconds=time.time()-started,
        forbidden_competition_columns_used=False,external_concentrations_used_as_features=False,
        public_score=None,validation='Development nested CV with scaffold and Butina .65 groups; external held-out structures/analogs excluded. Not an untouched holdout.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(metric.to_string(index=False));print(json.dumps(report,indent=2))
    return report

def predict(state,smiles):
    if state['scale']!=SCALE or state['external_weight']!=EXTERNAL_WEIGHT or state['source_bias']!=SOURCE_BIAS:
        raise ValueError('Model parameter/version mismatch')
    tr=pd.DataFrame({'SMILES':state['train_smiles'],'SI VACV':state['train_si']});tr['c']=tr.SMILES;tr['scaf']=tr.c.map(V.scaffold)
    ext=prepare_external(pd.DataFrame(state['external']));cs=[V.canon(s) for s in smiles]
    if any(s is None for s in cs):raise ValueError('Invalid SMILES')
    n=len(tr);_,names,experts=make_predictor(tr,ext,list(tr.c)+ext.SMILES.tolist()+cs)
    if names!=state['expert_names']:raise ValueError('Model source order mismatch')
    P=experts(np.arange(n),np.arange(n+len(ext),n+len(ext)+len(cs)),np.array([],int),'scaffold')
    return P@np.asarray(state['weights'])

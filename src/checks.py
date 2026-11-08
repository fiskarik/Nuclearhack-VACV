"""Checks of the competition rule and evaluation boundaries."""
from pathlib import Path
import json,tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import vacv_core as V
import source_model as M
import paired_data as D

def input_allowlist_check():
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)
        tr=pd.DataFrame({'SMILES':['CCO','CCN'],'SI VACV':[1.,2.],
            'CC50_uM':['DO_NOT_READ']*2,'IC50_uM_VACV':['DO_NOT_READ']*2})
        te=pd.DataFrame({'index':[7,3],'SMILES':['CCC','CCO'],
            'CC50_uM':[1e20,-1e20],'IC50_uM_VACV':[0,0]})
        sm=pd.DataFrame({'index':[3,7],'SI VACV':[987654321,-123456]})
        for name,x in [('train',tr),('test',te),('sample',sm)]:x.to_csv(p/f'{name}.csv',index=False)
        a=V.load(p/'train.csv',p/'test.csv',p/'sample.csv')
        tr.drop(columns=['CC50_uM','IC50_uM_VACV']).to_csv(p/'train.csv',index=False)
        te.drop(columns=['CC50_uM','IC50_uM_VACV']).to_csv(p/'test.csv',index=False)
        sm.drop(columns=['SI VACV']).to_csv(p/'sample.csv',index=False)
        b=V.load(p/'train.csv',p/'test.csv',p/'sample.csv')
        for x,y in zip(a,b):pd.testing.assert_frame_equal(x,y)
    return True

def external_allowlist_check(paired):
    a=M.prepare_external(paired)
    altered=paired.copy();altered['CC50_uM']=1e100;altered['IC50_uM']=1e-100
    altered['IC50_uM_VACV']='FORBIDDEN_FEATURE'
    b=M.prepare_external(altered);pd.testing.assert_frame_equal(a,b)
    assert set(a.columns)=={'SMILES','SI','document_id','scaf'}
    # Altering concentrations after the external label has been audited cannot affect model inputs.
    return True

def pair_arithmetic_check(paired):
    assert (paired.IC50_uM>0).all() and (paired.CC50_uM>0).all()
    assert paired.IC50_relation.eq('=').all() and paired.CC50_relation.eq('=').all()
    np.testing.assert_allclose(paired.SI,paired.CC50_uM/paired.IC50_uM,rtol=1e-12)
    c=paired[paired.si_activity_id.eq(D.CORRECTION['activity_id'])]
    assert len(c)==1 and c.SI_reported_raw.iloc[0]==156.73 and c.SI_reported_reviewed.iloc[0]==156733.
    assert c.SI.iloc[0]>150000
    return True

def split_isolation_check(tr,paired):
    ext=M.prepare_external(paired);n=len(tr)
    K,names,experts=M.make_predictor(tr,ext,list(tr.c)+ext.SMILES.tolist())
    groups={'scaffold':tr.scaf.to_numpy(),'cluster065':V.butina_groups(K['mc2'][:n,:n],.65)}
    for scheme,g in groups.items():
        for a,b in M.folds(g,5):
            for ia,ib in M.folds(g[a],3):
                blocked=np.r_[b,a[ib]];ei=M.external_for_fold(ext,tr,K,a[ia],blocked,scheme);e=ext.iloc[ei]
                assert not set(e.SMILES)&set(tr.iloc[blocked].c)
                assert not set(e.SMILES)&set(tr.iloc[a[ia]].c)
                if scheme=='scaffold':assert not set(e.scaf)&set(tr.iloc[blocked].scaf)
                if scheme=='cluster065' and len(ei):assert K['mc2'][np.ix_(n+ei,blocked)].max()<.65
    # Mutating labels of a held-out fold must not alter any expert prediction for it.
    a,b=M.folds(groups['scaffold'],5)[0]
    p=experts(a,b,b,'scaffold');altered=tr.copy();altered.loc[b,V.TGT]+=1e8
    _,_,other=M.make_predictor(altered,ext,list(tr.c)+ext.SMILES.tolist())
    np.testing.assert_allclose(p,other(a,b,b,'scaffold'),rtol=1e-12,atol=1e-9)
    return True

def output_check(out,te,sample):
    out=Path(out)
    for p in out.glob('submission*.csv'):
        f=pd.read_csv(p)
        assert list(f.columns)==['index','SI VACV']
        assert f['index'].tolist()==sample['index'].tolist()
        assert np.isfinite(f['SI VACV']).all() and (f['SI VACV']>=0).all()
    state=json.loads((out/'trained_model.json').read_text())
    assert all(set(x)=={'SMILES','SI','document_id'} for x in state['external'])
    with threadpool_limits(limits=2):p=M.predict(state,te.SMILES.iloc[:10].tolist())
    saved=pd.read_csv(out/'submission_v21_candidate.csv').set_index('index').loc[te['index'].iloc[:10],'SI VACV']
    np.testing.assert_allclose(p,saved,rtol=1e-10,atol=1e-8)
    return True

def run_checks(tr,te,sample,paired,out):
    with threadpool_limits(limits=2):
        report=dict(input_allowlist=input_allowlist_check(),external_feature_allowlist=external_allowlist_check(paired),
            external_arithmetic=pair_arithmetic_check(paired),fold_and_label_isolation=split_isolation_check(tr,paired),
            saved_model_and_submission=output_check(out,te,sample))
    (Path(out)/'checks.json').write_text(json.dumps(report,indent=2));print(report)
    return report

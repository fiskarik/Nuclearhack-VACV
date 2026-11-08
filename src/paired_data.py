"""External-data-only pairing and arithmetic audit. Never accepts competition CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import vacv_core as V
from fetch_sources import API,DOCS

# Explicit assay identities, not a join across arbitrary measurements.
# Each SI assay explicitly defines CC50/IC50 for the corresponding VACV cell assay.
PAIRS={
 'CHEMBL2321819':('CHEMBL2330536','CHEMBL2330528','CHEMBL2330527','BSC-1'),
 'CHEMBL4052722':('CHEMBL4056706','CHEMBL4056694','CHEMBL4056707','CV-1'),
 'CHEMBL4368876':('CHEMBL4369494','CHEMBL4369495','CHEMBL4369496','Vero'),
 'CHEMBL4699424':('CHEMBL4700293','CHEMBL4700294','CHEMBL4700295','Vero'),
 'CHEMBL4837211':('CHEMBL4838274','CHEMBL4838273','CHEMBL4838275','Vero'),
 'CHEMBL5223146':('CHEMBL5225279','CHEMBL5225278','CHEMBL5225280','Vero'),
 'CHEMBL6092128':('CHEMBL6092367','CHEMBL6092368','CHEMBL6092369','Vero'),
}
UNITS={'M':1e6,'mM':1000.,'uM':1.,'µM':1.,'μM':1.,'nM':.001,'pM':.000001}
# One reviewed correction, guarded by activity ID, DOI and observed value.
# Table 1 in the source uses comma as a thousands separator: 156,733.
CORRECTION={'activity_id':24978340,'doi':'10.1016/j.bmcl.2022.129080',
 'expected_value':156.73,'correct_value':156733.,
 'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC9674569/',
 'location':'Table 1, NIOCH-14, SI column','raw_source_text':'156,733',
 'reason':'Thousands separator in the primary table; corroborated by CC50/IC50.'}

def num(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan

def concentration(a):
    return num(a.get('standard_value'))*UNITS.get(a.get('standard_units'),np.nan)

def same_cell(ass,expected):
    normalize=lambda x:''.join(c for c in str(x).lower() if c.isalnum())
    observed=ass.get('assay_cell_type')
    if observed:return normalize(observed)==normalize(expected)
    return normalize(expected) in normalize(ass.get('description',''))

def audit(raw_sources,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];manifests=[]
    for raw in raw_sources:
        doc=raw['documents'][0]['document_chembl_id'];doi=raw['doi']
        if doc not in PAIRS:raise ValueError('Source not reviewed: '+doc)
        ia,ca,sa,cell=PAIRS[doc];am={a['assay_chembl_id']:a for a in raw['assays']}
        for aid in [ia,ca,sa]:
            if aid not in am:raise ValueError('Missing reviewed assay '+aid)
            if not same_cell(am[aid],cell):raise ValueError('Cell metadata changed: '+aid)
        description=am[sa]['description'].lower()
        if not all(w in description for w in ['cc50','ic50','vaccinia']):raise ValueError('SI definition changed')
        if 'vaccinia' not in am[ia]['description'].lower():raise ValueError('Non-VACV inhibition assay')
        manifests.append(dict(document_id=doc,doi=doi,ic_assay=ia,cc_assay=ca,si_assay=sa,
            cell=cell,strain=am[ia].get('assay_strain'),
            ic_description=am[ia]['description'],cc_description=am[ca]['description'],si_description=am[sa]['description']))
        acts=raw['activities']
        for si in [a for a in acts if a['assay_chembl_id']==sa]:
            mol=si['molecule_chembl_id'];reasons=[]
            ic=[a for a in acts if a['assay_chembl_id']==ia and a['molecule_chembl_id']==mol and a['standard_type']=='IC50']
            cc=[a for a in acts if a['assay_chembl_id']==ca and a['molecule_chembl_id']==mol and a['standard_type']=='CC50']
            c=V.canon(si.get('canonical_smiles'));raw_si=num(si.get('standard_value'));reported=raw_si;correction=''
            if si['activity_id']==CORRECTION['activity_id']:
                if doi.lower()!=CORRECTION['doi'] or not np.isclose(raw_si,CORRECTION['expected_value'],rtol=0,atol=.005):
                    raise ValueError('Reviewed correction no longer matches source; re-review, never silently override.')
                reported=CORRECTION['correct_value'];correction='primary_table_thousands_separator'
            row=dict(SMILES=c,IC50_uM=np.nan,CC50_uM=np.nan,SI=np.nan,
                SI_reported_raw=raw_si,SI_reported_reviewed=reported,doi=doi,document_id=doc,
                molecule_id=mol,cell=cell,strain=am[ia].get('assay_strain'),
                si_activity_id=si['activity_id'],ic_activity_id=None,cc_activity_id=None,
                ic_assay_id=ia,cc_assay_id=ca,si_assay_id=sa,
                si_source_url=f'{API}/activity/{si["activity_id"]}.json',
                correction=correction,definition='CC50/IC50',origin='external_ChEMBL',
                label_origin='ratio_of_paired_external_concentrations',
                ic_description=am[ia]['description'],cc_description=am[ca]['description'])
            if len(ic)!=1 or len(cc)!=1:reasons.append('missing_or_ambiguous_pair')
            else:
                ic,cc=ic[0],cc[0];iv,cv=concentration(ic),concentration(cc)
                row.update(IC50_uM=iv,CC50_uM=cv,ic_activity_id=ic['activity_id'],cc_activity_id=cc['activity_id'],
                    ic_source_url=f'{API}/activity/{ic["activity_id"]}.json',
                    cc_source_url=f'{API}/activity/{cc["activity_id"]}.json',
                    IC50_relation=ic.get('standard_relation'),CC50_relation=cc.get('standard_relation'))
                if not c or c!=V.canon(ic.get('canonical_smiles')) or c!=V.canon(cc.get('canonical_smiles')):
                    reasons.append('structure_mismatch')
                if any(a.get('standard_relation')!='=' for a in [ic,cc,si]):reasons.append('censored_or_unknown_relation')
                if any(a.get('data_validity_comment') for a in [ic,cc,si]):reasons.append('validity_flag')
                if not (np.isfinite(iv) and np.isfinite(cv) and iv>0 and cv>0):reasons.append('invalid_concentration_or_unit')
                else:
                    ratio=cv/iv;row['SI']=ratio
                    row['reported_vs_ratio_factor']=reported/ratio if np.isfinite(reported) else np.nan
                    # Integer SI may have rounding up to 1 unit; otherwise allow 5% rounding.
                    if not np.isfinite(reported) or abs(reported-ratio)>max(1.,.05*ratio):
                        reasons.append('SI_arithmetic_conflict')
            row['accepted']=not reasons;row['review_reasons']=';'.join(reasons);rows.append(row)
    review=pd.DataFrame(rows);review.to_csv(out/'pair_audit.csv',index=False)
    accepted=review[review.accepted].copy();accepted.to_csv(out/'external_pairs.csv',index=False)
    pd.DataFrame(manifests).to_csv(out/'assay_pair_manifest.csv',index=False)
    (out/'reviewed_corrections.json').write_text(json.dumps([CORRECTION],indent=2))
    report=dict(reviewed_rows=len(review),accepted_pairs=len(accepted),unique_structures=accepted.SMILES.nunique(),
        sources=int(accepted.document_id.nunique()),corrected_rows=int((accepted.correction!='').sum()),
        rejected_reasons=review.loc[~review.accepted,'review_reasons'].value_counts().to_dict(),
        scope='Published SI corroborated by unique concentration pair in explicitly mapped assays. No competition CC50/IC50 read. Structures are ChEMBL-curated; not all manually redrawn from papers.')
    (out/'audit_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return accepted

if __name__=='__main__':
    root=Path(__file__).parent
    audit([json.loads(p.read_text()) for p in sorted((root/'raw_sources').glob('*.json'))],root/'data_audit')

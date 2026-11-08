from pathlib import Path
import json,time,concurrent.futures
import requests
API='https://www.ebi.ac.uk/chembl/api/data'
DOCS=['CHEMBL2321819','CHEMBL4052722','CHEMBL4368876','CHEMBL4699424','CHEMBL4837211','CHEMBL5223146','CHEMBL6092128']

def pages(endpoint,key,params):
    out=[];offset=0
    while True:
        for attempt in range(4):
            try:
                r=requests.get(f'{API}/{endpoint}.json',params={**params,'limit':500,'offset':offset},timeout=(15,60));r.raise_for_status();j=r.json();break
            except requests.RequestException:
                if attempt==3:raise
                time.sleep(2**attempt)
        b=j[key];out+=b;offset+=len(b)
        if not j['page_meta'].get('next'):return out
        if not b:raise RuntimeError('Empty nonterminal API page')

def download_doc(doc,out,refresh=False):
    p=Path(out)/f'{doc}.json';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not refresh:return json.loads(p.read_text())
    acts=pages('activity','activities',{'document_chembl_id':doc})
    assays=pages('assay','assays',{'document_chembl_id':doc})
    docs=pages('document','documents',{'document_chembl_id':doc})
    if len(docs)!=1:raise ValueError('Ambiguous document')
    raw={'doi':docs[0].get('doi'),'documents':docs,'activities':acts,'assays':assays}
    p.write_text(json.dumps(raw));print(doc,'activities',len(acts),'assays',len(assays),flush=True)
    return raw

def download(out,refresh=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        return list(ex.map(lambda d:download_doc(d,out,refresh),DOCS))

if __name__=='__main__':download(Path(__file__).parent/'raw_sources')

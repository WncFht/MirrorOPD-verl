"""CPU FP64 spectra and frame trajectories on prespecified k_proj matrices.

Ledger: each row is one same-coordinate checkpoint matrix. Singular values are
complete and sorted. Subspace quantities compare projectors (no signed-vector
matching); rank denotes component count. Early means earliest available export,
not the measured time of subspace formation. Base/teacher source precision is
retained in the protocol; promotion cannot recover missing FP32 master updates.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from examples.mirroropd.research.core.io import dump  # noqa: E402
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest  # noqa: E402


def main():
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'manifest.json').read_text())
    output = ROOT/'temporal-svd'
    output.mkdir(exist_ok=True)
    protocol = dict(device='cpu', arithmetic='float64', layers=[0,13,27], modules=['self_attn.k_proj'],
                    ranks=[8,16,64], seed=20261110, torch_version=str(torch.__version__),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    semantics='earliest available is not subspace formation; stored tensors promoted before subtraction')
    dump(output/'protocol.json',protocol)
    count = 0
    for config in plan['configs']:
        if not config['same_coordinates']:
            continue
        base, teacher = [TensorCheckpoint(config[k]) for k in ('base','teacher')]
        base.compatible(teacher)
        for layer in protocol['layers']:
            destination = output/f"{config['label']}-l{layer:02d}.json"
            if destination.exists():
                raise FileExistsError(f'Use a new version for repeated CPU probes: {destination}')
            key = f'model.layers.{layer}.self_attn.k_proj.weight'
            rb, rt = base.get(key), teacher.get(key)
            b,t = rb.double(),rt.double()
            u0,s0,v0 = torch.linalg.svd(b,full_matrices=False)
            st = torch.linalg.svdvals(t)
            dt = t-b
            ut,sd,vt = torch.linalg.svd(dt,full_matrices=False)
            et, eb = dt.square().sum(), b.square().sum()
            generator = torch.Generator().manual_seed(protocol['seed']+layer)
            random = torch.linalg.qr(torch.randn(b.shape[1],64,generator=generator,dtype=torch.float64),mode='reduced').Q
            data = dict(label=config['label'], tensor=key, shape=list(b.shape), layers=[layer],
                        base_dtype=str(rb.dtype), teacher_dtype=str(rt.dtype),
                        base_sha256=tensor_digest(rb),teacher_sha256=tensor_digest(rt),
                        base_singular_values=s0.tolist(),teacher_singular_values=st.tolist(),
                        teacher_update_singular_values=sd.tolist(),teacher_energy=float(et),
                        stages=[],protocol=protocol)
            ue,ve = None,None
            for step,path in config['stages'].items():
                started = time.monotonic()
                checkpoint = TensorCheckpoint(path)
                base.compatible(checkpoint)
                raw = checkpoint.get(key)
                w = raw.double()
                delta = w-b
                u,s,v = torch.linalg.svd(delta,full_matrices=False)
                full = torch.linalg.svdvals(w)
                e = delta.square().sum()
                if ue is None:
                    ue,ve = u[:,:64].clone(),v[:64].clone()
                inner = (dt*delta).sum()
                values = dict(step=int(step),dtype=str(raw.dtype),sha256=tensor_digest(raw),
                              weight_singular_values=full.tolist(),update_singular_values=s.tolist(),
                              update_energy=float(e),stable_rank=float(e/s[0].square()),
                              spectral_drift_base=float((full-s0).norm()/eb.sqrt()),
                              spectral_drift_teacher=float((full-st).norm()/eb.sqrt()),
                              weight_missing=float((t-w).square().sum()/et),
                              update_cosine=float(inner/(et*e).sqrt()),subspaces={})
                for rank in protocol['ranks']:
                    basis = v[:rank].T
                    values['subspaces'][str(rank)] = dict(
                        teacher_right_overlap=float((vt[:rank]@basis).square().sum()/rank),
                        teacher_left_overlap=float((ut[:,:rank].T@u[:,:rank]).square().sum()/rank),
                        teacher_right_energy=float((delta@vt[:rank].T).square().sum()/e),
                        early_right_energy=float((delta@ve[:rank].T).square().sum()/e),
                        early_right_overlap=float((ve[:rank]@basis).square().sum()/rank),
                        base_principal_right_energy=float((delta@v0[:rank].T).square().sum()/e),
                        random_right_energy=float((delta@random[:,:rank]).square().sum()/e),
                        isotropic_right=rank/b.shape[1],
                        own_top_energy=float(s[:rank].square().sum()/e))
                assert abs(float(s.square().sum()-e))/float(e) < 1e-8
                assert float((delta-(u*s)@v).norm()/e.sqrt()) < 1e-8
                values['wall_seconds']=time.monotonic()-started
                data['stages'].append(values)
                dump(destination.with_suffix('.partial.json'),data)
                print(config['label'],layer,step,round(values['wall_seconds'],2),flush=True)
            dump(destination,data)
            destination.with_suffix('.partial.json').unlink()
            count += 1
            dump(output/'progress.json',dict(complete_matrices=count,total_matrices=9,updated_at=time.time()))
    dump(output/'complete.json',dict(matrices=count,completed_at=time.time()))


if __name__=='__main__':
    main()

import numpy as np, torch, importlib.util, sys, torchvision
from datasets import load_dataset
spec = importlib.util.spec_from_file_location("xai","xai.py")
xai = importlib.util.module_from_spec(spec); sys.modules["xai"]=xai; spec.loader.exec_module(xai)
xai.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xai.LIME_SAMPLES = 500; xai.FAITH_STEPS = 25
m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval().to(xai.DEVICE)
test = load_dataset("taufiktrf/AQUA20")["test"]
from torchvision import transforms
from torchvision.transforms import functional as TF
geom = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
gc = xai.GradCAM(m, m.layer4[-1], False); rs = np.random.RandomState(1)
agg={k:[] for k in ("gradcam","lime","random")}
for idx in (3,341,944,718,1076):
    raw=geom(test[idx]["image"].convert("RGB")); raw01=raw.permute(1,2,0).numpy()
    x=TF.normalize(raw,xai.IMAGENET_MEAN,xai.IMAGENET_STD).to(xai.DEVICE)
    pred=int(torch.softmax(m(x.unsqueeze(0)),1).argmax())
    maps={k:xai.normalize_saliency(v) for k,v in
          {"gradcam":gc(x,pred),"lime":xai.lime_saliency(m,raw01,pred),
           "random":rs.rand(224,224).astype(np.float32)}.items()}
    blur=TF.gaussian_blur(x.unsqueeze(0),51,11.0).squeeze(0)
    line=f"RESULT #{idx}: "
    for k,v in maps.items():
        d,i=xai.faithfulness(m,x,v,pred,blur); agg[k].append((d,i))
        line+=f"{k} d={d:.4f}/i={i:.4f} conc={xai.concentration(v):.2f}  "
    print(line, flush=True)
mean={k:(np.mean([a for a,_ in v]),np.mean([b for _,b in v])) for k,v in agg.items()}
print("RESULT === MEAN over 5 real images ===")
for k,(d,i) in mean.items(): print(f"RESULT   {k:8} del={d:.4f} ins={i:.4f}")
rd,ri=mean["random"]
for k in ("gradcam","lime"):
    d,i=mean[k]
    print(f"RESULT   {k:8} del {'OK ' if d<rd else 'FAIL'} ({d:.4f} vs {rd:.4f})   ins {'OK ' if i>ri else 'FAIL'} ({i:.4f} vs {ri:.4f})")

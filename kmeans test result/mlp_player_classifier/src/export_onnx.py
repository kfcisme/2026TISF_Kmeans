import yaml, torch, numpy as np
from model import MLP

def main():
    C = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    ckpt = torch.load("best.pt", map_location="cpu")
    feat_cols = ckpt["feat_cols"]
    classes  = ckpt["classes"]
    model = MLP(in_dim=len(feat_cols),
                num_classes=len(classes),
                hidden_sizes=tuple(C["model"]["hidden_sizes"]),
                dropout=C["model"]["dropout"],
                activation=C["model"]["activation"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.randn(1, len(feat_cols))
    torch.onnx.export(model, dummy, C["export"]["onnx_path"],
                      input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features":{0:"N"}, "logits":{0:"N"}},
                      opset_version=C["export"]["opset"])
    print(f"Exported to {C['export']['onnx_path']} with classes: {classes}")

if __name__ == "__main__":
    main()

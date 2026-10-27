
import torch
import onnx
from models.Mymodel import Mymodel, count_parameters

model = Mymodel(T=6)
model.eval()

n_params = count_parameters(model)
print(f"参数量 = {n_params:,}  ({n_params / 1e6:.3f}M)")

dummy_input = torch.randn(1, 3, 128, 128)


torch.onnx.export(
    model,
    dummy_input,
    "mymodel.onnx",
    input_names=["rainy"],
    output_names=["derained"],
    opset_version=11,
    dynamic_axes=None,
)
print("已导出 mymodel.onnx")


onnx_model = onnx.load("mymodel.onnx")
onnx.checker.check_model(onnx_model)
print("好了")

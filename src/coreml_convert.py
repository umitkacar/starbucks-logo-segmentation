import json

import coremltools as ct
import torch
import torch.nn as nn
from coremltools.models.neural_network import quantization_utils

from mobile_seg.modules.net import load_trained_model

if __name__ == "__main__":

    with open("params/config.json") as f:
        config = json.load(f)

    trained_model = load_trained_model(config)

    class Wrapper(nn.Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.model = model

        def forward(self, x):
            x = self.model(x)
            x = x * 255
            x = torch.cat((x, x, x), 1)
            return x

    model = Wrapper(model=trained_model).eval()
    # print(model)
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, config["input_size"], config["input_size"])
        jit_model = torch.jit.trace(model, input_tensor)
        out = jit_model(input_tensor)
        print(out.shape)

        model_fp32 = ct.convert(
            model=jit_model,
            inputs=[
                ct.ImageType(
                    name="input0",
                    color_layout=ct.colorlayout.RGB,
                    shape=input_tensor.shape,
                    scale=1.0 / 255.0,
                )
            ],
            outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
            convert_to="neuralnetwork",
        )
        model_fp32.save(config["model_fp32_path"])

        # Quantization 16 bits
        model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
        model_fp16.save(config["model_fp16_path"])

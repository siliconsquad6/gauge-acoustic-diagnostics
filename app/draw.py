import json, torch
from diffusers import DiffusionPipeline
d = json.load(open("/home/hp20/Desktop/edge-ai/gauge/out/diag.json"))
pipe = DiffusionPipeline.from_pretrained("/home/hp20/Desktop/edge-ai/gauge/models/qwen-image", torch_dtype=torch.bfloat16).to("cuda")
prompt = (f"Photorealistic 3D product render of a single industrial {d['machine']}, whole object centered, "
          f"three-quarter view, plain white background, soft studio lighting, no text. "
          f"The {d['likely_component']} glows bright orange-red, showing a {d['fault']}.")
img = pipe(prompt=prompt, negative_prompt="text, labels, blurry, cropped, multiple objects",
           width=1024, height=1024, num_inference_steps=20, true_cfg_scale=4.0,
           generator=torch.Generator("cuda").manual_seed(7)).images[0]
img.save("/home/hp20/Desktop/edge-ai/gauge/out/machine.png"); print("saved machine.png")

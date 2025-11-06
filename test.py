from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from chandra.model.schema import BatchInputItem
from chandra.model.hf import generate_hf
import torch

model_id = "datalab-to/chandra"

# Clear any existing CUDA memory
torch.cuda.empty_cache()

# Load the model with memory optimization
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(model_id)

# Attach the processor to the model so generate_hf can find it
model.processor = processor

url = "https://www.digitipps.ch/wp-content/uploads/2020/10/JPG-Format-Dateigroesse.jpg"
image = Image.open(requests.get(url, stream=True).raw)

batch = [
    BatchInputItem(
        image=image,
        prompt="<image>",
        prompt_type="ocr_layout" # isn't strictly restricted - it maps to different internal prompt templates that tell the model what task to perform.
    )
]

# Only pass batch and model (processor is accessed via model.processor)
result = generate_hf(batch, model)[0]
print(result.raw)

#results.raw output example
""" Hello! This is a detailed cross-section diagram illustrating the internal structure of a volcano during an eruption. The diagram shows the path of magma from the magma chamber, through the volcanic conduit and fissures, to the surface, where it forms lava flows and erupts as ash and debris. The sun is shown in the upper left corner.

Here is a numbered list of the labeled parts, based on the provided image and standard geological terminology:

1.  **Magma Chamber:** The large reservoir of molten rock deep underground, located below the Earth's crust.
2.  **Crust:** The outermost layer of the Earth.
3.  **Conduit (Volcanic Pipe):** The main vertical channel through which magma rises from the magma chamber to the surface. """

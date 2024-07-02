import os
import torch
import time
import logging
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import base64

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

models = {}
processors = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CaptionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int
    temperature: float

def load_model(model_dir, model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    models[model_name] = model
    processors[model_name] = processor

def generate_caption(model, processor, image, prompt="<CAPTION>", max_new_tokens=1024, num_beams=3, device="cuda"):
    image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        do_sample=False,
        num_beams=num_beams
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = generated_text.replace("</s>", "").strip()
    
    return parsed_answer

@app.on_event("startup")
def on_startup():
    default_model_name = "Florence-2-SD3-Captioner"
    default_model_dir = "/mnt/e/public_model/Florence-2-SD3-Captioner"
    load_model(default_model_dir, default_model_name, device)

@app.post("/v1/chat/completions")
async def generate_caption_api(caption_request: CaptionRequest):
    start_time = time.time()
    try:
        model_name = caption_request.model
        model_dir_mapping = {
            "Florence-2-large-ft": "/mnt/e/public_model/Florence-2-large-ft",
            "Florence-2-large": "/mnt/e/public_model/Florence-2-large",
            "Florence-2-base-ft": "/mnt/e/public_model/Florence-2-base-ft",
            "Florence-2-base": "/mnt/e/public_model/Florence-2-base",
            "Florence-2-SD3-Captioner": "/mnt/e/public_model/Florence-2-SD3-Captioner"
        }

        if model_name not in models:
            if model_name in model_dir_mapping:
                load_model(model_dir_mapping[model_name], model_name, device)
            else:
                raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

        model = models[model_name]
        processor = processors[model_name]
        
        prompt = caption_request.messages[0]["content"][0]["text"]
        image_base64 = caption_request.messages[0]["content"][1]["image_url"]["url"]
        image_data = base64.b64decode(image_base64.split(',')[1])
        max_tokens = caption_request.max_tokens
        temperature = caption_request.temperature
        
        caption = generate_caption(
            model,
            processor,
            image_data,
            prompt,
            max_new_tokens=max_tokens,
            num_beams=3,
            device=device
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"Request processed in {elapsed_time:.4f} seconds")
        
        return {"choices": [{"message": {"content": caption}}]}
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"Error processing request: {e}, time taken: {elapsed_time} seconds")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

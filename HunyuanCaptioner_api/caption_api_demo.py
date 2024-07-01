import torch
import os
import base64
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from io import BytesIO

app = FastAPI()

# 全局变量
models = {}
processors = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: List[MessageContent]

class CaptionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: Optional[float] = None
    num_beams: int = 1
    skip_special: bool = False

def load_model(model_path, model_name, device):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, llava_type_model=True, load_4bit=False)
    model.to(device)
    models[model_name] = {"tokenizer": tokenizer, "model": model, "image_processor": image_processor}

def eval_model(models,
               model_name,
               messages,
               temperature=0.2,
               top_p=None,
               num_beams=1,
               max_new_tokens=512,
               return_history=False,
               skip_special=False):
    disable_torch_init()

    model = models[model_name]
    text = None
    image_url = None

    for content in messages[0].content:
        if content.type == "text":
            text = content.text
        elif content.type == "image_url":
            image_url = content.image_url["url"]

    qs = text
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model["model"].config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model["model"].config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates['llava_v1'].copy()
    if skip_special:
        conv.append_message(conv.roles[0], text)
    else:
        conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image_url is not None:
        image_data = base64.b64decode(image_url.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_sizes = [image.size]
        images_tensor = process_images(
            [image],
            model["image_processor"],
            model["model"].config
        ).to(model["model"].device, dtype=torch.float16)
    else:
        image_sizes = [(1024, 1024)]
        images_tensor = torch.zeros(1, 5, 3, model["image_processor"].crop_size["height"], model["image_processor"].crop_size["width"])
        images_tensor = images_tensor.to(model["model"].device, dtype=torch.float16)

    tokenizer = model["tokenizer"]
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model["model"].device)
    )
    with torch.inference_mode():
        output_ids = model["model"].generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if return_history:
        return outputs, conv
    return outputs

@app.on_event("startup")
async def startup_event():
    model_dir_mapping = {
        "HunyuanCaptioner": "./ckpts/captioner"
    }
    for model_name, model_path in model_dir_mapping.items():
        load_model(model_path, model_name, device)

@app.post("/v1/chat/completions")
async def generate_caption_api(caption_request: CaptionRequest):
    try:
        model_name = caption_request.model
        model_dir_mapping = {
            "HunyuanCaptioner": "./ckpts/captioner"
        }

        if model_name not in models:
            if model_name in model_dir_mapping:
                load_model(model_dir_mapping[model_name], model_name, device)
            else:
                raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

        caption = eval_model(models, model_name, caption_request.messages,
                             caption_request.temperature, caption_request.top_p, caption_request.num_beams,
                             caption_request.max_tokens, skip_special=caption_request.skip_special)
        return {"choices": [{"message": {"content": caption}}]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

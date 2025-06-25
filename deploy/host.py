# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# import torch
# import uvicorn

# app = FastAPI()
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Alternative memory-efficient loading options without bitsandbytes

# model_id = "/home/siamai/data/huggingface/hub/models--tarun7r--Finance-Llama-8B/snapshots/7934db35d2374c1321b90a9deb0d84b97525b025"

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model.to(device)
# model.eval()

# class Query(BaseModel):
#     prompt: str
#     # max_length: int = 256  # optional generation length limit

# @app.post("/generate")
# async def generate_text(query: Query):
#     inputs = tokenizer(query.prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"generated_text": generated_text}

# if __name__ == "__main__":
#     uvicorn.run("host:app", host="0.0.0.0", port=6666, reload=True)



from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from starlette.concurrency import run_in_threadpool

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "/home/siamai/data/huggingface/hub/models--tarun7r--Finance-Llama-8B/snapshots/7934db35d2374c1321b90a9deb0d84b97525b025"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)
model.eval()

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 256  # default cap to avoid runaway memory usage

@app.post("/generate")
async def generate_text(query: Query):
    return await run_in_threadpool(sync_generate, query)

def sync_generate(query: Query):
    inputs = tokenizer(query.prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=query.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run("host:app", host="0.0.0.0", port=6666, reload=True)

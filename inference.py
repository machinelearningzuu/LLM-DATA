import time
from transformers import pipeline

llm = pipeline(
            "text-generation",
            model="zuu/phi-2-bio-tech-qbit-synthetic",
            trust_remote_code=True,
            device=0
            )

def inference_pi2(
                text,
                max_length=600,
                min_length=100
                ):
    response = llm(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    pad_token_id=50256
                    )
    response = response[0]['generated_text']
    response = response.replace(text, '')
    return response

text = """What are the main causes for Ret Syndromm ?"""

time0 = time.time()
response = inference_pi2(text)
time1 = time.time()

print("======================== Time Taken for Inference ========================")
print(f"Time Taken: {time1 - time0} seconds")
print("=============================== Response =================================")
print(response)
# Triton BLS Example of Text-Classification pipeline with Hugging Face x ORT

NVIDIA Triton example for Text-Classification pipeline with Hugging Face x ORT. This examples deploys 2 models to NVIDIA Triton 1x a BERT based ONNX Model (not included) and a Python, which is a BLS to create a e2e pipeline expecting a JSON And returning a JSON.

To use this example you need to put your ONNX model into the `models/bert/1` folder and adjust the `tokenizer.json` and `config.json` files in `models/pipeline/1`.

## Build Docker

```Bash
docker build -t triton-bls-example .
```

## Start Triton

```bash
	docker run  -t -i	-p 8000:8000 \
  -v $(pwd)/models:/opt/tritonserver/models \
  -v $(pwd)/tokenizer.json:/tmp/transformers/tokenizer.json \
  triton-bls-example \
  tritonserver --model-repository=/opt/tritonserver/models
```

## Run client

```python
from tritonclient.utils import *
import tritonclient.http as httpclient
import timeit
import json
import numpy as np

model_name = "pipeline"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

triton_client = httpclient.InferenceServerClient(url=url, verbose=False)
text = "I like you. I love you"


def send_request(input_text):
    # prepare request
    query = httpclient.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
    model_score = httpclient.InferRequestedOutput(name="PREDICTION", binary_data=False)
    query.set_data_from_numpy(np.asarray([input_text] * batch_size, dtype=object))

    # send request
    response = triton_client.infer(
        model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
    )

    resp = json.loads(response.get_response()["outputs"][0]["data"][0])
    return resp

print(send_request(text))
```

## Benchmark

DistilBERT test.

```
############# Start of benchmark ###############
Benchmark for sequence length: 8:
Avg e2e time: 5057.015421999495µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 16:
Avg e2e time: 6803.2411310005045µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 32:
Avg e2e time: 10443.32282599953µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 64:
Avg e2e time: 17943.68026900065µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 128:
Avg e2e time: 30601.669228999526µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 256:
Avg e2e time: 67596.62277200005µs
############# End of benchmark ###############
############# Start of benchmark ###############
Benchmark for sequence length: 512:
Avg e2e time: 162650.63517µs
############# End of benchmark ###############
```

## Resources

* [BLS example documentation](https://github.com/triton-inference-server/python_backend/tree/main/examples/bls)
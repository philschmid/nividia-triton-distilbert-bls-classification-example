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


if __name__ == "__main__":
    seq_lenghts = [8, 16, 32, 64, 128, 256, 512]
    for seq_len in seq_lenghts:
        loops = 1000
        b_i = " l" * (seq_len - 2)
        duration_e2e = (timeit.timeit(f"send_request('{b_i}')", globals=locals(), number=loops) / loops) * 1000 * 1000

        print(f"############# Start of benchmark ###############")
        print(f"Benchmark for sequence length: {seq_len}:")
        print(f"Avg e2e time: {duration_e2e}µs")
        print(f"############# End of benchmark ###############")

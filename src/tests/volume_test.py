import modal

app = modal.App("test-volume")
image = modal.Image.debian_slim()
vol = modal.Volume.from_name("my-test-volume")


@app.function(volumes={"/data": vol}, image=image)
def run():
    import numpy as np
    # with open("/data/xyz.txt", "w") as f:
    #     f.write("hello")
    with open("/data/xyz.txt", "wb") as f:
        # f.write("hello")
        np.save(f, np.array([1, 2, 3]), allow_pickle=False)

    vol.commit()  # Needed to make sure all changes are persisted

@app.function(volumes={"/my_vol": vol})
def some_func():
    import os
    os.listdir("/my_vol")


@app.local_entrypoint()
def main():
    run.remote()
    print("=========")
    some_func.remote()
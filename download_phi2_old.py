import os
# os.environ["https_proxy"] = "http://xxx.xxx.xxx.xxx:xx"  # in case you need proxy to access Huggingface Hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/phi-2", 
    revision="d3186761bf5c4409f7679359284066c25ab668ee",
    local_dir='checkpoints/base/phi-2-old',
    local_dir_use_symlinks=False
)
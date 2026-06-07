---
title: "[Bug] The trained dflash has an extremely low acceptance rate · Issue #455 · sgl-project/SpecForge"
source: "https://github.com/sgl-project/SpecForge/issues/455"
author:
  - "[[baifanxxx]]"
published: 2026-01-27
created: 2026-04-17
description: "Checklist 1. I have searched related issues but cannot get the expected help. 2. The bug has not been fixed in the latest version. 3."
tags:
  - "clippings"
---
### Checklist

- 1\. I have searched related issues but cannot get the expected help.
	2\. The bug has not been fixed in the latest version.
	3\. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
	4\. If the issue you raised is not a bug but a question, please raise a discussion at [https://github.com/sgl-project/SpecForge/discussions/new/choose](https://github.com/sgl-project/SpecForge/discussions/new/choose) Otherwise, it will be closed.
	5\. Please use English, otherwise it will be closed.
	To pick up a draggable item, press the space bar. While dragging, use the arrow keys to move the item. Press space again to drop the item in its new position, or press escape to cancel.

### Describe the bug

I cloned the latest SpecForge codebase and noticed that it now supports training with DFlash. Based on this, I launched a DFlash training job using the script below.

During training, both **loss** and **accuracy** behaved normally. On the evaluation set, there was only mild overfitting, which did not seem significant. However, when I loaded the trained weights into the **official DFlash benchmark script** on gsm8k dataset , ([https://github.com/z-lab/dflash/blob/main/run\_benchmark.sh](https://github.com/z-lab/dflash/blob/main/run_benchmark.sh)), I observed an **acceptance rate of only 1.29/(1+3)**, which is extremely low. This suggests that the training has effectively failed.

I would like to ask whether anyone has **successfully trained and inferred a DFlash model**. Any discussion or help in locating the root cause would be greatly appreciated.

For context, I have previously **successfully trained an Eagle3 model**, which indicates that my data preprocessing, training pipeline, and evaluation setup should generally be correct.

Below is the training script I used:

```shell
#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=16

ATTENTION_BACKEND=sdpa

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /data/weights/qwen3-8b \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-perfectblend-baseline \
    --num-epochs 15 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --log-interval 100 \
    --eval-interval 5000 \
    --save-interval 10000 \
    --eval-data-path $ROOT_DIR/cache/dataset/opc_test.jsonl \
    --cache-dir $ROOT_DIR/cache \
    --report-to tensorboard \
    --target-model-backend sglang \
    --resume
```

Looking forward to any insights or suggestions. Thanks!

Best regards,  
BAI Fan

### Reproduction

Below is the training script I used:

```shell
#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=16

ATTENTION_BACKEND=sdpa

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /data/weights/qwen3-8b \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-perfectblend-baseline \
    --num-epochs 15 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --log-interval 100 \
    --eval-interval 5000 \
    --save-interval 10000 \
    --eval-data-path $ROOT_DIR/cache/dataset/opc_test.jsonl \
    --cache-dir $ROOT_DIR/cache \
    --report-to tensorboard \
    --target-model-backend sglang \
    --resume
```

### Environment

SpecForge-main

---

## Comments

> **Walid-Ahmed** · 2026-01-27
> 
> Which dataset you used as  
> "perfectblend\_train.jsonl" ?

> **baifanxxx** · 2026-01-27
> 
> Hi,  
> This is a sufficiently large mixed dataset that can support multiple tasks. The link is as follows:  
> [https://huggingface.co/datasets/mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)

> **baifanxxx** · 2026-01-28
> 
> [@xiaomin-D](https://github.com/xiaomin-D)  
> I hope you can discuss and resolve this issue with me, and I look forward to your reply.

> **xiaomin-D** · 2026-01-28
> 
> > [@xiaomin-D](https://github.com/xiaomin-D) I hope you can discuss and resolve this issue with me, and I look forward to your reply.
> 
> Thanks for your experiments~  
> Since ZLab hasn’t released the official training details yet, we currently have to reverse-engineer the setup from the inference code, so we can’t fully align with the official pipeline for now. We’ll follow up once their paper or code is public.
> 
> For the current situation:
> 
> 1. Please try aligning the chat template. By default, the dflash model disables , while SpecForge enables it. A mismatch between training and inference chat templates can significantly affect acceptance length. The current SpecForge default keeps mainly for better generality across different models and use cases.
> 2. On the loss side, different choices can matter. We use CE loss by default, but in our experiments, KL loss — as well as other loss variants — can sometimes give better results depending on the setup.
> 3. It may also help to further increase the dataset size, and make sure the dataset is regenerated , which tends to improve stability and acceptance length.
> 
> Although there is still a gap compared to the official model, with aligned chat templates and KL loss, we can reach around above 3 tokens on average acceptance length, for reference.
> 
> You can also try different training techniques, and any contributions would be appreciated.
> 
> [#427 (comment)](https://github.com/sgl-project/SpecForge/pull/427#issuecomment-3802852027)

> **baifanxxx** · 2026-01-28
> 
> Thank you for your reply. In fact, when I trained dflash using only two repeated data samples, the loss could be overfitted to zero, but I still couldn't achieve a satisfactory acceptance rate during dflash inference. This doesn't seem to be an issue with training techniques or the loss function; there must be some bug causing the misalignment.

> **xiaomin-D** · 2026-01-28
> 
> > Thank you for your reply. In fact, when I trained dflash using only two repeated data samples, the loss could be overfitted to zero, but I still couldn't achieve a satisfactory acceptance rate during dflash inference. This doesn't seem to be an issue with training techniques or the loss function; there must be some bug causing the misalignment.
> 
> There must also be some training techniques that are still not well understood yet, but I think you can try following what I suggested first, because the inference and training templates, data regeneration, and different loss functions all have a significant impact on the acceptance rate.

> **xiaomin-D** · 2026-01-28
> 
> > Thank you for your reply. In fact, when I trained dflash using only two repeated data samples, the loss could be overfitted to zero, but I still couldn't achieve a satisfactory acceptance rate during dflash inference. This doesn't seem to be an issue with training techniques or the loss function; there must be some bug causing the misalignment.
> 
> I don’t think this should be considered a bug unless a concrete implementation issue is found. Given the current differences between training and inference losses and the lack of a reference implementation, this could be an implementation or objective mismatch rather than a correctness issue.
> 
> Maybe you can wait for the official release of the training details.

> **ggg-s** · 2026-02-02
> 
> I trained the DFlash model using the latest SpecForge code, and it converged normally during training. However, the inference performance is surprisingly poor with an acceptance step of 1.02, which is quite strange. What could be the reason for this discrepancy? hi [@baifanxxx](https://github.com/baifanxxx) Have you resolved this problem?

> **Ximingwang-09** · 2026-02-02
> 
> > 我使用最新的 SpecForge 代码训练了 DFlash 模型，训练过程正常收敛。然而，推理性能却出奇地差，接受步长设置为 1.02，这非常奇怪。造成这种差异的原因可能是什么？[@baifanxxx](https://github.com/baifanxxx)这个问题解决了吗？
> 
> Same problem.  
> I tried using the data from [https://www.modelscope.cn/models/eigen-ai-labs/qwen3-8b\_dflash\_regen/files](https://www.modelscope.cn/models/eigen-ai-labs/qwen3-8b_dflash_regen/files) to train a qwen3-8b-dflash model. The training results show the accuracy can reach 70%+, and the loss also converges normally. However, when I test the trained model, I find that the generated output length is almost 0. What could be the possible reason? The sglang code I used is: [https://github.com/eigen-ai-labs/sglang-public/tree/release-dflash](https://github.com/eigen-ai-labs/sglang-public/tree/release-dflash)
> 
> Training script:
> 
> ```
> torchrun \
>     --standalone \
>     --nproc_per_node 8 \
>     $ROOT_DIR/scripts/train_dflash.py \
>     --target-model-path /root/Qwen3-8B \
>     --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
>     --train-data-path /mnt4/data/eagle/sharegpt_train_regenerated.jsonl \
>     --output-dir /mnt4/data/qwen3-8b-dflash-offline-0131 \
>     --target-model-backend sglang \
>     --attention-backend flex_attention \
>     --num-epochs 20 \
>     --block-size 16 \
>     --batch-size 4 \
>     --learning-rate 1e-4 \
>     --max-length 4096 \
>     --chat-template qwen \
>     --log-interval 50 \
>     --save-interval 1000 \
>     --report-to tensorboard
> ```
> 
> [![image](https://private-user-images.githubusercontent.com/72070413/543553483-9512afe9-c84b-4b32-9ef9-a198caa2768a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MjgsIm5iZiI6MTc3NjQxNDYyOCwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM0ODMtOTUxMmFmZTktYzg0Yi00YjMyLTllZjktYTE5OGNhYTI3NjhhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTk0YTQ0NDY4OTBkYjk1NmYyZDQwMGUyNmY2ZGJmYTQyYjVhZjM0ZGVhMzBkMzU4YWZmNDNkMmRkYWQwMDgyOTYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.ja5BIHMl6ECZ8KWTGO2od5q0rQll7xXuoqMTDVGgBHo)](https://private-user-images.githubusercontent.com/72070413/543553483-9512afe9-c84b-4b32-9ef9-a198caa2768a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MjgsIm5iZiI6MTc3NjQxNDYyOCwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM0ODMtOTUxMmFmZTktYzg0Yi00YjMyLTllZjktYTE5OGNhYTI3NjhhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTk0YTQ0NDY4OTBkYjk1NmYyZDQwMGUyNmY2ZGJmYTQyYjVhZjM0ZGVhMzBkMzU4YWZmNDNkMmRkYWQwMDgyOTYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.ja5BIHMl6ECZ8KWTGO2od5q0rQll7xXuoqMTDVGgBHo) [![image](https://private-user-images.githubusercontent.com/72070413/543553706-7a47066f-f341-4202-82d2-a9b70f3e6aed.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MjgsIm5iZiI6MTc3NjQxNDYyOCwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM3MDYtN2E0NzA2NmYtZjM0MS00MjAyLTgyZDItYTliNzBmM2U2YWVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVkNTM4MjRjYjg4YzY2NmViZGNjNjdmNGZkNjZiYzYwOWE1ZmNkM2YzY2FiN2JmZmU3NjcwNWI2MTc4MjBhZWMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.n0dGpgaXsz2xR1smVNhaK1CXgzazuOSxyLsuy0Vz2bA)](https://private-user-images.githubusercontent.com/72070413/543553706-7a47066f-f341-4202-82d2-a9b70f3e6aed.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MjgsIm5iZiI6MTc3NjQxNDYyOCwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM3MDYtN2E0NzA2NmYtZjM0MS00MjAyLTgyZDItYTliNzBmM2U2YWVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVkNTM4MjRjYjg4YzY2NmViZGNjNjdmNGZkNjZiYzYwOWE1ZmNkM2YzY2FiN2JmZmU3NjcwNWI2MTc4MjBhZWMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.n0dGpgaXsz2xR1smVNhaK1CXgzazuOSxyLsuy0Vz2bA)

> **xiaomin-D** · 2026-02-02
> 
> > > 我使用最新的 SpecForge 代码训练了 DFlash 模型，训练过程正常收敛。然而，推理性能却出奇地差，接受步长设置为 1.02，这非常奇怪。造成这种差异的原因可能是什么？[@baifanxxx](https://github.com/baifanxxx)这个问题解决了吗？
> > 
> > Same problem. I tried using the data from [https://www.modelscope.cn/models/eigen-ai-labs/qwen3-8b\_dflash\_regen/files](https://www.modelscope.cn/models/eigen-ai-labs/qwen3-8b_dflash_regen/files) to train a qwen3-8b-dflash model. The training results show the accuracy can reach 70%+, and the loss also converges normally. However, when I test the trained model, I find that the generated output length is almost 0. What could be the possible reason? The sglang code I used is: [https://github.com/eigen-ai-labs/sglang-public/tree/release-dflash](https://github.com/eigen-ai-labs/sglang-public/tree/release-dflash)
> > 
> > Training script:
> > 
> > ```
> > torchrun \
> >     --standalone \
> >     --nproc_per_node 8 \
> >     $ROOT_DIR/scripts/train_dflash.py \
> >     --target-model-path /root/Qwen3-8B \
> >     --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
> >     --train-data-path /mnt4/data/eagle/sharegpt_train_regenerated.jsonl \
> >     --output-dir /mnt4/data/qwen3-8b-dflash-offline-0131 \
> >     --target-model-backend sglang \
> >     --attention-backend flex_attention \
> >     --num-epochs 20 \
> >     --block-size 16 \
> >     --batch-size 4 \
> >     --learning-rate 1e-4 \
> >     --max-length 4096 \
> >     --chat-template qwen \
> >     --log-interval 50 \
> >     --save-interval 1000 \
> >     --report-to tensorboard
> > ```
> > 
> > [![image](https://private-user-images.githubusercontent.com/72070413/543553483-9512afe9-c84b-4b32-9ef9-a198caa2768a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzAwMDY1NTUsIm5iZiI6MTc3MDAwNjI1NSwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM0ODMtOTUxMmFmZTktYzg0Yi00YjMyLTllZjktYTE5OGNhYTI3NjhhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjAyVDA0MjQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhNGZkMmM4ZjJiOTAxYjRlYTk5YjdmNzMxN2MyNzYxNTNiODYzYzgyMzc3OThkNmQwYWJmYmFmOGU3ZTY2ZmMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.aQ809xW1eomsJHDBdOosc8TeEAYFejIiq7Q9IhRDAaw)](https://private-user-images.githubusercontent.com/72070413/543553483-9512afe9-c84b-4b32-9ef9-a198caa2768a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzAwMDY1NTUsIm5iZiI6MTc3MDAwNjI1NSwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM0ODMtOTUxMmFmZTktYzg0Yi00YjMyLTllZjktYTE5OGNhYTI3NjhhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjAyVDA0MjQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhNGZkMmM4ZjJiOTAxYjRlYTk5YjdmNzMxN2MyNzYxNTNiODYzYzgyMzc3OThkNmQwYWJmYmFmOGU3ZTY2ZmMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.aQ809xW1eomsJHDBdOosc8TeEAYFejIiq7Q9IhRDAaw) [![image](https://private-user-images.githubusercontent.com/72070413/543553706-7a47066f-f341-4202-82d2-a9b70f3e6aed.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzAwMDY1NTUsIm5iZiI6MTc3MDAwNjI1NSwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM3MDYtN2E0NzA2NmYtZjM0MS00MjAyLTgyZDItYTliNzBmM2U2YWVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjAyVDA0MjQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZmZmUzNzE4ZmVkNGVmYjIxYzhlMzU4YTQ0MWYzYzg0MDNhOGVmMjRhNTgyZGQwOWQxYzk5YjAyNzVjNThhN2ImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.BzLj2Bin_ckb-G1v1ZG5U7pA12MR5GIazLG6crN-zuU)](https://private-user-images.githubusercontent.com/72070413/543553706-7a47066f-f341-4202-82d2-a9b70f3e6aed.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzAwMDY1NTUsIm5iZiI6MTc3MDAwNjI1NSwicGF0aCI6Ii83MjA3MDQxMy81NDM1NTM3MDYtN2E0NzA2NmYtZjM0MS00MjAyLTgyZDItYTliNzBmM2U2YWVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjAyVDA0MjQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZmZmUzNzE4ZmVkNGVmYjIxYzhlMzU4YTQ0MWYzYzg0MDNhOGVmMjRhNTgyZGQwOWQxYzk5YjAyNzVjNThhN2ImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.BzLj2Bin_ckb-G1v1ZG5U7pA12MR5GIazLG6crN-zuU)
> 
> The poor performance of sglang is likely due to inconsistencies in template training and inference.
> 
> The specific training formula is currently unknown; after trying several methods, I could only achieve a maximum of 3 tokens. You could try different approaches to improve the acceptance rate and we can discuss it together.

> **ggg-s** · 2026-02-02
> 
> hi [@xiaomin-D](https://github.com/xiaomin-D) Even when trying inference with the same chat template, the acceptance rate is still very low.

> **wengsnow** · 2026-02-02
> 
> > I trained the DFlash model using the latest SpecForge code, and it converged normally during training. However, the inference performance is surprisingly poor with an acceptance step of 1.02, which is quite strange. What could be the reason for this discrepancy? hi [@baifanxxx](https://github.com/baifanxxx) Have you resolved this problem?
> 
> I found that the training dataset significantly impacts acceptance length. Using ShareGPT v4.3 + UltraChat 200k and disabling the thinking model during regeneration improved the acceptance length to 1.6–2.5. However, a large gap remains compared to official weights.

> **baifanxxx** · 2026-02-02
> 
> I don't think the dataset is a reason that should be considered. Since the size of the dataset affects the training cost, we should prioritize ensuring data consistency between training and inference to identify issues in other areas. I tried creating a dataset by repeatedly using a single sample multiple times to train the model, which would cause the model to overfit to the sample where the training loss becomes 0. Even in this case, when using the same dataset for inference, it was difficult to test any valid acceptance rate.

> **wengsnow** · 2026-02-02
> 
> > I don't think the dataset is a reason that should be considered. Since the size of the dataset affects the training cost, we should prioritize ensuring data consistency between training and inference to identify issues in other areas. I tried creating a dataset by repeatedly using a single sample multiple times to train the model, which would cause the model to overfit to the sample where the training loss becomes 0. Even in this case, when using the same dataset for inference, it was difficult to test any valid acceptance rate.
> 
> Thanks for the reply! I’ve actually run similar experiments. I trained on 10k samples and tested on same data, but the accept length varied a lot by dataset: ShareGPT gave ~1.5, UltraChat ~2.5, and our internal business data hit ~5.0. You can try testing with different datasets as well.
> 
> It seems SpecForge can successfully train a DFlash drafter, but without the official training code, there's still a performance gap. Also, I’ve verified with official weights that once the chat\_template is aligned, [SGLang's DFlash implementation](https://github.com/sgl-project/sglang/pull/16818) matches the official Transformer performance.
> 
> I'm still new to drafter training, especially regarding dLLM training, so I’d love to keep discussing this with everyone.

> **ggg-s** · 2026-02-02
> 
> [@wengsnow](https://github.com/wengsnow) Hi! Thanks for sharing. I tried “aligning the chat template” but my DFlash accept length is still low. Could you clarify what exactly you did for chat\_template alignment?

> **ggg-s** · 2026-02-10
> 
> hi [@xiaomin-D](https://github.com/xiaomin-D), I’m still seeing very low acceptance rates(1.04) and severe overfitting when training with your latest code, even though I used 40k Chinese samples. Do you have any idea what might be causing this?

> **xiaomin-D** · 2026-02-10
> 
> > hi [@xiaomin-D](https://github.com/xiaomin-D), I’m still seeing very low acceptance rates(1.04) and severe overfitting when training with your latest code, even though I used 40k Chinese samples. Do you have any idea what might be causing this?
> 
> Are your tokenID aligned?

> **ggg-s** · 2026-02-10
> 
> Thanks for the suggestion.  
> I’m using your code as-is without any modifications, including the tokenizer setup. Could you clarify where token ID misalignment might occur in this pipeline?

> **xiaomin-D** · 2026-02-10
> 
> > Thanks for the suggestion. I’m using your code as-is without any modifications, including the tokenizer setup. Could you clarify where token ID misalignment might occur in this pipeline?
> 
> in training config：  
> "dflash\_config": {  
> "mask\_token\_id": 151669,  
> "target\_layer\_ids": \[1, 9, 17, 25, 33\]  
> },
> 
> while in inference ，also should use '151669'

> **ggg-s** · 2026-02-10
> 
> I didn’t explicitly specify these during training. Aren’t they automatically computed and saved in the config? I didn’t make any modifications.  
> I noticed that the loss at epoch 0 is already very low during training, which seems quite strange.
> 
> [![Image](https://private-user-images.githubusercontent.com/71912051/547459299-8c47427c-8d3a-46b5-a122-2add036e4dc0.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MzIsIm5iZiI6MTc3NjQxNDYzMiwicGF0aCI6Ii83MTkxMjA1MS81NDc0NTkyOTktOGM0NzQyN2MtOGQzYS00NmI1LWExMjItMmFkZDAzNmU0ZGMwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzMmM0MjMwYWRiZDNlMzdmZDgwZTI5YzQ4NzViYjNiOWMyOTQzNzg3MzY1OWYyM2E5MWFkMDQ0NGY0ZmNiMWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.y-cLxY8t3Tc0FO-rqLTTfAp8IL4mU-GbJ4hkCdX8PCs)](https://private-user-images.githubusercontent.com/71912051/547459299-8c47427c-8d3a-46b5-a122-2add036e4dc0.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzY0MTQ5MzIsIm5iZiI6MTc3NjQxNDYzMiwicGF0aCI6Ii83MTkxMjA1MS81NDc0NTkyOTktOGM0NzQyN2MtOGQzYS00NmI1LWExMjItMmFkZDAzNmU0ZGMwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDE3VDA4MzAzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzMmM0MjMwYWRiZDNlMzdmZDgwZTI5YzQ4NzViYjNiOWMyOTQzNzg3MzY1OWYyM2E5MWFkMDQ0NGY0ZmNiMWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.y-cLxY8t3Tc0FO-rqLTTfAp8IL4mU-GbJ4hkCdX8PCs)

> **xiaomin-D** · 2026-02-10
> 
> > I didn’t explicitly specify these during training. Aren’t they automatically computed and saved in the config? I didn’t make any modifications. I noticed that the loss at epoch 0 is already very low during training, which seems quite strange.
> > 
> > [![Image](https://private-user-images.githubusercontent.com/71912051/547459299-8c47427c-8d3a-46b5-a122-2add036e4dc0.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA2OTcwODMsIm5iZiI6MTc3MDY5Njc4MywicGF0aCI6Ii83MTkxMjA1MS81NDc0NTkyOTktOGM0NzQyN2MtOGQzYS00NmI1LWExMjItMmFkZDAzNmU0ZGMwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjEwVDA0MTMwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUxMDEwNzdmN2ZkM2MxOTU0ZmMwNmMyNmQyZGM4NGIzOTVlZjBlMGE2Njk5Mjc1NDQ3ZTA3ZDkwNjA1YjY0NGMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hTmJqjDSiFVDPrjKEi_BHi6odMNU4HslXKP7KNI1VjI)](https://private-user-images.githubusercontent.com/71912051/547459299-8c47427c-8d3a-46b5-a122-2add036e4dc0.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA2OTcwODMsIm5iZiI6MTc3MDY5Njc4MywicGF0aCI6Ii83MTkxMjA1MS81NDc0NTkyOTktOGM0NzQyN2MtOGQzYS00NmI1LWExMjItMmFkZDAzNmU0ZGMwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjEwVDA0MTMwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUxMDEwNzdmN2ZkM2MxOTU0ZmMwNmMyNmQyZGM4NGIzOTVlZjBlMGE2Njk5Mjc1NDQ3ZTA3ZDkwNjA1YjY0NGMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hTmJqjDSiFVDPrjKEi_BHi6odMNU4HslXKP7KNI1VjI)
> 
> I think this is normal.

> **ggg-s** · 2026-02-12
> 
> hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?

> **yuyangxie96** · 2026-02-12
> 
> > hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?  
> > [@ggg-s](https://github.com/ggg-s)  
> > **1.02 ~ 1.05**，almost the same setting in [#465](https://github.com/sgl-project/SpecForge/issues/465) .  
> > The modifications I made were limited to turning off thinking. I checked that the regenerated data is normal, the tokenizer is fine during training, and I also turned off thinking during inference, but the acceptance length is very poor.

> **ggg-s** · 2026-02-12
> 
> > > hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?  
> > > [@ggg-s](https://github.com/ggg-s)  
> > > **1.02 ~ 1.05**，almost the same setting in [#465](https://github.com/sgl-project/SpecForge/issues/465) .  
> > > The modifications I made were limited to turning off thinking. I checked that the regenerated data is normal, the tokenizer is fine during training, and I also turned off thinking during inference, but the acceptance length is very poor.
> 
> Which dataset are you using? In [#465](https://github.com/sgl-project/SpecForge/issues/465), the data used was with "thinking" enabled. You must also enable "thinking" during inference to achieve a better acceptance length.

> **yuyangxie96** · 2026-02-12
> 
> > > > hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?  
> > > > [@ggg-s](https://github.com/ggg-s)  
> > > > **1.02 ~ 1.05**，almost the same setting in [#465](https://github.com/sgl-project/SpecForge/issues/465) .  
> > > > The modifications I made were limited to turning off thinking. I checked that the regenerated data is normal, the tokenizer is fine during training, and I also turned off thinking during inference, but the acceptance length is very poor.
> > 
> > Which dataset are you using? In [#465](https://github.com/sgl-project/SpecForge/issues/465), the data used was with "thinking" enabled. You must also enable "thinking" during inference to achieve a better acceptance length.
> 
> [@ggg-s](https://github.com/ggg-s) I use the perfectblend dataset, and to regenerate data with the thinking close. After regenerating the data, I also check the jsonl file. I turned off thinking throughout the entire process, including inference. I don’t think this has much impact (since keeping thinking slows down experiment iterations too much).

> **ggg-s** · 2026-02-12
> 
> hi [@yuyangxie96](https://github.com/yuyangxie96) That's really weird. The code is the same, and the data is the same. By the way, didn't you test their model's performance?

> **yuyangxie96** · 2026-02-12
> 
> > hi [@yuyangxie96](https://github.com/yuyangxie96) That's really weird. The code is the same, and the data is the same. By the way, didn't you test their model's performance?
> 
> [@ggg-s](https://github.com/ggg-s) I’m not sure if you’ve successfully reproduced the results in the paper, or at least achieved somewhat reasonable acceptance lengths. In my case, both training loss and accuracy look very good, but inference performs poorly. The inference script is exactly the one from the official dflash repo, and when I use it to test the checkpoint released by the authors, the acceptance length is largely consistent with what’s reported in the paper.

> **wengsnow** · 2026-02-13
> 
> > > > hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?  
> > > > [@ggg-s](https://github.com/ggg-s)  
> > > > **1.02 ~ 1.05**，almost the same setting in [#465](https://github.com/sgl-project/SpecForge/issues/465) .  
> > > > The modifications I made were limited to turning off thinking. I checked that the regenerated data is normal, the tokenizer is fine during training, and I also turned off thinking during inference, but the acceptance length is very poor.
> > 
> > Which dataset are you using? In [#465](https://github.com/sgl-project/SpecForge/issues/465), the data used was with "thinking" enabled. You must also enable "thinking" during inference to achieve a better acceptance length.
> 
> I used the full perfectblend datasets and regenerate without thinking mode. After trained two epoches, get the relatively normal accept lengths：GSM8K: 2.92, Alpaca: 2.14, HumanEval: 4.50, MT-Bench: 2.50. The code is the old version without sampling anchor and loss decay.

> **yuyangxie96** · 2026-02-13
> 
> > > > > hi [@jiahe7ay](https://github.com/jiahe7ay) [@baifanxxx](https://github.com/baifanxxx) [@wengsnow](https://github.com/wengsnow) What average acceptance length can you achieve after training and evaluation?  
> > > > > [@ggg-s](https://github.com/ggg-s)  
> > > > > **1.02 ~ 1.05**，almost the same setting in [#465](https://github.com/sgl-project/SpecForge/issues/465) .  
> > > > > The modifications I made were limited to turning off thinking. I checked that the regenerated data is normal, the tokenizer is fine during training, and I also turned off thinking during inference, but the acceptance length is very poor.
> > > 
> > > Which dataset are you using? In [#465](https://github.com/sgl-project/SpecForge/issues/465), the data used was with "thinking" enabled. You must also enable "thinking" during inference to achieve a better acceptance length.
> > 
> > I used the full perfectblend datasets and regenerate without thinking mode. After trained two epoches, get the relatively normal accept lengths：GSM8K: 2.92, Alpaca: 2.14, HumanEval: 4.50, MT-Bench: 2.50. The code is the old version without sampling anchor and loss decay.
> 
> [@wengsnow](https://github.com/wengsnow)  
> Thanks for sharing. So I’d like to ask, are you running the Commit [b85f89c](https://github.com/sgl-project/SpecForge/commit/b85f89cd69efc2f08796c6b9a9e3c692ea918ec0) version?
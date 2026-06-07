

## 8. 模块到算子的总表

|模块/步骤|代码位置|使用的主要算子/接口|
|---|---|---|
|Embedding|`DeepseekV4Model.embed_input_ids`|`VocabParallelEmbedding`|
|模型级 HC 展开|`hidden_states.unsqueeze(1).repeat(...)`|PyTorch tensor op|
|Layer mHC pre|`DeepseekV2DecoderLayer.hc_pre`|`torch.ops._C_ascend.npu_hc_pre`|
|Layer mHC post|`DeepseekV2DecoderLayer.hc_post`|`torch.ops._C_ascend.npu_hc_post`|
|Q projection|`_forward_prefill/_forward_decode`|`wq_a`、`RMSNorm`、`wq_b`、`triton_q_rms`|
|Decode Q quant|`_forward_decode`|`npu_rms_norm_dynamic_quant`、`torch_npu.npu_quant_matmul`|
|Q/KV RoPE|`_forward_prefill/_forward_decode`|`torch.ops._C_ascend.inplace_partial_rotary_mul`|
|SWA KV 写入|`_forward_prefill/_forward_decode`|`torch_npu.npu_scatter_nd_update_`|
|SWA KV 整理|`_forward_prefill`|`cat_swa_to_kv` / `pad_to_blocks`|
|CSA/C4 Indexer|`indexer_select_qli`|`npu_quant_matmul`、`inplace_partial_rotary_mul`、`compressor`、`npu_dynamic_quant`、`npu_quant_lightning_indexer`|
|HCA/C4/C128 Compressor|`_forward_prefill/_forward_decode`|`torch.ops._C_ascend.compressor`|
|Sparse attention|`_forward_prefill/_forward_decode`|`torch.ops._C_ascend.npu_sparse_attn_sharedkv`|
|Attention output inverse RoPE|`AscendDSAImpl.forward`|`inplace_partial_rotary_mul(..., -sin)`|
|O projection A|`AscendDSAImpl.forward`|`torch_npu.npu_transpose_batchmatmul` 或 `wo_a`|
|O projection B|`AscendDSAImpl.forward`|`wo_b` / `RowParallelLinear`|
|MoE router|`DeepseekV4MoE.forward`|`F.linear`|
|MoE experts|`DeepseekV4MoE.forward`|`SharedFusedMoE` 内部 fused/quant MoE|
|MoE 后通信|`DeepseekV4MoE.forward`|`tensor_model_parallel_all_gather` / allreduce|
|模型级 hc head|`DeepseekV4Model.hc_head`|`torch.rsqrt`、`torch.nn.functional.linear`、`torch.sigmoid`、`torch.sum`|
|Final norm|`DeepseekV4Model.forward`|`RMSNorm`|

---
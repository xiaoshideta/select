import torch

# 加载模型文件
model_path = '/home/xiaozhongyu/CV/acm_mm/RGBX_kl_re_initial/pretrained/pre/epoch-231-depth-38.52.pth'
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 查看模型的状态字典中的键
print("Keys in the state dict:")
for key in model_state_dict.keys():
    print(key)

# 如果模型文件中还包含其他内容（例如优化器状态），可以同样查看它们
# 例如，如果模型文件是使用 torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path) 保存的
if isinstance(model_state_dict, dict) and 'model' in model_state_dict:
    model_keys = model_state_dict['model'].keys()
    print("\nModel keys:")
    for key in model_keys:
        print(key)
    # 类似地查看优化器状态
    if 'optimizer' in model_state_dict:
        optimizer_keys = model_state_dict['optimizer'].keys()
        print("\nOptimizer keys:")
        for key in optimizer_keys:
            print(key)

# 如果需要查看具体参数的尺寸，可以这样做
print("\nParameter shapes:")
# for key, value in model_state_dict.items():
#     print(f"{key}: {value.size()}")
breakpoint()





# single_model_b2
# EncoderDecoder2(
#   (backbone): single_mit_b2(
#     (patch_embed1): OverlapPatchEmbed(
#       (proj): Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
#       (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     )
#     (patch_embed2): OverlapPatchEmbed(
#       (proj): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
#     )
#     (patch_embed3): OverlapPatchEmbed(
#       (proj): Conv2d(128, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#     )
#     (patch_embed4): OverlapPatchEmbed(
#       (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#     )
#     (block1): ModuleList(
#       (0): Block(
#         (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=64, out_features=64, bias=True)
#           (kv): Linear(in_features=64, out_features=128, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=64, out_features=64, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
#           (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): Identity()
#         (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=64, out_features=256, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=256, out_features=64, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (1): Block(
#         (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=64, out_features=64, bias=True)
#           (kv): Linear(in_features=64, out_features=128, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=64, out_features=64, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
#           (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.007)
#         (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=64, out_features=256, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=256, out_features=64, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (2): Block(
#         (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=64, out_features=64, bias=True)
#           (kv): Linear(in_features=64, out_features=128, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=64, out_features=64, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
#           (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.013)
#         (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=64, out_features=256, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=256, out_features=64, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
#     (block2): ModuleList(
#       (0): Block(
#         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=128, out_features=128, bias=True)
#           (kv): Linear(in_features=128, out_features=256, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=128, out_features=128, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
#           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.020)
#         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=128, out_features=512, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=512, out_features=128, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (1): Block(
#         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=128, out_features=128, bias=True)
#           (kv): Linear(in_features=128, out_features=256, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=128, out_features=128, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
#           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.020)
#         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=128, out_features=512, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=512, out_features=128, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (2): Block(
#         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=128, out_features=128, bias=True)
#           (kv): Linear(in_features=128, out_features=256, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=128, out_features=128, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
#           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.020)
#         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=128, out_features=512, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=512, out_features=128, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (3): Block(
#         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=128, out_features=128, bias=True)
#           (kv): Linear(in_features=128, out_features=256, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=128, out_features=128, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
#           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.020)
#         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=128, out_features=512, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=512, out_features=128, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
#     (block3): ModuleList(
#       (0): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.047)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (1): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.053)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (2): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.060)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (3): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.067)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (4): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.073)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (5): Block(
#         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=320, out_features=320, bias=True)
#           (kv): Linear(in_features=320, out_features=640, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=320, out_features=320, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
#           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#         (drop_path): DropPath(drop_prob=0.080)
#         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=320, out_features=1280, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=1280, out_features=320, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (norm3): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
#     (block4): ModuleList(
#       (0): Block(
#         (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=512, out_features=512, bias=True)
#           (kv): Linear(in_features=512, out_features=1024, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=512, out_features=512, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#         )
#         (drop_path): DropPath(drop_prob=0.087)
#         (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (1): Block(
#         (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=512, out_features=512, bias=True)
#           (kv): Linear(in_features=512, out_features=1024, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=512, out_features=512, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#         )
#         (drop_path): DropPath(drop_prob=0.093)
#         (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (2): Block(
#         (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (q): Linear(in_features=512, out_features=512, bias=True)
#           (kv): Linear(in_features=512, out_features=1024, bias=True)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=512, out_features=512, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#         )
#         (drop_path): DropPath(drop_prob=0.100)
#         (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (dwconv): DWConv(
#             (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
#           )
#           (act): GELU()
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (drop): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
#   )
#   (decode_head): DecoderHead(
#     (dropout): Dropout2d(p=0.1, inplace=False)
#     (linear_c4): MLP(
#       (proj): Linear(in_features=512, out_features=512, bias=True)
#     )
#     (linear_c3): MLP(
#       (proj): Linear(in_features=320, out_features=512, bias=True)
#     )
#     (linear_c2): MLP(
#       (proj): Linear(in_features=128, out_features=512, bias=True)
#     )
#     (linear_c1): MLP(
#       (proj): Linear(in_features=64, out_features=512, bias=True)
#     )
#     (linear_fuse): Sequential(
#       (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
#       (1): SyncBatchNorm(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#     )
#     (linear_pred): Conv2d(512, 40, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (criterion): CrossEntropyLoss()
# )


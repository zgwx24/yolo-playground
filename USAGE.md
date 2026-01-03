# YOLO 摄像头检测 - 使用指南

## 快速开始

### 方案 1：纯 YOLO（快速实时检测）✓ 推荐日常使用
```bash
uv run ./main.py
```

**特点：**
- 速度极快（100+ FPS）
- 支持 80 个 COCO 类别
- 实时检测和分类

**参数选项：**
```bash
# 查看所有 80 个支持的类别
uv run ./main.py --show-classes

# 使用更强大的模型
uv run ./main.py --model yolov8l    # large
uv run ./main.py --model yolov8x    # extra-large

# 设置置信度阈值
uv run ./main.py --confidence 0.6

# 保存输出视频
uv run ./main.py --save

# 切换到 CPU（可选）
uv run ./main.py --device cpu
```

---

### 方案 2：SAM 主导（超精准分割）⭐ 推荐高精度需求
```bash
python main_sam.py
```

**特点：**
- **SAM 主导** - 无限制自由分割任意物体
- YOLO 可选 - 可关闭以获得最大性能
- 像素级精度
- 不受 80 个类别限制

**参数选项：**
```bash
# SAM 纯分割模式（不需要 YOLO）
python main_sam.py --no-yolo

# 加上 YOLO 的检测框作为参考
python main_sam.py                    # 默认用 yolov8m + SAM vit_b

# 更精确的 SAM 模型（更慢）
python main_sam.py --sam-model vit_l    # vit_b(快) / vit_l(中) / vit_h(精)

# 调整分割的点数（更多点 = 更多物体）
python main_sam.py --points 10

# 保存高精度分割结果
python main_sam.py --save
```

---

## 按键控制

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `s` | 保存当前帧 |

---

## 模式对比

| 特性 | YOLO 仅 | SAM 纯 | SAM + YOLO |
|------|--------|-------|-----------|
| 速度 | 极快 ⚡⚡⚡ | 中等 ⚡ | 中等 ⚡ |
| 精度 | 中高 | 极高 ✓✓✓ | 极高 ✓✓✓ |
| 类别数 | 80 固定 | 无限 ✓ | 无限 ✓ |
| 分割 | 框 | 像素级 | 像素级 |
| 使用场景 | 日常监控 | 科研精算 | 平衡应用 |

---

## 主要特性：

✅ **SAM 自由分割** → 任意物体无限制  
✅ **无需类别限制** → 不依赖预训练类别  
✅ **YOLO 可选** → 可加可不加  
✅ **实时可视化** → 彩色分割掩码显示  
✅ **灵活配置** → 多种模型大小可选  
✅ **完整文档** → [USAGE.md](USAGE.md) 详细说明  
✅ **已上传 GitHub** → `yolo-playground` 仓库

## 快速开始：

```bash
# SAM 纯分割（最强推荐，无 YOLO 限制）
python main_sam.py --no-yolo

# SAM + YOLO 可选框（参考检测）
python main_sam.py

# 第一次运行会自动下载 SAM 模型（~375MB）
# 按 'q' 退出，'s' 保存帧
```

现在可以做：
- 🚗 自动驾驶级精确分割
- 🏥 医疗图像精细分割（无类别限制）
- 📸 高精度任意物体分离
- 🎬 视频内容智能提取（无类别限制）

---

## GPU 支持

已配置 **CUDA 11** 支持：
- ✓ NVIDIA GTX 1060（和更新的卡）
- ✓ PyTorch 1.13.1
- ✓ 自动使用 GPU 加速

**若要使用 CPU：**
```bash
uv run ./main.py --device cpu
python main_sam.py --device cpu
```

---

## 关键改进：SAM 现在是主导

**新版本亮点：**
- ✅ **SAM 主导地位** - 不再受 YOLO 80 个类别的限制
- ✅ **自由分割** - 使用随机点采样，自动发现图像中的所有物体
- ✅ **YOLO 可选** - 可以 `--no-yolo` 完全关闭，获得最大性能
- ✅ **灵活配置** - 调整 `--points` 参数控制分割细度

**推荐使用方式：**

```bash
# 最强推荐：纯 SAM，无 YOLO 限制
python main_sam.py --no-yolo --sam-model vit_b

# 备选：SAM + YOLO 检测框参考
python main_sam.py --points 8
```

---

## 常见问题

**Q: YOLO + SAM 太慢了？**
A: 正常的。SAM 需要时间。建议用 vit_b + yolov8m 组合。

**Q: 支持自定义类别吗？**
A: 
- YOLO：需要训练自定义模型（有数据的话）
- SAM：不需要，直接分割任意物体

**Q: 内存不足？**
A: 用更小的模型：
```bash
uv run ./main.py --model yolov8n
python main_sam.py --sam-model vit_b
```

---

## 项目结构

```
yolo/
├── main.py           # YOLO 检测（推荐）
├── main_sam.py       # YOLO + SAM 分割
├── pyproject.toml    # 依赖配置
└── README.md
```

---

祝使用愉快！🚀

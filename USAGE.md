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

### 方案 2：YOLO + SAM（超精准分割）⭐ 推荐高精度需求
```bash
python main_sam.py
```

**特点：**
- YOLO 快速定位物体
- SAM 精细分割轮廓
- 无类别限制（可分割任意物体）
- 像素级精度

**参数选项：**
```bash
# 使用更强的 YOLO 模型
python main_sam.py --model yolov8l

# 更精确的 SAM 模型（更慢）
python main_sam.py --sam-model vit_l    # vit_b(快) / vit_l(中) / vit_h(精)

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

## 模型选择指南

### YOLO 模型大小
| 模型 | 大小 | 速度 | 精度 | GPU内存 |
|------|------|------|------|---------|
| yolov8n | 6.3M | 极快 | 低 | 低 |
| yolov8s | 22M | 快 | 中 | 中 |
| **yolov8m** | 49M | 中等 | 中高 | 中 |
| yolov8l | 94M | 较慢 | 高 | 高 |
| yolov8x | 167M | 慢 | 极高 | 极高 |

### SAM 模型大小
| 模型 | 大小 | 速度 | 精度 | GPU内存 |
|------|------|------|------|---------|
| vit_b | 375M | 快 | 中高 | 3-4GB |
| vit_l | 1.2GB | 中等 | 高 | 6-8GB |
| vit_h | 2.5GB | 慢 | 极高 | 12GB+ |

---

## 实际场景推荐

### 📹 实时监控（帧率优先）
```bash
uv run ./main.py --model yolov8n --confidence 0.4
```
→ 最快速度，适合直播

### 🔍 精准检测（精度优先）
```bash
uv run ./main.py --model yolov8x --confidence 0.6
```
→ 最高精度，适合分析

### ✨ 精细分割（混合优先）
```bash
python main_sam.py --model yolov8m --sam-model vit_b
```
→ 平衡速度和精度，适合高质量分割

### 🎬 录制高质量视频
```bash
python main_sam.py --save
```
→ 保存为 `output_sam.mp4`

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

## 类别支持

### YOLO（80 个 COCO 类别）
```bash
uv run ./main.py --show-classes
```

包含：person, car, dog, cat, chair, bottle, laptop, ...

### SAM（无限制）
YOLO 无法识别的物体，SAM 仍可精确分割其轮廓！

---

## 输出说明

### 方案 1（YOLO）
```
Frame: 123 | Detections: 5 | Model: yolov8m
Classes: person(2), car(1), dog(2)
```

### 方案 2（YOLO + SAM）
```
Frame: 123 | Detections: 5 | YOLO: yolov8m + SAM: vit_b
Classes: person(2), car(1), dog(2)
[显示彩色分割掩码]
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

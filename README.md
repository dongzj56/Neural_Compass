# Neural Compass：“神经罗盘”———多模态阿尔茨海默病早筛系统v1.0

本项目提供一个面向阿尔茨海默病（AD）/轻度认知障碍（MCI）早期筛查的原型系统，整合表格临床数据与可选影像信息，包含模型训练、后端推理服务与前端交互界面。前端以温和的可视化风格展示风险结果，并提供简单的问答辅助。

## 功能概览
- 表格分支（`AD_tabular_Branch/`）：TabPFN 等方法的表格数据建模、特征重要性分析与实验脚本
- 影像分支（`AD_Image_Branch/`）：MRI/PET 相关的 3D UNet 等模型与数据处理工具
- 后端服务（`backend/`）：
  - 预测服务（FastAPI，端口 `8000`）：`/predict`、`/predict_async`、`/predict_job/{job_id}`、`/feature_cols`
  - 问答服务（Flask，端口 `5000`）：`/chat`，可与外部大模型 API 交互
- 前端界面（`frontend/`）：`index.html` 首页与 `predict.html` 筛查页，`js/api.js` 负责与后端交互

## 目录结构
- `frontend/` 静态网页与脚本（`index.html`、`predict.html`、`js/api.js`）
- `backend/` 推理与问答服务（`main.py` FastAPI；`app.py` Flask；模型与数据文件）
- `AD_tabular_Branch/` 表格数据建模与分析
- `AD_Image_Branch/` 影像模型与数据集代码

## 环境要求
- Python 3.10+
- 依赖（核心）：`fastapi`、`uvicorn`、`flask`、`flask-cors`、`requests`、`pandas`、`numpy`
- 可选（影像/训练）：`torch` 等

示例安装：
```
pip install fastapi uvicorn flask flask-cors requests pandas numpy
```

## 快速开始
1. 准备模型与数据（后端目录）
   - 将 `tabpfn_model.pkl`、`train.csv`（以及可选的 `test.csv`）放置在 `backend/` 目录
   - 预测服务会读取 `train.csv` 的第 5–40 列作为特征列；可通过 `GET /feature_cols` 查看当前特征列表
2. 启动预测后端（FastAPI，端口 `8000`）
   - 在项目根目录执行：
```
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
3. 配置并启动问答后端（Flask，端口 `5000`）
   - 编辑 `backend/chatbot.py` 中的 `API_KEY`，或在代码中改为从环境变量读取
   - 启动：
```
python backend/app.py
```
4. 打开前端
   - 直接在浏览器打开 `frontend/index.html`（或使用任意静态服务器）；筛查页为 `frontend/predict.html`
   - 前端默认访问 `http://localhost:8000` 的预测接口；`predict.html` 的聊天区使用 `http://127.0.0.1:5000/chat`

## 接口说明
- `GET /`：服务健康检查与说明
- `GET /feature_cols`：返回当前使用的特征列与数量
- `POST /predict`：同步预测
  - 请求体：`{"features": {"AGE": 70, "MMSE": 28, ...}}`（缺失值可用空字符串或 `null`）
  - 返回：`pred`、`pred_label`、`proba`、`classes`、`label_names`、`used_features`、`elapsed`
- `POST /predict_async` → `GET /predict_job/{job_id}`：异步预测与轮询结果

- `POST /chat`（Flask，端口 `5000`）：请求体为 `{"message": "..."}`，返回 `{"response": "..."}`
- `POST /chat`（FastAPI，端口 `8000`）：返回 `{"reply": "..."}`，首页脚本使用该形式

## 安全与隐私
- 请勿在代码仓库中提交任何真实密钥或隐私数据
- 将外部大模型的密钥改为环境变量读取（例如 `VOLC_API_KEY`），并在部署环境中安全注入
- 本项目结果仅用于教育与科研演示，不构成临床诊断或治疗建议

## 许可
- 仅用于学术研究与非商业用途

# backend/main.py
import pickle
import pandas as pd
import numpy as np
import time
from typing import Dict, Optional, List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, threading

try:
    with open("tabpfn_model.pkl", "rb") as f:
        clf = pickle.load(f)
    print("模型 tabpfn_model.pkl 加载成功！")

    print("模型 clf.classes_ =", getattr(clf, "classes_", None))
except FileNotFoundError:
    print("错误：模型文件 tabpfn_model.pkl 未找到。请确保文件存在于 backend 目录下。")
    exit(1)
except Exception as e:
    print(f"错误：加载模型时发生未知错误：{e}")
    exit(1)

try:
    df_train = pd.read_csv("train.csv")
    print("数据文件 train.csv 加载成功！")
except FileNotFoundError:
    print("错误：数据文件 train.csv 未找到。请确保文件存在于 backend 目录下。")
    exit(1) 
except Exception as e:
    print(f"错误：加载 train.csv 时发生未知错误：{e}")
    exit(1)

feature_cols: List[str] = df_train.columns[5:40].tolist()
n_features = len(feature_cols)

print("模型已加载，期望特征列为：")
print(feature_cols)

app = FastAPI(
    title="TabPFN 模型服务",
    description="前端可通过 /predict 调用分类模型",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Any, Dict, Optional, List, Union
from pydantic import BaseModel, root_validator

class PredictRequest(BaseModel):
    features: Dict[str, Any]

    @root_validator(pre=True)
    def wrap_if_plain_features(cls, values):
        if "features" not in values and values:
            return {"features": values}
        return values


class PredictResponse(BaseModel):
    pred: Union[int, str]
    pred_label: str
    proba: List[float]
    classes: List[str]
    label_names: Dict[str, str]
    used_features: List[str]
    elapsed: float

def build_feature_vector(sample_dict: Dict[str, Optional[float]]) -> np.ndarray:
    x_list = []
    for col in feature_cols:
        val = sample_dict.get(col, None)
        if val is None or val == "":
            x_list.append(np.nan)
        else:
            x_list.append(float(val))
    x = np.array(x_list, dtype="float32").reshape(1, -1)
    return x


@app.get("/")
def root():
    return {"message": "TabPFN 后端已启动，可以访问 /docs 查看接口文档。"}


@app.get("/feature_cols")
def get_feature_cols():
    return {
        "feature_cols": feature_cols,
        "n_features": n_features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import time
    start = time.time()

    raw = req.features or {}
    sample_dict: Dict[str, Optional[float]] = {}
    for k, v in raw.items():
        if v is None or v == "":
            sample_dict[k] = None
        else:
            try:
                sample_dict[k] = float(v)
            except (TypeError, ValueError):
                sample_dict[k] = None

    print(f"\n🚀 接收到前端特征字典: {sample_dict}")
    x = build_feature_vector(sample_dict)

    pred_raw = clf.predict(x)[0]
    proba_raw = clf.predict_proba(x)[0]

    if isinstance(pred_raw, (np.generic,)):
        pred = pred_raw.item()
    else:
        pred = pred_raw

    proba = [float(p) for p in proba_raw.tolist()]
    elapsed = time.time() - start

    try:
        classes_raw = getattr(clf, "classes_", None)
        if classes_raw is None:
            classes = [str(i) for i in range(len(proba))]
        else:
            classes = [str(c) for c in classes_raw]
    except Exception:
        classes = [str(i) for i in range(len(proba))]

    known_names = {"CN", "AD", "MCI"}
    if all(c in known_names for c in classes):
        label_names = {c: c for c in classes}
    else:
        default_names = ["CN", "AD", "MCI"]
        label_names = {}
        for i, c in enumerate(classes):
            if i < len(default_names):
                label_names[c] = default_names[i]
            else:
                label_names[c] = f"Class {c}"

    pred_label = label_names.get(str(pred), str(pred))

    print(f"模型预测结果: ")
    print(f"   - 预测类别 (pred): {pred}")
    print(f"   - 预测标签 (pred_label): {pred_label}")
    print(f"   - 预测概率 (proba): {proba}")
    print(f"   - 耗时 (elapsed): {elapsed:.4f} 秒")

    return PredictResponse(
        pred=str(pred),
        pred_label=pred_label,
        proba=proba,
        classes=classes,
        label_names=label_names,
        used_features=feature_cols,
        elapsed=elapsed,
    )

jobs: Dict[str, Dict] = {}

def _run_job(job_id: str, features: Dict[str, Optional[float]]):
    try:
        sample_dict: Dict[str, Optional[float]] = {}
        for k, v in features.items():
            if v is None or v == "":
                sample_dict[k] = None
            else:
                try:
                    sample_dict[k] = float(v)
                except (TypeError, ValueError):
                    sample_dict[k] = None

        print(f"\n异步任务 [{job_id}] 接收到特征字典: {sample_dict}")

        x = build_feature_vector(sample_dict)
        pred_raw = clf.predict(x)[0]
        proba_raw = clf.predict_proba(x)[0]
        pred = pred_raw.item() if isinstance(pred_raw, (np.generic,)) else pred_raw
        proba = [float(p) for p in proba_raw.tolist()]

        try:
            classes_raw = getattr(clf, "classes_", None)
            if classes_raw is None:
                classes = [str(i) for i in range(len(proba))]
            else:
                classes = [str(c) for c in classes_raw]
        except Exception:
            classes = [str(i) for i in range(len(proba))]

        known_names = {"CN", "AD", "MCI"}
        if all(c in known_names for c in classes):
            label_names = {c: c for c in classes}
        else:
            default_names = ["CN", "AD", "MCI"]
            label_names = {}
            for i, c in enumerate(classes):
                if i < len(default_names):
                    label_names[c] = default_names[i]
                else:
                    label_names[c] = f"Class {c}"

        pred_label = label_names.get(str(pred), str(pred))

        print(f"异步任务 [{job_id}] 模型预测结果: ")
        print(f"   - 预测类别 (pred): {pred}")
        print(f"   - 预测标签 (pred_label): {pred_label}")
        print(f"   - 预测概率 (proba): {proba}")

        jobs[job_id] = {
            "status": "done",
            "result": {
                "pred": str(pred),
                "pred_label": pred_label,
                "proba": proba,
                "classes": classes,
                "label_names": label_names
            }
        }
    except Exception as e:
        print(f"异步任务 [{job_id}] 发生错误: {e}")
        jobs[job_id] = {"status": "error", "message": str(e)}

@app.post("/predict_async")
def predict_async(req: PredictRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running"}
    threading.Thread(target=_run_job, args=(job_id, req.features or {}), daemon=True).start()
    return {"job_id": job_id, "status": "running"}

@app.get("/predict_job/{job_id}")
def predict_job(job_id: str):
    return jobs.get(job_id, {"status": "unknown"})

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    q = (req.message or "").lower()
    if any(k in q for k in ["mci", "轻度认知障碍"]):
        return {"reply": "MCI（轻度认知障碍）指存在轻度的认知下降，但对日常生活影响较小。建议完善认知评估（如 MMSE/MoCA）、关注记忆变化并定期随访。"}
    if any(k in q for k in ["ad", "阿尔茨海默", "阿尔茨海默病", "老年痴呆"]):
        return {"reply": "阿尔茨海默病是一种进行性神经退行性疾病，核心表现为记忆障碍与其他认知功能下降。早识别、早干预可延缓进展，建议就诊记忆门诊或神经科。"}
    if any(k in q for k in ["cn", "认知正常"]):
        return {"reply": "CN（认知正常）指标准化评估下认知功能总体在正常范围。仍建议保持良好生活方式并定期随访。"}
    if "mmse" in q:
        return {"reply": "MMSE（简易智力状态检查）满分 30 分，通常以 24/27 等阈值进行分层，需结合年龄、受教育程度综合判断。"}
    return {"reply": "请说明您的问题（如：MCI 定义、AD 的早期表现、MMSE 阈值）。我将基于一般知识为您解答。"}

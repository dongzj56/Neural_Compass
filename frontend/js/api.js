const API_BASE = "http://localhost:8000";

function valNum(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const v = el.value;
  if (v === "" || v === null || v === undefined) return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}

function collectFeatures() {
  return {
    AGE: valNum("age"),
    PTGENDER: valNum("sex"),
    PTEDUCAT: valNum("educationAge"),
    PTETHCAT: valNum("ethnicity"),
    PTRACCAT: valNum("race"),
    PTMARRY: valNum("marital"),
    APOE4: valNum("apoe4"),
    CDRSB: valNum("cdrsb"),
    ADAS11: valNum("adas11"),
    ADAS13: valNum("adas13"),
    ADASQ4: valNum("adasq4"),
    MMSE: valNum("mmse"),
    RAVLT_immediate: valNum("ravlt_immediate"),
    RAVLT_learning: valNum("ravlt_learning"),
    RAVLT_forgetting: valNum("ravlt_forgetting"),
    RAVLT_perc_forgetting: valNum("ravlt_perc_forgetting"),
    LDELTOTAL: valNum("ldeltotal"),
    DIGITSCOR: valNum("digitscor"),
    TRABSCOR: valNum("trabscor"),
    FAQ: valNum("faq"),
    MOCA: valNum("moca"),
    EcogPtMem: valNum("EcogPtMem"),
    EcogPtLang: valNum("EcogPtLang"),
    EcogPtVisspat: valNum("EcogPtVisspat"),
    EcogPtPlan: valNum("EcogPtPlan"),
    EcogPtOrgan: valNum("EcogPtOrgan"),
    EcogPtDivatt: valNum("EcogPtDivatt"),
    EcogPtTotal: valNum("EcogPtTotal"),
    EcogSPMem: valNum("EcogSPMem"),
    EcogSPLang: valNum("EcogSPLang"),
    EcogSPVisspat: valNum("EcogSPVisspat"),
    EcogSPPlan: valNum("EcogSPPlan"),
    EcogSPOrgan: valNum("EcogSPOrgan"),
    EcogSPDivatt: valNum("EcogSPDivatt"),
    EcogSPTotal: valNum("EcogSPTotal")
  };
}

async function startPredictJob(features) {
  const resp = await fetch(`${API_BASE}/predict_async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}
async function getJobStatus(jobId) {
  const resp = await fetch(`${API_BASE}/predict_job/${jobId}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}

function renderResult(res) {
  const levelEl = document.getElementById("riskLevel");
  const chipEl = document.getElementById("riskChip");
  const textEl = document.getElementById("riskText");
  const chart = document.getElementById("probaChart");
  levelEl.style.color = "#111";
  levelEl.style.fontWeight = "700";
  const displayPred = res.pred_label || res.pred;
  levelEl.textContent = `预测结果：${displayPred}`;
  chipEl.textContent = "分类概率";
  chipEl.style.background = "rgba(224, 240, 207, 0.95)";
  chipEl.style.borderColor = "rgba(145, 166, 112, 0.9)";
  if (chart) {
    chart.innerHTML = "";
    const cls = (res.classes || []).map(String);
    const nameMap = res.label_names || {};
    res.proba.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("div"); label.className = "bar-label"; const key = cls[i] ?? String(i); label.textContent = nameMap[key] || `Class ${key}`;
      const bar = document.createElement("div"); bar.className = "bar"; bar.style.width = `${Math.round(p*100)}%`;
      const val = document.createElement("div"); val.className = "bar-val"; val.textContent = `${(p*100).toFixed(1)}%`;
      row.appendChild(label); const track = document.createElement("div"); track.className = "bar-track"; track.appendChild(bar); row.appendChild(track); row.appendChild(val);
      chart.appendChild(row);
    });
  }
  textEl.textContent = "类别说明：CN 为认知正常，MCI 为轻度认知障碍，AD 为阿尔茨海默病风险的痴呆。";
}

function renderError(err) {
  const levelEl = document.getElementById("riskLevel");
  const chipEl = document.getElementById("riskChip");
  const textEl = document.getElementById("riskText");
  levelEl.textContent = "请求失败";
  chipEl.textContent = "请检查后端是否已启动或网络设置";
  chipEl.style.background = "rgba(250, 222, 214, 0.96)";
  chipEl.style.borderColor = "rgba(179, 91, 75, 0.9)";
  textEl.textContent = String(err);
}

function initPredict() {
  const form = document.getElementById("predictForm");
  if (!form) return;
  const wrap = document.getElementById("progressWrap");
  const f1 = document.getElementById("progressFill1");
  const f2 = document.getElementById("progressFill2");
  const f3 = document.getElementById("progressFill3");
  const PHASES = [10000, 100000, 50000];
  const TOTAL_MS = PHASES[0] + PHASES[1] + PHASES[2];

  function startPhased(startTime) {
    let rafId = null;
    const animate = () => {
      const t = Date.now() - startTime;
      const s0 = PHASES[0], s1 = s0 + PHASES[1], s2 = s1 + PHASES[2];
      const w1 = t <= 0 ? 0 : Math.min(100, (Math.max(0, Math.min(t, s0)) / s0) * 100);
      const w2 = t <= s0 ? 0 : Math.min(100, (Math.max(0, Math.min(t - s0, PHASES[1])) / PHASES[1]) * 100);
      const w3 = t <= s1 ? 0 : Math.min(100, (Math.max(0, Math.min(t - s1, PHASES[2])) / PHASES[2]) * 100);
      if (f1) f1.style.width = w1 + "%";
      if (f2) f2.style.width = w2 + "%";
      if (f3) f3.style.width = w3 + "%";
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => { if (rafId) cancelAnimationFrame(rafId); };
  }

  const saved = localStorage.getItem("predictFormState");
  if (saved) {
    try { const st = JSON.parse(saved); Object.keys(st).forEach(id => { const el = document.getElementById(id); if (el) el.value = st[id]; }); } catch {}
  }
  form.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("input", () => {
      const state = {}; form.querySelectorAll("input, select").forEach(e => state[e.id] = e.value);
      localStorage.setItem("predictFormState", JSON.stringify(state));
    });
  });

  const jobRaw = localStorage.getItem("predictJob");
  if (jobRaw) {
    try {
      const job = JSON.parse(jobRaw);
      if (job.status === "pending" && job.job_id) {
        if (wrap) { wrap.style.display = "block"; if (f1) f1.style.width = "0%"; if (f2) f2.style.width = "0%"; if (f3) f3.style.width = "0%"; }
        const stop = startPhased(job.startTime);
        const poll = async () => {
          try {
            const s = await getJobStatus(job.job_id);
            if (s.status === "done" && s.result) {
              stop(); if (f1) f1.style.width = "100%"; if (f2) f2.style.width = "100%"; if (f3) f3.style.width = "100%"; setTimeout(()=>{ if(wrap) wrap.style.display="none"; }, 400);
              renderResult(s.result);
              localStorage.setItem("predictJob", JSON.stringify({ ...job, status: "done", result: s.result }));
            } else if (s.status === "error") {
              stop(); if (wrap) wrap.style.display = "none"; renderError(s.message || "error"); localStorage.removeItem("predictJob");
            } else { setTimeout(poll, 1000); }
          } catch (err) {
            setTimeout(poll, 2000);
          }
        };
        poll();
      } else if (job.status === "done" && job.result) {
        renderResult(job.result);
      }
    } catch {}
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const features = collectFeatures();
    try {
      const started = await startPredictJob(features);
      const job = { startTime: Date.now(), total: TOTAL_MS, status: "pending", job_id: started.job_id };
      localStorage.setItem("predictJob", JSON.stringify(job));
      if (wrap) { wrap.style.display = "block"; if (f1) f1.style.width = "0%"; if (f2) f2.style.width = "0%"; if (f3) f3.style.width = "0%"; }
      const stop = startPhased(job.startTime);
      const poll = async () => {
        try {
          const s = await getJobStatus(job.job_id);
          if (s.status === "done" && s.result) {
            stop(); if (f1) f1.style.width = "100%"; if (f2) f2.style.width = "100%"; if (f3) f3.style.width = "100%"; setTimeout(()=>{ if(wrap) wrap.style.display="none"; }, 400);
            renderResult(s.result);
            localStorage.setItem("predictJob", JSON.stringify({ ...job, status: "done", result: s.result }));
          } else if (s.status === "error") {
            stop(); if (wrap) wrap.style.display = "none"; renderError(s.message || "error"); localStorage.removeItem("predictJob");
          } else { setTimeout(poll, 1000); }
        } catch (err) { setTimeout(poll, 2000); }
      };
      poll();
    } catch (err) {
      if (wrap) wrap.style.display = "none"; renderError(err); localStorage.removeItem("predictJob");
    }
  });
}

function initChat(){
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const list = document.getElementById("chatList");
  if (!form || !input || !list) return;
  function addMsg(role, text){
    const li = document.createElement("div"); li.className = `msg ${role}`; li.textContent = text; list.appendChild(li); list.scrollTop = list.scrollHeight;
  }
  form.addEventListener("submit", async (e)=>{
    e.preventDefault(); const q = input.value.trim(); if (!q) return; addMsg("user", q); input.value="";
    try { const r = await fetch(`${API_BASE}/chat`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({message:q}) }); const data = await r.json(); addMsg("bot", data.reply || "(无回复)"); }
    catch(err){ addMsg("bot", "请求失败，请稍后再试"); }
  });
}

document.addEventListener("DOMContentLoaded", ()=>{ initPredict(); initChat(); });
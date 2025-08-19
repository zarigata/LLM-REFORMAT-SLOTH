const $ = (s) => document.querySelector(s);
const logs = $("#logs");
const summary = $("#summary");
const actions = $("#actions");
const diagOut = $("#diagnosticsOut");
let currentJob = null;
let currentModelId = null;

async function fetchGPU() {
  try {
    const r = await fetch('/api/metrics/gpu');
    const j = await r.json();
    $('#gpu').textContent = `GPU: NVIDIA=${j.nvidia} AMD=${j.amd}`;
  } catch {}
}

async function poll(jobId) {
  const r = await fetch(`/api/status/${jobId}`);
  const j = await r.json();
  logs.textContent = j.logs_tail.join('\n');
  if (j.artifacts && j.artifacts.model_id) currentModelId = j.artifacts.model_id;
  if (j.status === 'done' || j.status === 'error') {
    actions.hidden = false;
    return;
  }
  setTimeout(() => poll(jobId), 750);
}

$('#form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = Object.fromEntries(f.entries());
  body.dry_run = !!body.dry_run;
  body.quantization_target = body.quantization_target || 'int8';
  const r = await fetch('/api/fine-tune', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  currentJob = j.job_id;
  poll(currentJob);
});

$('#exportBtn').addEventListener('click', async () => {
  if (!currentModelId) return;
  const r = await fetch(`/api/export/${currentModelId}`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_id: currentModelId, export_format: 'gguf'})});
  const j = await r.json();
  poll(j.job_id);
  setTimeout(async () => {
    const s = await fetch(`/api/summary/${currentModelId}`);
    summary.textContent = JSON.stringify(await s.json(), null, 2);
  }, 1500);
});

$('#publishBtn').addEventListener('click', async () => {
  if (!currentModelId) return;
  const name = prompt('Ollama model name', 'llm-factory-sample');
  if (!name) return;
  const r = await fetch(`/api/ollama/publish/${currentModelId}`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_id: currentModelId, ollama_name: name, auto_serve: true})});
  const j = await r.json();
  poll(j.job_id);
});

fetchGPU();

$('#diagnoseBtn').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/metrics/diagnose');
    const j = await r.json();
    diagOut.textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    diagOut.textContent = String(e);
  }
});

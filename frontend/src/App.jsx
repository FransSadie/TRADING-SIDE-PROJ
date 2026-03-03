import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

const tabs = [
  { id: "ops", label: "Operations" },
  { id: "model", label: "Model & Predict" },
  { id: "quality", label: "Data Quality" },
  { id: "docs", label: "Docs" }
];

function JsonBlock({ data }) {
  return (
    <pre className="overflow-auto rounded-lg border border-line/70 bg-bg/70 p-3 text-xs text-mute">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function StatCard({ label, value, tone = "ink" }) {
  const toneClass = {
    ink: "text-ink",
    good: "text-good",
    warn: "text-warn",
    bad: "text-bad",
    accent: "text-accent"
  }[tone] || "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel/70 p-4 shadow-glow">
      <div className="text-xs uppercase tracking-wide text-mute">{label}</div>
      <div className={`mt-2 text-2xl font-display font-semibold ${toneClass}`}>{value}</div>
    </div>
  );
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export default function App() {
  const [activeTab, setActiveTab] = useState("ops");
  const [ticker, setTicker] = useState("SPY");
  const [busy, setBusy] = useState({});
  const [messages, setMessages] = useState([]);

  const [health, setHealth] = useState(null);
  const [dataStatus, setDataStatus] = useState(null);
  const [quality, setQuality] = useState(null);
  const [ingestStatus, setIngestStatus] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionLogs, setPredictionLogs] = useState([]);
  const [docs, setDocs] = useState({ readme_markdown: "", project_overview_text: "" });

  const pushMessage = (text, tone = "info") => {
    setMessages((prev) => [{ id: Date.now(), text, tone }, ...prev].slice(0, 8));
  };

  const runAction = async (key, fn) => {
    setBusy((prev) => ({ ...prev, [key]: true }));
    try {
      const data = await fn();
      pushMessage(`${key} completed`, "ok");
      return data;
    } catch (err) {
      pushMessage(`${key} failed: ${err.message}`, "err");
      throw err;
    } finally {
      setBusy((prev) => ({ ...prev, [key]: false }));
    }
  };

  const refreshAll = async () => {
    await Promise.all([
      api("/health").then(setHealth),
      api("/data/status").then(setDataStatus),
      api("/data/quality").then(setQuality),
      api("/ingest/status").then(setIngestStatus),
      api("/pipeline/status").then(setPipelineStatus),
      api("/model/status").then(setModelStatus),
      api("/prediction/logs?limit=100").then((d) => setPredictionLogs(d.rows || [])),
      api("/docs/text").then(setDocs)
    ]);
  };

  useEffect(() => {
    refreshAll().catch((err) => pushMessage(`Initial load failed: ${err.message}`, "err"));
    const t = setInterval(() => {
      refreshAll().catch(() => null);
    }, 15000);
    return () => clearInterval(t);
  }, []);

  const totalRows = useMemo(() => {
    if (!dataStatus?.totals) return 0;
    return Object.values(dataStatus.totals).reduce((a, b) => a + Number(b || 0), 0);
  }, [dataStatus]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_10%_10%,#1e2f5f_0,#0b1020_45%,#050810_100%)] font-body text-ink">
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8 md:py-10">
        <header className="mb-6 rounded-2xl border border-line bg-panel/70 p-5 shadow-glow">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="font-display text-3xl font-semibold">Market Bot Control Deck</h1>
              <p className="mt-1 text-sm text-mute">
                Operate ingestion, pipeline, model training, prediction, and quality checks from one screen.
              </p>
            </div>
            <button
              className="rounded-lg border border-accent/60 bg-accent/20 px-4 py-2 text-sm font-medium hover:bg-accent/30"
              onClick={() => runAction("refresh", refreshAll)}
            >
              {busy.refresh ? "Refreshing..." : "Refresh Now"}
            </button>
          </div>
        </header>

        <section className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Health" value={health?.status || "unknown"} tone={health?.status === "ok" ? "good" : "warn"} />
          <StatCard label="Total Rows" value={totalRows} />
          <StatCard label="Latest Model" value={modelStatus?.metadata?.version_id || "none"} tone="accent" />
          <StatCard
            label="Latest Run"
            value={pipelineStatus?.latest_run?.status || ingestStatus?.latest_run?.status || "none"}
            tone={pipelineStatus?.latest_run?.status === "success" ? "good" : "warn"}
          />
        </section>

        <nav className="mb-5 flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`rounded-lg border px-4 py-2 text-sm ${
                activeTab === tab.id
                  ? "border-accent bg-accent/20 text-ink"
                  : "border-line bg-panel/50 text-mute hover:text-ink"
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === "ops" && (
          <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Pipeline Actions</h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() =>
                    runAction("ingest_run", async () => {
                      const out = await api("/ingest/run", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                >
                  <div className="font-medium">Run Ingestion</div>
                  <div className="text-xs text-mute">News + market pull</div>
                </button>
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() =>
                    runAction("pipeline_run", async () => {
                      const out = await api("/pipeline/run", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                >
                  <div className="font-medium">Run NLP + Features</div>
                  <div className="text-xs text-mute">Signal, feature, label jobs</div>
                </button>
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() =>
                    runAction("model_train", async () => {
                      const out = await api("/model/train", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                >
                  <div className="font-medium">Train Model</div>
                  <div className="text-xs text-mute">Versioned artifact save</div>
                </button>
                <button
                  className="rounded-lg border border-accent/70 bg-accent/20 px-4 py-3 text-left hover:bg-accent/30"
                  onClick={() =>
                    runAction("run_full", async () => {
                      const out = await api("/run/full?train_model=true", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                >
                  <div className="font-medium">Run Full Refresh</div>
                  <div className="text-xs text-mute">Ingest {"->"} pipeline {"->"} train</div>
                </button>
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <JsonBlock data={ingestStatus || {}} />
                <JsonBlock data={pipelineStatus || {}} />
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Operator Feed</h2>
              <div className="mt-4 space-y-2">
                {messages.length === 0 && <div className="text-sm text-mute">No events yet.</div>}
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`rounded-lg border px-3 py-2 text-sm ${
                      m.tone === "err"
                        ? "border-bad/60 bg-bad/10 text-bad"
                        : m.tone === "ok"
                        ? "border-good/60 bg-good/10 text-good"
                        : "border-line bg-bg/50 text-mute"
                    }`}
                  >
                    {m.text}
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {activeTab === "model" && (
          <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Predict</h2>
              <div className="mt-3 flex gap-2">
                <input
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="w-40 rounded-lg border border-line bg-bg/70 px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <button
                  className="rounded-lg border border-accent/60 bg-accent/20 px-4 py-2 text-sm hover:bg-accent/30"
                  onClick={() =>
                    runAction("predict", async () => {
                      const out = await api(`/predict?ticker=${encodeURIComponent(ticker)}`);
                      setPrediction(out);
                      await refreshAll();
                      return out;
                    })
                  }
                >
                  Predict
                </button>
              </div>
              <div className="mt-4">
                <JsonBlock data={prediction || { hint: "Run prediction to view output" }} />
              </div>
              <div className="mt-4">
                <JsonBlock data={modelStatus || {}} />
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Prediction Log</h2>
              <div className="mt-3 max-h-[520px] overflow-auto rounded-lg border border-line">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-bg/90 text-mute">
                    <tr>
                      <th className="px-3 py-2 text-left">Time</th>
                      <th className="px-3 py-2 text-left">Ticker</th>
                      <th className="px-3 py-2 text-left">Pred</th>
                      <th className="px-3 py-2 text-right">Conf</th>
                      <th className="px-3 py-2 text-left">Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionLogs.map((row) => (
                      <tr key={row.id} className="border-t border-line/50">
                        <td className="px-3 py-2">{row.created_at || "-"}</td>
                        <td className="px-3 py-2">{row.ticker}</td>
                        <td className="px-3 py-2">{row.prediction}</td>
                        <td className="px-3 py-2 text-right">{Number(row.confidence || 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{row.model_version || "-"}</td>
                      </tr>
                    ))}
                    {predictionLogs.length === 0 && (
                      <tr>
                        <td className="px-3 py-4 text-mute" colSpan={5}>
                          No prediction logs yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {activeTab === "quality" && (
          <section className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Data Status</h2>
              <div className="mt-4">
                <JsonBlock data={dataStatus || {}} />
              </div>
            </div>
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Data Quality</h2>
              <div className="mt-4">
                <JsonBlock data={quality || {}} />
              </div>
            </div>
          </section>
        )}

        {activeTab === "docs" && (
          <section className="grid gap-6 lg:grid-cols-2">
            <article className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="mb-3 font-display text-xl">README</h2>
              <div className="prose prose-invert max-w-none text-sm prose-headings:font-display">
                <ReactMarkdown>{docs.readme_markdown || "_No README text loaded_"}</ReactMarkdown>
              </div>
            </article>
            <article className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="mb-3 font-display text-xl">Project Overview</h2>
              <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-bg/60 p-3 text-xs text-mute">
                {docs.project_overview_text || "No overview loaded."}
              </pre>
            </article>
          </section>
        )}
      </div>
    </div>
  );
}

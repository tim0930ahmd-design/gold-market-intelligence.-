type Candle = { time: string; open: number; high: number; low: number; close: number };
type Decision = { time: string; symbol: "XAUUSD"; decision: "NO_TRADE"; reason: string; engine_version: string };
const candles: Candle[] = [];
const decisions: Decision[] = [];
const VERSION = "paper-scaffold-0.1.0";
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
const validCandle = (x: any): x is Candle => x && typeof x.time === "string" && ["open","high","low","close"].every(k => Number.isFinite(x[k])) && x.high >= Math.max(x.open,x.close,x.low) && x.low <= Math.min(x.open,x.close,x.high);
const server = Bun.serve({
  port: Number(Bun.env.PORT || 3000),
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === "GET" && url.pathname === "/health") return json({ ok: true, service: "gold-ai-intelligence-paper", mode: "PAPER", engine_version: VERSION });
    if (req.method === "GET" && url.pathname === "/status") return json({ mode: "PAPER", engine_version: VERSION, candles_loaded: candles.length, decisions_recorded: decisions.length, trading_enabled: false, persistence: "memory_only_until_volume_or_database_is_configured" });
    if (req.method === "GET" && url.pathname === "/candles") return json({ count: candles.length, candles: candles.slice(-500) });
    if (req.method === "POST" && url.pathname === "/candles") {
      let body: any; try { body = await req.json(); } catch { return json({ error: "invalid_json" }, 400); }
      const rows = Array.isArray(body) ? body : body?.candles;
      if (!Array.isArray(rows) || rows.length < 1 || rows.length > 10000 || !rows.every(validCandle)) return json({ error: "expected 1..10000 valid OHLC candles" }, 400);
      const byTime = new Map(candles.map(c => [c.time,c]));
      for (const c of rows) byTime.set(c.time,c);
      candles.splice(0,candles.length,...[...byTime.values()].sort((a,b)=>a.time.localeCompare(b.time)).slice(-50000));
      return json({ accepted: rows.length, total: candles.length });
    }
    if (req.method === "POST" && url.pathname === "/analyze") {
      const decision: Decision = { time: new Date().toISOString(), symbol: "XAUUSD", decision: "NO_TRADE", reason: "Research scaffold only: no validated edge/model is deployed; PAPER trading is disabled.", engine_version: VERSION };
      decisions.push(decision);
      if (decisions.length > 1000) decisions.shift();
      return json(decision);
    }
    if (req.method === "GET" && url.pathname === "/") return json({ service: "Gold AI Intelligence PAPER", status: "running", endpoints: ["/health","/status","/candles","/analyze"] });
    return json({ error: "not_found" }, 404);
  }
});
console.log(`Gold AI PAPER service listening on ${server.url}`);

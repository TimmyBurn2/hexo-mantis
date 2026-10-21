// Drives headless Chromium through webgpu_page.mjs on a non-isolated loopback page (no COOP/COEP), sampling the GPU's
// memory with nvidia-smi when it is present; writes the page's result, its console and the samples to OUT.json.
// Usage: node scripts/webgpu.mjs EXPORT_DIR MANIFEST_FILE REFERENCE.json OUT.json
import { execFile } from "node:child_process";
import { readFile, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import puppeteer from "puppeteer-core";

const [exportDir, manifestFile, reference, out] = process.argv.slice(2);
const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const TYPES = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
  ".json": "application/json" };
// The raw binary, not the /usr/bin wrapper with the desk user's flags; without --disable-vulkan-surface headless
// Chromium offers only SwiftShader, and a relaunch within seconds can fall back to it again.
const CHROMIUM = "/usr/lib/chromium/chromium";
const ARGS = ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan",
  "--disable-vulkan-surface", "--ignore-gpu-blocklist"];
const PAGE = '<!doctype html><meta charset="utf-8"><script type="module" src="/scripts/webgpu_page.mjs"></script>';

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path === "/" || path === "/index.html") return res.writeHead(200, { "Content-Type": "text/html" }).end(PAGE);
  const file = path === "/ref.json" ? reference : path.startsWith("/net/") ? normalize(join(exportDir, path.slice(5)))
    : normalize(join(web, path));
  try {
    if (!(await stat(file)).isFile()) throw new Error("not a file");
    res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store" });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

const smi = async () => {
  try {
    const { stdout } = await promisify(execFile)("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]);
    return Number(stdout.trim().split("\n")[0]);
  } catch {
    return null;
  }
};
await new Promise((resolve) => setTimeout(resolve, 8_000));
const baseline = await smi();
const samples = [];
const sampler = setInterval(async () => samples.push([Date.now(), await smi()]), 100);

const browser = await puppeteer.launch({ executablePath: CHROMIUM, headless: true, args: ARGS, protocolTimeout: 3_600_000 });
const page = await browser.newPage();
const consoleLines = [];
const fetched = [];
page.on("console", (m) => consoleLines.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => consoleLines.push(`pageerror: ${e}`));
page.on("requestfinished", (r) => fetched.push(r.url()));
await page.goto(`http://127.0.0.1:${port}/?manifest=${manifestFile}`);
await page.waitForFunction("window.__result !== undefined", { timeout: 3_600_000, polling: 1000 });
const result = await page.evaluate("window.__result");
clearInterval(sampler);
result.chromiumVersion = await browser.version();
await browser.close();
server.close();

const inWindow = samples.filter(([t, mb]) => mb !== null && result.p99 && t >= result.p99.startedAt && t <= result.p99.endedAt);
result.gpuMemoryMiB = { baseline, peakDuringP99: inWindow.length ? Math.max(...inWindow.map(([, mb]) => mb)) : null,
  peakOverall: samples.length ? Math.max(...samples.map(([, mb]) => mb ?? 0)) : null, samples: samples.length };
result.thirdPartyFetches = fetched.filter((u) => !u.startsWith(`http://127.0.0.1:${port}/`));
result.console = consoleLines;
await writeFile(out, JSON.stringify(result, null, 1));
console.log(result.ok ? "ok" : `FAILED ${result.error?.split("\n")[0]}`, JSON.stringify(result.witness ?? {}), JSON.stringify(result.timing ?? {}),
  JSON.stringify(result.gpuMemoryMiB), `third-party fetches: ${result.thirdPartyFetches.length}`);

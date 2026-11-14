// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the trial's brief (the provider directory, the sealed quality table and the
// workload manifest), probes every provider's catalog in its own wire, registers one catalog
// with the env, then serves the client's twenty requests: POST /v1/chat/completions and
// POST /v1/messages, each in JSON or server-sent events, and GET /v1/models in the shape of
// the auth header it arrived with. It polls the env's /work and exits 0 when the work is done.
//
// Routing is where the score is. Quality is 60% of it, so every request goes to the best-rated
// free model that meets the request's requirements — the quality table's rating_milli, never a
// paid model — with ties spread across providers by the calls each has already taken, so one
// provider's per-trial budget is not burned early while another idles. Refusals fail over to
// the next candidate for that request alone; a 402 retires its provider for the trial (the
// free allowance is gone), and three 429s in a row do the same, while a lone 429 is just one
// scheduled bad moment and the provider keeps its place. Failover is 10% and served is 15%:
// a request is only lost when every candidate has refused it.
//
// All three provider wires are spoken. Every client request is first normalized to a Chat
// Completions body, then translated into the routed provider's wire — OpenAI as it stands,
// Anthropic Messages, or Gemini generateContent — and its answer, JSON or event stream, is
// translated back through the same pivot, so an Anthropic client can be served by a Gemini
// provider and the reverse. The translations live in lib/wire.ts; the brief and the catalog
// probe in lib/providers.ts.
//
// Everything the server decides derives from the brief, the probes and the providers' own
// answers: no clock and no randomness in any choice, so a replayed trial reproduces the live
// one byte for byte. `node main.ts --list` prints the catalog document the register body
// carries, and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicToOpenAI,
  geminiRequestFromOpenAI,
  geminiResponseToOpenAI,
  openAIChunkStringsFromJson,
  openAIChunksFromAnthropic,
  openAIChunksFromGemini,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  streamFromStrings,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take to answer with its headers, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls answered so far, per provider id: the spread the routing ties break on. */
const callsMade = new Map<string, number>();
/**
 * Providers out of the rotation: a 402 means the free allowance is gone for the trial, and
 * three 429s in a row mean an outage or an empty budget rather than a bad moment. A single 429
 * retires nobody — the trial's providers refuse single calls on their failure schedules and
 * answer the next one — so strikes count consecutive refusals and any answer clears them.
 */
const downProviders = new Set<string>();
/** Consecutive 429s per provider, cleared by any answer that is not one. */
const strikes = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the request's requirements, best first: by the quality table's
 * rating, then by the calls its provider has already taken (the spread that keeps one budget
 * from draining while another idles), then the brief's provider order and the model id, so the
 * order is a pure function of the brief and the trial so far. Providers that have refused with
 * 402 or 429 queue behind every provider that has not.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const list: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // The anthropic wire has no JSON mode to answer in: a request the client put in JSON mode
    // routed there would come back speaking the wrong shape, so those providers sit it out.
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      list.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const order = new Map(probes.map((probe, index) => [probe.provider.id, index]));
  list.sort(
    (a, b) =>
      b.rating - a.rating ||
      (callsMade.get(a.probe.provider.id) ?? 0) - (callsMade.get(b.probe.provider.id) ?? 0) ||
      (order.get(a.probe.provider.id) ?? 0) - (order.get(b.probe.provider.id) ?? 0) ||
      byId(a.model, b.model),
  );
  return [...list.filter((c) => !downProviders.has(c.probe.provider.id)), ...list.filter((c) => downProviders.has(c.probe.provider.id))];
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A fetch whose wait for the answer's headers is capped; the body then reads at its own pace. */
async function fetchUpstream(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/** One call to a candidate's provider, the request spelled in that provider's own wire. */
async function callUpstream(candidate: Candidate, oaiBody: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    // A streamed answer must end with its usage, and an openai provider only sends the usage
    // chunk when the call asks for it: ask on every streamed call, whatever the client sent.
    const upstreamBody: Json = { ...oaiBody, model: candidate.model.id };
    if (stream) {
      const options = isRecord(oaiBody.stream_options) ? oaiBody.stream_options : {};
      upstreamBody.stream_options = { ...options, include_usage: true };
    }
    return fetchUpstream(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
    });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return fetchUpstream(`${base}/v1/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify(openAIRequestToAnthropic(oaiBody, candidate.model.id)),
    });
  }
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return fetchUpstream(`${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(geminiRequestFromOpenAI(oaiBody)),
  });
}

/** A provider's JSON answer in the OpenAI shape the pivot works in. */
function normalizeJson(wire: 'openai' | 'anthropic' | 'gemini', raw: unknown, model: string): Json {
  const body = isRecord(raw) ? raw : {};
  if (wire === 'anthropic') return anthropicResponseToOpenAI(body, model);
  if (wire === 'gemini') return geminiResponseToOpenAI(body, model);
  return body;
}

/** A provider's event stream as OpenAI chunk strings; null when the wire needs no translation. */
function chunkStreamFor(
  wire: 'openai' | 'anthropic' | 'gemini',
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> | null {
  if (wire === 'anthropic') return openAIChunksFromAnthropic(upstream, model);
  if (wire === 'gemini') return openAIChunksFromGemini(upstream, model);
  return null;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write chunk strings to the client as they come. */
async function writePieces(pieces: AsyncIterable<string>, res: ServerResponse): Promise<void> {
  for await (const piece of pieces) res.write(piece);
  res.end();
}

/**
 * Serve one client request on either wire: normalize it to a Chat Completions body, then walk
 * the ranked candidates until one answers. A 402 or 429 retires its provider for the trial; any
 * other refusal moves to the next candidate. The first provider that answers owns the response:
 * its JSON or its event stream is translated into the client's wire and sent.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const wantStream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : null;
  const oaiBody: Json = clientWire === 'openai' ? body : anthropicToOpenAI(body, '');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = clientWire === 'openai' && (format === 'json_object' || format === 'json_schema');
  const candidates = candidatesFor(requirementsFor(body), jsonMode);
  if (candidates.length === 0) {
    return sendError(
      res,
      clientWire,
      503,
      clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, oaiBody, wantStream);
    } catch {
      continue;
    }
    callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
    if (upstream.status === 402) {
      downProviders.add(candidate.probe.provider.id);
      await upstream.text().catch(() => '');
      continue;
    }
    if (upstream.status === 429) {
      const count = (strikes.get(candidate.probe.provider.id) ?? 0) + 1;
      strikes.set(candidate.probe.provider.id, count);
      if (count >= 3) downProviders.add(candidate.probe.provider.id);
      await upstream.text().catch(() => '');
      continue;
    }
    if (!upstream.ok) {
      await upstream.text().catch(() => '');
      continue;
    }
    strikes.delete(candidate.probe.provider.id);
    const wire = candidate.probe.provider.wire;
    const model = candidate.model.id;
    if (wantStream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const contentType = upstream.headers.get('content-type') ?? '';
      if (contentType.includes('text/event-stream') && upstream.body !== null) {
        const chunks = chunkStreamFor(wire, upstream.body, model);
        if (chunks === null) {
          // An openai provider for an openai client: its stream is already the client's shape.
          if (clientWire === 'openai') await pipeStream(upstream.body, res);
          else await writePieces(anthropicEventsFromOpenAI(upstream.body, requested ?? model), res);
        } else if (clientWire === 'openai') {
          await writePieces(chunks, res);
        } else {
          await writePieces(anthropicEventsFromOpenAI(streamFromStrings(chunks), requested ?? model), res);
        }
      } else {
        // The provider answered JSON to a streaming request: render it as a chunk stream.
        const answer = normalizeJson(wire, await upstream.json().catch(() => null), model);
        const pieces = openAIChunkStringsFromJson(answer, model);
        if (clientWire === 'openai') await writePieces(pieces, res);
        else await writePieces(anthropicEventsFromOpenAI(streamFromStrings(pieces), requested ?? model), res);
      }
      return;
    }
    const answer = normalizeJson(wire, await upstream.json().catch(() => null), model);
    if (clientWire === 'openai') sendJson(res, 200, answer);
    else sendJson(res, 200, openAIToAnthropic(answer, requested ?? model));
    return;
  }
  sendError(
    res,
    clientWire,
    503,
    clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'every free model that meets this request refused it',
  );
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

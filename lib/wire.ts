// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires live at the bottom of this file, past the baseline's own
// translators: an OpenAI-shaped request becomes an Anthropic Messages request or a Gemini
// generateContent request, and their answers and event streams come back as Chat Completions objects
// and chunks, so the one pivot serves every pairing of client wire and provider wire.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The anthropic and gemini upstream wires, pivoted through the OpenAI shape.
//
// main.ts normalizes every client request to a Chat Completions body, so a provider only ever
// needs the OpenAI shape translated into its own wire and its answer translated back:
//   openAIRequestToAnthropic / anthropicResponseToOpenAI / openAIChunksFromAnthropic
//   geminiRequestFromOpenAI  / geminiResponseToOpenAI  / openAIChunksFromGemini
// A provider that answers JSON to a streaming request is served by openAIChunkStringsFromJson,
// which renders the whole answer as the chunk stream a streamed call would have sent. Like the
// rest of the file, everything is a pure function of the provider's answer: no clock, no
// randomness, so a replay reproduces the same bytes.

/** A stream of text pieces as a web ReadableStream, for feeding one translator into another. */
export function streamFromStrings(pieces: AsyncIterable<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const iterator = pieces[Symbol.asyncIterator]();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) controller.close();
      else controller.enqueue(encoder.encode(value));
    },
  });
}

/** The `data:` payloads of an SSE stream, in order, one string per data line. */
async function* ssePayloads(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const payloadsOf = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) out.push(trimmed.slice(5).trim());
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const payload of payloadsOf(frame)) yield payload;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const payload of payloadsOf(buffer)) yield payload;
  } finally {
    reader.releaseLock();
  }
}

/** One Chat Completions chunk as an SSE frame. */
function chunkFrame(id: string, model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, {
    id,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

/** The closing pair of a chunk stream: the usage chunk, then [DONE]. */
function usageFrames(id: string, model: string, input: number, output: number, total: number): string[] {
  return [
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: total },
    }),
    'data: [DONE]\n\n',
  ];
}

// --- Anthropic upstream ---

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: system
 * messages lift into the system field, tool results become tool_result blocks, tool calls
 * become tool_use blocks, image_url parts become image blocks, and consecutive user turns
 * merge, as the Messages API wants alternating roles.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  let system = '';
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    const blocks: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const match = /^data:([^;,]+);base64,(.*)$/.exec(part.image_url.url);
          if (match) blocks.push({ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url: part.image_url.url } });
        }
      }
    }
    messages.push({ role: 'user', content: blocks });
  }
  const merged: Json[] = [];
  for (const message of messages) {
    const prev = merged[merged.length - 1];
    if (prev !== undefined && prev.role === message.role && Array.isArray(prev.content) && Array.isArray(message.content)) {
      prev.content = [...(prev.content as Json[]), ...(message.content as Json[])];
    } else merged.push(message);
  }
  const out: Json = { model, messages: merged };
  // The output cap passes through exactly, and only when the client named one: a cap invented
  // here would not be the cap the client sent, and the answer is checked against the client's.
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as a Chat Completions object naming `model`. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** An Anthropic Messages event stream as Chat Completions chunks. */
export async function* openAIChunksFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let id = 'chatcmpl-stream';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  let roleSent = false;
  let nextTool = 0;
  const toolOrdinal = new Map<number, number>();
  for await (const payload of ssePayloads(upstream)) {
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (event.type === 'message_start' && isRecord(event.message)) {
      if (typeof event.message.id === 'string') id = event.message.id;
      if (isRecord(event.message.usage)) inputTokens = numberOrZero(event.message.usage.input_tokens);
      if (!roleSent) {
        roleSent = true;
        yield chunkFrame(id, model, { role: 'assistant', content: '' }, null);
      }
    } else if (event.type === 'content_block_start' && isRecord(event.content_block) && event.content_block.type === 'tool_use') {
      const ordinal = nextTool;
      nextTool += 1;
      if (typeof event.index === 'number') toolOrdinal.set(event.index, ordinal);
      yield chunkFrame(
        id,
        model,
        {
          tool_calls: [
            {
              index: ordinal,
              id: typeof event.content_block.id === 'string' ? event.content_block.id : `toolu_${ordinal}`,
              type: 'function',
              function: { name: typeof event.content_block.name === 'string' ? event.content_block.name : '', arguments: '' },
            },
          ],
        },
        null,
      );
    } else if (event.type === 'content_block_delta' && isRecord(event.delta)) {
      if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string' && event.delta.text !== '') {
        yield chunkFrame(id, model, { content: event.delta.text }, null);
      } else if (event.delta.type === 'input_json_delta' && typeof event.delta.partial_json === 'string' && event.delta.partial_json !== '') {
        const ordinal = typeof event.index === 'number' ? toolOrdinal.get(event.index) ?? 0 : 0;
        yield chunkFrame(id, model, { tool_calls: [{ index: ordinal, function: { arguments: event.delta.partial_json } }] }, null);
      }
    } else if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stop = finishReasonOf(event.delta.stop_reason);
      if (isRecord(event.usage)) outputTokens = numberOrZero(event.usage.output_tokens);
    }
  }
  yield chunkFrame(id, model, {}, stop ?? 'stop');
  for (const frame of usageFrames(id, model, inputTokens, outputTokens, inputTokens + outputTokens)) yield frame;
}

// --- Gemini upstream ---

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request (the model rides in
 * the URL, not the body): system messages lift into system_instruction, assistant turns become
 * model turns with functionCall parts, tool results become functionResponse parts (their names
 * recovered from the assistant turn that called them), image data URLs become inline_data, and
 * tools, tool choice and the generation knobs are rewritten in Gemini's spelling.
 */
export function geminiRequestFromOpenAI(body: Json): Json {
  let system = '';
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const toolNames = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'assistant' || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (isRecord(call) && isRecord(call.function) && typeof call.id === 'string' && typeof call.function.name === 'string') {
        toolNames.set(call.id, call.function.name);
      }
    }
  }
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      const name = typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) ?? '' : '';
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: toolResultText(raw.content) } } }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const match = /^data:([^;,]+);base64,(.*)$/.exec(part.image_url.url);
          if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
          else parts.push({ fileData: { fileUri: part.image_url.url } });
        }
      }
    }
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (choice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  const generation: Json = {};
  if (typeof body.max_tokens === 'number') generation.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') generation.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') generation.temperature = body.temperature;
  if (typeof body.top_p === 'number') generation.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) generation.stopSequences = body.stop;
  else if (typeof body.stop === 'string') generation.stopSequences = [body.stop];
  const format = isRecord(body.response_format) ? body.response_format : null;
  if (format !== null && (format.type === 'json_object' || format.type === 'json_schema')) {
    generation.responseMimeType = 'application/json';
    if (format.type === 'json_schema' && isRecord(format.json_schema) && isRecord(format.json_schema.schema)) {
      generation.responseSchema = format.json_schema.schema;
    }
  }
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  return out;
}

/** A Gemini generateContent answer as a Chat Completions object naming `model`. */
export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  let finish = 'stop';
  if (candidate.finishReason === 'MAX_TOKENS') finish = 'length';
  else if (candidate.finishReason === 'SAFETY' || candidate.finishReason === 'RECITATION') finish = 'content_filter';
  else if (toolCalls.length > 0) finish = 'tool_calls';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: finish,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: numberOrZero(usage.totalTokenCount) || input + output },
  };
}

/** A Gemini streamGenerateContent event stream as Chat Completions chunks. */
export async function* openAIChunksFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const id = 'chatcmpl-gemini-stream';
  let inputTokens = 0;
  let outputTokens = 0;
  let totalTokens = 0;
  let stop: string | null = null;
  let roleSent = false;
  let toolCount = 0;
  for await (const payload of ssePayloads(upstream)) {
    if (payload === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (!roleSent) {
      roleSent = true;
      yield chunkFrame(id, model, { role: 'assistant', content: '' }, null);
    }
    if (isRecord(event.usageMetadata)) {
      inputTokens = numberOrZero(event.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(event.usageMetadata.candidatesTokenCount);
      totalTokens = numberOrZero(event.usageMetadata.totalTokenCount);
    }
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string' && part.text !== '') {
        yield chunkFrame(id, model, { content: part.text }, null);
      } else if (isRecord(part.functionCall)) {
        const ordinal = toolCount;
        toolCount += 1;
        yield chunkFrame(
          id,
          model,
          {
            tool_calls: [
              {
                index: ordinal,
                id: `call_${ordinal}`,
                type: 'function',
                function: {
                  name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
                  arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
                },
              },
            ],
          },
          null,
        );
      }
    }
    if (typeof candidate.finishReason === 'string') {
      stop =
        candidate.finishReason === 'MAX_TOKENS'
          ? 'length'
          : candidate.finishReason === 'SAFETY'
            ? 'content_filter'
            : toolCount > 0
              ? 'tool_calls'
              : 'stop';
    }
  }
  yield chunkFrame(id, model, {}, stop ?? 'stop');
  for (const frame of usageFrames(id, model, inputTokens, outputTokens, totalTokens || inputTokens + outputTokens)) yield frame;
}

/** A whole Chat Completions answer rendered as the chunk stream a streamed call would have sent. */
export function* openAIChunkStringsFromJson(body: Json, model: string): Generator<string> {
  const id = typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate';
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  yield chunkFrame(id, model, { role: 'assistant', content: '' }, null);
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') yield chunkFrame(id, model, { content: text }, null);
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  let ordinal = 0;
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    yield chunkFrame(
      id,
      model,
      {
        tool_calls: [
          {
            index: ordinal,
            id: typeof call.id === 'string' ? call.id : `call_${ordinal}`,
            type: 'function',
            function: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
            },
          },
        ],
      },
      null,
    );
    ordinal += 1;
  }
  yield chunkFrame(id, model, {}, typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop');
  const usage = isRecord(body.usage) ? body.usage : {};
  for (const frame of usageFrames(id, model, numberOrZero(usage.prompt_tokens), numberOrZero(usage.completion_tokens), numberOrZero(usage.total_tokens))) {
    yield frame;
  }
}

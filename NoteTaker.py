#!/usr/bin/env python3
"""
Real-time speech-to-text web app. Single file, local, CPU-only.

    pip install "faster-whisper" "fastapi" "uvicorn[standard]" numpy
    python live_transcribe.py                        # http://127.0.0.1:8000
    python live_transcribe.py --model large-v3-turbo # if you have the cores

Weights come from Hugging Face on first run (CTranslate2 int8 Whisper builds).
Browser captures mic audio -> 16 kHz PCM over a websocket -> energy VAD splits it
into utterances -> a small draft model keeps the live text flowing while the big
model re-decodes each finished utterance with the previous lines as context.
"""

import argparse
import asyncio
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model

# ---------------------------------------------------------------- audio config

SR = 16000
FRAME = 512                     # 32 ms, the frame size Silero VAD expects
PREROLL = 16                    # 512 ms kept before speech onset
START_FRAMES = 2                # 64 ms of speech opens an utterance
END_FRAMES = 30                 # ~1.0 s of silence closes it
MAX_SEG = 15 * SR               # hard cap so latency stays bounded
VAD_ON = 0.5                    # Silero probability to open an utterance
VAD_OFF = 0.35                  # ... and to keep it open (hysteresis)
CONTEXT_CHARS = 180             # of previous transcript fed to the decoder
MIN_SEG = int(0.25 * SR)
PARTIAL_EVERY = SR              # re-decode the open utterance once a second

MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small.en": "Systran/faster-whisper-small.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "medium.en": "Systran/faster-whisper-medium.en",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

# Whisper's favourite things to say when it hears nothing much.
HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thanks for watching!",
    "bye", "bye bye", "so", "okay", "oh", "hmm", "mm", "uh", "um",
    "please subscribe", "subtitles by the amara.org community",
}

LANG = "en"
BEAM = 2
HOTWORDS = None
NAMES = ""
# (model, lock, single-thread pool) for the accurate pass and the live draft pass
FINAL = DRAFT = None


def _load(repo, threads):
    m = WhisperModel(repo, device="cpu", compute_type="int8", cpu_threads=threads)
    return [m, threading.Lock(), ThreadPoolExecutor(max_workers=1)]


def transcribe(audio: np.ndarray, final: bool, context: str = "") -> str:
    # Whisper's front end does not normalise level, so a quiet microphone
    # decodes badly even once the VAD has correctly found the speech.
    peak = float(np.abs(audio).max())
    if 0 < peak < 0.7:
        audio = audio * min(0.85 / peak, 20.0)
    model, lock, _ = FINAL if final else DRAFT
    with lock:
        segments, _ = model.transcribe(
            audio,
            language=LANG,
            beam_size=BEAM if final else 1,
            temperature=0.0 if final else 0.0,
            # Context is passed as a prompt, not via condition_on_previous_text,
            # so it stays bounded and cannot spiral over a long session.
            initial_prompt=context or None,
            hotwords=HOTWORDS if final else None,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=final,
            no_speech_threshold=0.6,
        )
        parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.7]
    text = " ".join(p for p in parts if p).strip()
    if len(audio) / SR < 1.6 and text.lower().strip(" .,!?") in HALLUCINATIONS:
        return ""
    return text


class Vad:
    """Silero VAD driven one 32 ms frame at a time, carrying its own LSTM state.

    Unlike an energy gate this is independent of how loud the microphone is, so
    a quiet mic or aggressive browser noise suppression cannot silence it.
    """

    def __init__(self):
        self.s = get_vad_model().session
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame.reshape(1, -1)], 1)
        out, self.h, self.c = self.s.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = frame[-64:].reshape(1, -1).copy()
        return float(out[0])


class Segmenter:
    """Chops the incoming stream into utterances using Silero VAD."""

    def __init__(self):
        self.vad = Vad()
        self.tail = np.zeros(0, np.float32)
        self.pre = deque(maxlen=PREROLL)
        self.seg = None
        self.speech = 0
        self.silence = 0
        self.level = 0.0
        self.last_partial = 0
        self.sid = 0

    def feed(self, x: np.ndarray):
        events = []
        buf = np.concatenate([self.tail, x]) if len(self.tail) else x
        n = len(buf) // FRAME
        for i in range(n):
            events += self._frame(buf[i * FRAME:(i + 1) * FRAME])
        self.tail = buf[n * FRAME:].copy()
        return events

    def _frame(self, f):
        self.level = float(np.sqrt(np.mean(f * f)))
        p = self.vad(f)

        if self.seg is None:
            self.pre.append(f)
            self.speech = self.speech + 1 if p > VAD_ON else 0
            if self.speech >= START_FRAMES:
                self.seg = list(self.pre)
                self.pre.clear()
                self.speech = self.silence = self.last_partial = 0
                self.sid += 1
            return []

        self.seg.append(f)
        self.silence = 0 if p > VAD_OFF else self.silence + 1
        n = len(self.seg) * FRAME
        if self.silence >= END_FRAMES or n >= MAX_SEG:
            return self.flush()
        if n - self.last_partial >= PARTIAL_EVERY:
            self.last_partial = n
            return [("partial", np.concatenate(self.seg), self.sid)]
        return []

    def flush(self):
        if self.seg is None:
            return []
        audio = np.concatenate(self.seg)
        self.seg = None
        self.silence = 0
        if len(audio) < MIN_SEG:
            return []
        return [("final", audio, self.sid)]


# ---------------------------------------------------------------------- server

app = FastAPI()


@app.get("/")
def index():
    return HTMLResponse(PAGE.replace("__MODEL__", NAMES))


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"t": "ready", "model": NAMES}))
    seg = Segmenter()
    loop = asyncio.get_running_loop()
    fq: asyncio.Queue = asyncio.Queue()      # finals, accurate model
    pq: asyncio.Queue = asyncio.Queue()      # partials, draft model
    context = ""                             # tail of the transcript so far
    last = ""

    async def finals():
        nonlocal context, last
        while True:
            _, audio, sid = await fq.get()
            # If the accurate model cannot keep up, fall back to the draft model
            # for the backlog rather than drifting further behind forever.
            big = fq.qsize() < 3
            t0 = time.monotonic()
            text = await loop.run_in_executor(
                (FINAL if big else DRAFT)[2], transcribe, audio, big, context)
            rtf = round((len(audio) / SR) / max(time.monotonic() - t0, 1e-3), 1)
            if text and text == last:        # decoder echoed the prompt
                text = ""
            if text:
                last = text
                context = (context + " " + text)[-CONTEXT_CHARS:]
            await ws.send_text(json.dumps({
                "t": "final" if text else "drop", "id": sid, "text": text,
                "rtf": rtf, "big": big, "dur": round(len(audio) / SR, 1)}))

    async def partials():
        while True:
            _, audio, sid = await pq.get()
            if not pq.empty():
                continue                     # stale, a newer one is waiting
            text = await loop.run_in_executor(
                DRAFT[2], transcribe, audio, False, context)
            if text:
                await ws.send_text(json.dumps({"t": "partial", "id": sid, "text": text}))

    tasks = [asyncio.create_task(finals()), asyncio.create_task(partials())]
    last_level = 0.0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            events = []
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], np.int16).astype(np.float32) / 32768.0
                events = seg.feed(pcm)
                now = time.monotonic()
                if now - last_level > 0.1:
                    last_level = now
                    await ws.send_text(json.dumps({
                        "t": "level",
                        "v": min(1.0, seg.level / 0.12),
                        "on": seg.seg is not None,
                        "lag": fq.qsize(),
                    }))
            elif msg.get("text") and json.loads(msg["text"]).get("t") == "flush":
                events = seg.flush()
            for ev in events:
                if ev[0] == "final":
                    fq.put_nowait(ev)
                elif pq.empty():
                    pq.put_nowait(ev)
    except Exception:
        pass
    finally:
        for t in tasks:
            t.cancel()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Transcript</title>
<style>
:root{--bg:#000;--fg:#fff;--dim:#8a8a8a;--line:#242424}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg)}
body{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-weight:700;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:2px solid var(--fg);
  display:flex;align-items:center;gap:18px;padding:14px 20px}
h1{margin:0;font-size:15px;letter-spacing:.24em;text-transform:uppercase}
.tag{font-size:11px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.grow{flex:1}
#clock{font-size:15px;letter-spacing:.08em}
.bar{position:sticky;top:53px;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:10px;padding:12px 20px;flex-wrap:wrap}
button{border:2px solid var(--fg);background:var(--bg);color:var(--fg);padding:9px 15px;cursor:pointer;
  font:700 12px/1 "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.16em;text-transform:uppercase}
button:hover:not(:disabled){background:var(--fg);color:var(--bg)}
button:disabled{opacity:.28;cursor:not-allowed}
button.live{background:var(--fg);color:var(--bg)}
#meter{display:flex;gap:3px;margin-left:auto}
#meter i{width:5px;height:22px;background:#1e1e1e}
#meter i.on{background:var(--fg)}
main{padding:30px 20px 55vh;max-width:1040px}
.row{display:flex;gap:20px;margin-bottom:20px}
.ts{flex:0 0 58px;padding-top:9px;font-size:13px;color:var(--dim);letter-spacing:.05em}
.tx{font-size:27px;line-height:1.34;word-wrap:break-word}
#partial{display:none}
#partial .tx{color:var(--dim)}
#hint{font-size:16px;color:var(--dim);line-height:1.6;font-weight:400}
.caret{display:inline-block;width:.42em;height:.95em;background:var(--dim);vertical-align:-.1em;
  margin-left:.08em;animation:blink 1.05s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
</style></head><body>
<header>
  <h1>Live Transcript</h1>
  <span class="tag mono" id="model">__MODEL__</span>
  <span class="grow"></span>
  <span class="tag mono" id="rt"></span>
  <span class="tag mono" id="words">0 words</span>
  <span class="mono" id="clock">00:00:00</span>
</header>
<div class="bar">
  <button id="rec">Start</button>
  <button id="clr">Clear</button>
  <button id="cp">Copy</button>
  <button id="dl">Save .txt</button>
  <span class="tag mono" id="status">idle</span>
  <span id="meter"></span>
</div>
<main>
  <p id="hint">Press START and allow microphone access. Speech is transcribed locally on your CPU;
  nothing leaves this machine. A pause of about a second ends a line, which is then re-decoded by the larger model.</p>
  <div id="log"></div>
  <div class="row" id="partial"><div class="ts mono"></div><div class="tx"></div></div>
</main>
<script>
const $ = s => document.querySelector(s);
const meter = $('#meter');
for (let i = 0; i < 22; i++) meter.appendChild(document.createElement('i'));
const bars = [...meter.children];

let ws, ctx, node, stream, gain, running = false, t0 = 0, tick, lines = [], done = 0;

const WORKLET = `class P extends AudioWorkletProcessor{
constructor(){super();this.b=new Float32Array(1024);this.i=0}
process(inp){const c=inp[0][0];if(!c)return true;
for(let k=0;k<c.length;k++){this.b[this.i++]=c[k];
if(this.i===1024){this.port.postMessage(this.b.slice(0));this.i=0}}return true}}
registerProcessor('cap',P)`;

const hms = s => [s/3600|0, s/60%60|0, s%60|0].map(v => String(v).padStart(2,'0')).join(':');

function resample(x, from, to){
  if (from === to) return x;
  const r = to/from, n = Math.round(x.length*r), o = new Float32Array(n);
  for (let i = 0; i < n; i++){
    const p = i/r, j = p|0, t = p-j;
    o[i] = x[j]*(1-t) + (x[Math.min(j+1, x.length-1)])*t;
  }
  return o;
}
function pcm16(f){
  const o = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++){
    const v = Math.max(-1, Math.min(1, f[i]));
    o[i] = v < 0 ? v*0x8000 : v*0x7fff;
  }
  return o.buffer;
}

function autoscroll(){
  if (innerHeight + scrollY >= document.body.scrollHeight - 160)
    scrollTo({top: document.body.scrollHeight});
}
function showPartial(txt){
  const p = $('#partial');
  if (!txt){ p.style.display = 'none'; return; }
  p.style.display = 'flex';
  p.querySelector('.ts').textContent = hms((Date.now()-t0)/1000);
  p.querySelector('.tx').innerHTML = '';
  p.querySelector('.tx').append(txt, Object.assign(document.createElement('span'), {className:'caret'}));
  autoscroll();
}
function addLine(txt){
  const ts = hms((Date.now()-t0)/1000);
  lines.push(ts + '  ' + txt);
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = '<div class="ts mono"></div><div class="tx"></div>';
  row.children[0].textContent = ts;
  row.children[1].textContent = txt;
  $('#log').appendChild(row);
  $('#words').textContent = lines.join(' ').split(/\s+/).filter(w => /\w/.test(w)).length + ' words';
  autoscroll();
}

async function start(){
  $('#hint').style.display = 'none';
  $('#status').textContent = 'connecting';
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.t === 'level'){
      const k = Math.round(m.v * bars.length);
      bars.forEach((b, i) => b.classList.toggle('on', i < k));
      $('#status').textContent = m.lag > 1 ? 'decoding \u00d7' + m.lag
        : m.on ? 'speech' : 'listening';
    }
    else if (m.t === 'partial'){ if (m.id > done) showPartial(m.text); }
    else if (m.t === 'final'){
      done = m.id; showPartial(''); addLine(m.text);
      $('#rt').textContent = m.rtf + '\u00d7 rt' + (m.big ? '' : ' draft');
    }
    else if (m.t === 'drop'){ done = m.id; showPartial(''); }
  };
  ws.onclose = () => { if (running) stop(); $('#status').textContent = 'disconnected'; };
  await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true}
  });
  ctx = new AudioContext({sampleRate: 16000});
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type:'application/javascript'})));
  node = new AudioWorkletNode(ctx, 'cap');
  node.port.onmessage = e => {
    if (ws.readyState === 1) ws.send(pcm16(resample(e.data, ctx.sampleRate, 16000)));
  };
  gain = ctx.createGain(); gain.gain.value = 0;         // keeps the graph pulling
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(gain).connect(ctx.destination);

  running = true;
  if (!t0) t0 = Date.now();
  tick = setInterval(() => $('#clock').textContent = hms((Date.now()-t0)/1000), 500);
  $('#rec').textContent = 'Stop';
  $('#rec').classList.add('live');
  $('#status').textContent = 'listening';
}

function stop(){
  running = false;
  clearInterval(tick);
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({t:'flush'}));
  stream && stream.getTracks().forEach(t => t.stop());
  ctx && ctx.close();
  node = ctx = stream = null;
  bars.forEach(b => b.classList.remove('on'));
  $('#rec').textContent = 'Start';
  $('#rec').classList.remove('live');
  $('#status').textContent = 'stopped';
}

$('#rec').onclick = () => running ? stop() : start().catch(e => {
  $('#status').textContent = 'error: ' + e.message; running = false;
});
$('#clr').onclick = () => { lines = []; done = 0; $('#log').innerHTML = ''; showPartial(''); $('#words').textContent = '0 words'; };
$('#cp').onclick = () => navigator.clipboard.writeText(lines.join('\n'))
  .then(() => { $('#cp').textContent = 'Copied'; setTimeout(() => $('#cp').textContent = 'Copy', 1200); });
$('#dl').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], {type:'text/plain'}));
  a.download = 'transcript-' + new Date().toISOString().slice(0,19).replace(/[:T]/g,'-') + '.txt';
  a.click();
};
addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target === document.body){ e.preventDefault(); $('#rec').click(); }
});
</script></body></html>
"""


def main():
    global FINAL, DRAFT, NAMES, LANG, HOTWORDS, BEAM
    p = argparse.ArgumentParser(description="Local real-time speech-to-text")
    p.add_argument("--model", default="small.en",
                   help="accurate model used for finished lines: shorthand (%s) "
                        "or any CTranslate2 repo id on Hugging Face" % ", ".join(MODELS))
    p.add_argument("--draft", default="base.en",
                   help="fast model used for the live in-progress text, "
                        "or 'same' to reuse --model")
    p.add_argument("--hotwords", default=None,
                   help="names/jargon to bias towards, e.g. 'Craigs, CDQ, Aotearoa'")
    p.add_argument("--beam", type=int, default=2,
                   help="beam size for finished lines; 1 is fastest, 5 is most accurate")
    p.add_argument("--lang", default=None, help="force a language code, e.g. en, de, mi")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    HOTWORDS = a.hotwords
    BEAM = a.beam
    main_repo = MODELS.get(a.model, a.model)
    draft_repo = main_repo if a.draft == "same" else MODELS.get(a.draft, a.draft)
    LANG = a.lang or ("en" if ".en" in main_repo else None)
    NAMES = a.model if draft_repo == main_repo else "%s + %s" % (a.model, a.draft)

    t = time.time()
    print("loading %s (int8, cpu) ..." % main_repo, flush=True)
    FINAL = _load(main_repo, a.threads)
    if draft_repo == main_repo:
        DRAFT = FINAL
    else:
        print("loading draft %s ..." % draft_repo, flush=True)
        DRAFT = _load(draft_repo, a.threads)
    for m, _, _ in {id(FINAL): FINAL, id(DRAFT): DRAFT}.values():
        m.transcribe(np.zeros(SR, np.float32), language=LANG or "en", beam_size=1)
    print("ready in %.1fs  ->  http://%s:%d" % (time.time() - t, a.host, a.port), flush=True)

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
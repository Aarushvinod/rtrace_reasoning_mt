"""
vllm_server.py
──────────────
vLLM OpenAI-compatible server management and the OpenAI-client driver that
walks a translation direction and calls the served model with reasoning
support. Used by `reasoning_main.py` to bring a Qwen3 / Mistral reasoning
model online, translate every direction and k value once, and shut the server
back down.

Inputs:  VLLMServerSpec (model id, port, reasoning parser, GPU utilisation
         cap, etc.), pool of retrieval selections + FLORES source sentences.
Outputs: JSONL files with per-sentence `translation` and `reasoning_trace`
         fields, plus a subprocess log per server run.
"""

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.retrieval_helpers import (
    Template,
    _ensure_dir,
    generate_prompt_template11,
    postprocess_generation,
)


def _load_existing_jsonl_field(path: str, field: str, n: int) -> List[Optional[str]]:
    """
    Purpose: Read up to n entries from an existing JSONL file and return the
             value of `field` for each line, or None if the file is missing,
             a line is malformed, or the field is absent.
    Inputs:  path  — path to the JSONL file.
             field — key to extract from each JSON object.
             n     — expected number of entries (list is always length n).
    Outputs: list of length n with the extracted values (or None).
    """
    results: List[Optional[str]] = [None] * n
    if not os.path.exists(path):
        return results
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    results[i] = obj.get(field)
                except Exception:
                    pass
    except Exception:
        pass
    return results


def _count_jsonl_lines(path: str) -> int:
    """
    Purpose: Count the lines currently present in a JSONL file.
    Inputs:  path — file to count.
    Outputs: line count (0 if the file is missing or unreadable).
    """
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _prepare_resume(
    out_jsonl_path: str,
    out_reasoning_jsonl_path: str,
    n: int,
) -> Tuple[bool, List[Optional[str]], List[Optional[str]]]:
    """
    Purpose: Build the resume state for a direction, recovering banked work
             from both the published files and any .tmp left by an attempt
             that was preempted mid-walk.
    Inputs:  final translation/reasoning paths and expected sentence count n.
    Outputs: (complete, translations, reasoning). complete is True only when
             the PUBLISHED translation file already holds n non-empty entries,
             so the caller can skip the direction without rewriting anything.
             The lists merge published and .tmp values pairwise, preferring
             the .tmp pair wherever its translation is non-empty (that is
             newer regenerated work not yet published).
    """
    main_t = _load_existing_jsonl_field(out_jsonl_path, "translation", n)
    main_r = _load_existing_jsonl_field(out_reasoning_jsonl_path, "reasoning", n)
    complete = (
        all(t for t in main_t)
        and _count_jsonl_lines(out_jsonl_path) >= n
        and _count_jsonl_lines(out_reasoning_jsonl_path) >= n
    )
    if complete:
        return True, main_t, main_r
    tmp_t = _load_existing_jsonl_field(out_jsonl_path + ".tmp", "translation", n)
    tmp_r = _load_existing_jsonl_field(out_reasoning_jsonl_path + ".tmp", "reasoning", n)
    merged_t = [tmp_t[i] if tmp_t[i] else main_t[i] for i in range(n)]
    merged_r = [tmp_r[i] if tmp_t[i] else main_r[i] for i in range(n)]
    return False, merged_t, merged_r


@dataclass
class VLLMServerSpec:
    """
    Purpose: Configuration for launching a vLLM OpenAI-compatible server.
    Inputs: model_key/model_id/port and optional vLLM flags.
    Outputs: spec object.
    """
    model_key: str
    model_id: str
    port: int
    reasoning_parser: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    trust_remote_code: bool = True
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    extra_args: Optional[List[str]] = None
    env_overrides: Optional[Dict[str, str]] = None


class VLLMServerProcess:
    """
    Purpose: Manage lifecycle of a vLLM server subprocess.
    Inputs: VLLMServerSpec and server host/log dir and timeouts.
    Outputs: context manager for running server.
    """
    def __init__(
        self,
        spec: VLLMServerSpec,
        *,
        host: str,
        log_dir: str,
        start_timeout_s: int,
        poll_interval_s: float,
        openai_api_key: str,
    ):
        self.spec = spec
        self.host = host
        self.log_dir = log_dir
        self.start_timeout_s = int(start_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.openai_api_key = openai_api_key
        self.proc: Optional[subprocess.Popen] = None
        self._log_f = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.spec.port}/v1"

    def _build_command(self) -> List[str]:
        """
        Purpose: Build the `vllm serve` command.
        Inputs: None.
        Outputs: argv list.
        """
        cmd = ["vllm", "serve", self.spec.model_id, "--host", self.host, "--port", str(self.spec.port), "--enable-prefix-caching"]
        if 'mistral' in self.spec.model_id.lower():
            cmd += ['--tokenizer_mode', 'mistral', '--config_format', 'mistral', '--load_format', 'mistral']
        if self.spec.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.spec.tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(int(self.spec.tensor_parallel_size))]
        if self.spec.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(float(self.spec.gpu_memory_utilization))]
        if self.spec.max_model_len is not None:
            cmd += ["--max-model-len", str(int(self.spec.max_model_len))]
        if self.spec.reasoning_parser:
            cmd += ["--reasoning-parser", self.spec.reasoning_parser]
        if self.spec.extra_args:
            cmd += list(self.spec.extra_args)
        return cmd

    def start(self) -> None:
        """
        Purpose: Start vLLM server and wait until ready.
        Inputs: None.
        Outputs: None.
        """
        from openai import OpenAI

        _ensure_dir(self.log_dir)
        log_path = os.path.join(self.log_dir, f"{self.spec.model_key}_port{self.spec.port}.log")

        env = os.environ.copy()
        # The compute nodes have no CUDA toolkit (no nvcc), and vLLM's default
        # FlashInfer top-k/top-p sampler JIT-compiles a kernel at engine
        # startup — which crashes every server with "Could not find nvcc".
        # Force the PyTorch-native sampler; explicit ambient overrides win.
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        if self.spec.env_overrides:
            env.update({k: str(v) for k, v in self.spec.env_overrides.items()})

        cmd = self._build_command()
        self._log_f = open(log_path, "w", encoding="utf-8")

        self.proc = subprocess.Popen(cmd, stdout=self._log_f, stderr=self._log_f, env=env)

        client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        t0 = time.time()
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server '{self.spec.model_key}' exited early with code {self.proc.returncode}.")
            try:
                _ = client.models.list()
                return
            except Exception:
                pass
            if time.time() - t0 > self.start_timeout_s:
                raise TimeoutError(f"Timed out waiting for vLLM server '{self.spec.model_key}' at {self.base_url}.")
            time.sleep(self.poll_interval_s)

    def stop(self) -> None:
        """
        Purpose: Stop vLLM server subprocess.
        Inputs: None.
        Outputs: None.
        """
        if self.proc is None:
            if self._log_f is not None:
                try:
                    self._log_f.close()
                except Exception:
                    pass
                self._log_f = None
            return

        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=15)
                except Exception:
                    self.proc.kill()

        self.proc = None
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None

    def __enter__(self) -> "VLLMServerProcess":
        """
        Purpose: Start server in context manager.
        Inputs: None.
        Outputs: self.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Purpose: Stop server in context manager.
        Inputs: exception info.
        Outputs: None.
        """
        self.stop()


def get_served_model_id(client) -> str:
    """
    Purpose: Fetch served model id from /v1/models.
    Inputs: OpenAI client.
    Outputs: model id string.
    """
    models = client.models.list()
    if not models.data:
        raise RuntimeError("vLLM server returned no models from /v1/models.")
    return models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


_THINK_OPEN_MARKERS = ("<think>", "[THINK]")
_THINK_CLOSE_MARKERS = ("</think>", "[/THINK]")


def _split_think_block(text: str) -> Tuple[Optional[str], str]:
    """
    Purpose: Split raw model text that carries literal think markers into
             (reasoning, answer) — the answer is everything after the LAST
             close marker; the reasoning is everything before it with the
             markers removed.
    Inputs:  raw text.
    Outputs: (reasoning, answer). reasoning is None when the text has no think
             markers at all (the text is then entirely answer); with an open
             marker but no close marker the whole text is unterminated
             reasoning and the answer is "".
    """
    close_idx = -1
    close_len = 0
    for mk in _THINK_CLOSE_MARKERS:
        i = text.rfind(mk)
        if i > close_idx:
            close_idx = i
            close_len = len(mk)
    if close_idx >= 0:
        reasoning = text[:close_idx]
        for mk in _THINK_OPEN_MARKERS + _THINK_CLOSE_MARKERS:
            reasoning = reasoning.replace(mk, "")
        return reasoning.strip(), text[close_idx + close_len:].strip()
    if any(mk in text for mk in _THINK_OPEN_MARKERS):
        cleaned = text
        for mk in _THINK_OPEN_MARKERS:
            cleaned = cleaned.replace(mk, "")
        return cleaned.strip(), ""
    return None, text


def _extract_translation_and_reasoning(
    content_val: str, reasoning_val: str, template
) -> Tuple[str, str]:
    """
    Purpose: Turn one attempt's raw (content, reasoning) channel pair into a
             (translation, reasoning) pair, robust to every observed routing of
             think blocks: clean content (parser engaged), literal think
             markers left in content (parser not engaged — split and translate
             from the post-think answer instead of discarding the attempt),
             and the whole output misrouted onto the reasoning channel
             (recover the post-think answer from there). The think split runs
             on RAW text, BEFORE postprocess_generation — postprocess cuts at
             the first '###', which reasoning about ###-separated demos
             routinely contains.
    Inputs:  raw content channel text, raw reasoning channel text, template.
    Outputs: (translation, reasoning) — translation is "" only when no usable
             post-think answer exists anywhere (caller then retries).
    """
    raw = (content_val or "").strip()
    reasoning_val = (reasoning_val or "").strip()
    think_text, answer = _split_think_block(raw)
    if think_text is not None:
        reasoning_val = (reasoning_val + "\n" + think_text).strip() if reasoning_val else think_text
        translation = postprocess_generation(answer, template) if answer else ""
    else:
        translation = postprocess_generation(raw, template)
    if translation == "" and any(mk in reasoning_val for mk in _THINK_CLOSE_MARKERS):
        r_think, r_answer = _split_think_block(reasoning_val)
        if r_answer:
            translation = postprocess_generation(r_answer, template)
            if r_think is not None:
                reasoning_val = r_think
    return translation, reasoning_val


def _attempt_translation(
    client,
    *,
    served_model_id: str,
    messages: List[Dict[str, Any]],
    template,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: List[str],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Purpose: Make a single non-streaming translation attempt and return the
             postprocessed translation together with its reasoning trace,
             via _extract_translation_and_reasoning (think-marker robust).
    Inputs:  OpenAI client, served model id, fully-built messages list,
             postprocessing template, and decoding params.
    Outputs: tuple of (translation, reasoning_val) — both are strings,
             translation may be "" if the model produced no usable output.
    """
    resp = client.chat.completions.create(
        model=served_model_id,
        messages=messages,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        stop=stop_sequences,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    reasoning_val = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    content_val = getattr(msg, "content", None) or ""
    return _extract_translation_and_reasoning(content_val, reasoning_val, template)


def _attempt_translation_streaming(
    client,
    *,
    served_model_id: str,
    messages: List[Dict[str, Any]],
    template,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: List[str],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Purpose: Make a single streaming translation attempt and return the
             postprocessed translation together with its reasoning trace,
             via _extract_translation_and_reasoning (think-marker robust).
    Inputs:  OpenAI client, served model id, fully-built messages list,
             postprocessing template, and decoding params.
    Outputs: tuple of (translation, reasoning_val) — both are strings,
             translation may be "" if the model produced no usable output.
    """
    reasoning_val, content_val = stream_reasoning_and_content_for_messages(
        client,
        served_model_id=served_model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        extra_body=extra_body,
    )

    return _extract_translation_and_reasoning(content_val, reasoning_val, template)


def run_openai_for_direction(
    client,
    *,
    served_model_id: str,
    model_key: str,
    src_lang_code: str,
    tgt_lang_code: str,
    k: int,
    src_dev: List[str],
    src_devtest: List[str],
    tgt_dev: List[str],
    dev_indices_per_devtest: List[List[int]],
    out_jsonl_path: str,
    out_reasoning_jsonl_path: str,
    request_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: List[str],
    extra_body: Optional[Dict[str, Any]],
    log_reasoning_traces: bool,
    prompt_header: str,
    src_name: str,
    tgt_name: str,
    system_prompt: Optional[Dict[str, Any]] = None,
    skip_existing_translations: bool = False,
    max_retries: int = 0,
) -> None:
    """
    Purpose: Generate translations via vLLM OpenAI-compatible server and optionally log reasoning separately.
             When skip_existing_translations is True, sentences that already have a non-empty translation
             in out_jsonl_path are skipped; only empty/None entries are (re-)translated, and their
             reasoning traces are always overwritten with the newly produced reasoning.
             When max_retries > 0, any attempt that produces an empty translation is retried up to
             max_retries additional times; the first non-empty result is kept immediately.
             Outputs are rewritten via sibling .tmp files and published with an atomic os.replace
             only after the full walk, so a mid-walk kill never loses previously banked sentences;
             fully-complete directions are skipped without touching the published files.
    Inputs: OpenAI client, model id, data lists, indices, decoding params, output paths, prompt config,
            skip_existing_translations flag, max_retries count.
    Outputs: None.
    """
    _ensure_dir(os.path.dirname(out_jsonl_path))
    _ensure_dir(os.path.dirname(out_reasoning_jsonl_path))

    n_sentences = len(src_devtest)

    if skip_existing_translations:
        complete, existing_translations, existing_reasoning = _prepare_resume(
            out_jsonl_path, out_reasoning_jsonl_path, n_sentences
        )
        if complete:
            # Nothing to redo: leave the published files untouched (no
            # truncate/rewrite window at all) and refresh their timestamps so
            # scratch-purge age sweeps never see them as stale.
            now = time.time()
            for p in (out_jsonl_path, out_reasoning_jsonl_path):
                try:
                    os.utime(p, (now, now))
                except OSError:
                    pass
            print(f"[skip-file] {out_jsonl_path} complete ({n_sentences}/{n_sentences}); left untouched.")
            return
    else:
        existing_translations = [None] * n_sentences
        existing_reasoning = [None] * n_sentences

    prompts: List[str] = []
    templates_for_each: List[Template] = []

    for i, src in enumerate(src_devtest):
        idxs = dev_indices_per_devtest[i]
        demos = [(src_dev[j], tgt_dev[j]) for j in idxs] if idxs else []
        prompt, template = generate_prompt_template11(
            src_sentence=src,
            demonstrations=demos,
            src_name=src_name,
            tgt_name=tgt_name,
            header=prompt_header,
        )
        prompts.append(prompt)
        templates_for_each.append(template)

    # All (re)writes go to sibling .tmp files; the previously published files
    # stay intact on disk until every sentence is written, then an atomic
    # os.replace publishes the new versions. A preemption/kill at any point
    # therefore never loses banked sentences — at worst the in-flight one,
    # and the partial .tmp is itself recovered by _prepare_resume next time.
    tmp_jsonl_path = out_jsonl_path + ".tmp"
    tmp_reasoning_jsonl_path = out_reasoning_jsonl_path + ".tmp"
    with open(tmp_jsonl_path, "w", encoding="utf-8") as f_out, \
         open(tmp_reasoning_jsonl_path, "w", encoding="utf-8") as f_r:

        for start in range(0, len(prompts), int(request_batch_size)):
            batch_prompts = prompts[start : start + int(request_batch_size)]

            for local_j, prompt in enumerate(batch_prompts):
                global_idx = start + local_j

                # Reuse the existing translation if it is non-empty and skipping is enabled.
                if skip_existing_translations:
                    stored = existing_translations[global_idx]
                    if stored is not None and stored != "":
                        print(f"[skip] sentence {global_idx} already translated.")
                        f_out.write(json.dumps({"translation": stored}, ensure_ascii=False) + "\n")
                        f_r.write(json.dumps({"reasoning": existing_reasoning[global_idx]}, ensure_ascii=False) + "\n")
                        # Flush per line so a killed attempt's .tmp always
                        # carries everything walked so far.
                        f_out.flush()
                        f_r.flush()
                        continue

                if local_j == 0:
                    print(prompt)

                messages: List[Dict[str, Any]] = []
                if system_prompt:
                    messages.append(system_prompt)
                messages.append({"role": "user", "content": prompt})

                template = templates_for_each[global_idx]

                # Initial attempt plus up to max_retries retries if the
                # translation comes back empty.
                translation, reasoning_val = _attempt_translation(
                    client,
                    served_model_id=served_model_id,
                    messages=messages,
                    template=template,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stop_sequences,
                    extra_body=extra_body,
                )

                for retry_num in range(1, int(max_retries) + 1):
                    if translation != "":
                        break
                    print(f"[retry {retry_num}/{max_retries}] sentence {global_idx} returned empty translation, retrying...")
                    translation, reasoning_val = _attempt_translation(
                        client,
                        served_model_id=served_model_id,
                        messages=messages,
                        template=template,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        stop_sequences=stop_sequences,
                        extra_body=extra_body,
                    )

                if translation == "":
                    print(f"[warning] sentence {global_idx} still empty after {max_retries} retries.")

                print(translation)
                f_out.write(json.dumps({"translation": translation}, ensure_ascii=False) + "\n")

                if log_reasoning_traces:
                    f_r.write(json.dumps({"reasoning": reasoning_val}, ensure_ascii=False) + "\n")
                else:
                    f_r.write(json.dumps({"reasoning": None}, ensure_ascii=False) + "\n")

                # Flush after every completed sentence so the .tmp survives a
                # SIGKILL with all regenerated work up to the mid-flight one.
                f_out.flush()
                f_r.flush()

    # Walk finished every sentence: publish atomically over the old files.
    os.replace(tmp_jsonl_path, out_jsonl_path)
    os.replace(tmp_reasoning_jsonl_path, out_reasoning_jsonl_path)


def stream_reasoning_and_content_for_messages(
    client,
    *,
    served_model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: List[str],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Purpose: Stream one chat completion and collect reasoning/content separately.
    Inputs: OpenAI client, served model id, chat messages, decoding params, and extra body.
    Outputs: tuple of (reasoning_text, content_text).
    """
    stream = client.chat.completions.create(
        model=served_model_id,
        messages=messages,
        stream=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        stop=stop_sequences,
        extra_body=extra_body,
    )

    reasoning_parts: List[str] = []
    content_parts: List[str] = []

    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        if not chunk.choices:
            continue

        delta = getattr(chunk.choices[0], "delta", None)
        if delta is None:
            continue

        reasoning_piece = getattr(delta, "reasoning_content", None)
        if reasoning_piece is None:
            reasoning_piece = getattr(delta, "reasoning", None)

        content_piece = getattr(delta, "content", None)

        if reasoning_piece:
            reasoning_parts.append(reasoning_piece)
        if content_piece:
            content_parts.append(content_piece)

    return "".join(reasoning_parts), "".join(content_parts)


def run_openai_for_direction_streaming(
    client,
    *,
    served_model_id: str,
    model_key: str,
    src_lang_code: str,
    tgt_lang_code: str,
    k: int,
    src_dev: List[str],
    src_devtest: List[str],
    tgt_dev: List[str],
    dev_indices_per_devtest: List[List[int]],
    out_jsonl_path: str,
    out_reasoning_jsonl_path: str,
    request_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: List[str],
    extra_body: Optional[Dict[str, Any]],
    log_reasoning_traces: bool,
    prompt_header: str,
    src_name: str,
    tgt_name: str,
    system_prompt: Optional[Dict[str, Any]] = None,
    skip_existing_translations: bool = False,
    max_retries: int = 0,
    use_streaming: bool = True,
) -> None:
    """
    Purpose: Generate translations with streaming and store reasoning/content separately.
             use_streaming=False sends plain (non-streaming) requests instead — the Mistral
             reasoning parser's streaming channel split is unreliable, so the Mistral family
             runs non-streaming; the client needs a timeout sized for full-budget generations.
             When skip_existing_translations is True, sentences that already have a non-empty
             translation in out_jsonl_path are skipped; only empty/None entries are (re-)translated,
             and their reasoning traces are always overwritten with the newly produced reasoning.
             When max_retries > 0, any attempt that produces an empty translation is retried up to
             max_retries additional times; the first non-empty result is kept immediately.
             Outputs are rewritten via sibling .tmp files and published with an atomic os.replace
             only after the full walk, so a mid-walk kill never loses previously banked sentences;
             fully-complete directions are skipped without touching the published files.
    Inputs: OpenAI client, model id, data lists, indices, decoding params, output paths, prompt config,
            skip_existing_translations flag, max_retries count.
    Outputs: None.
    """
    _ensure_dir(os.path.dirname(out_jsonl_path))
    _ensure_dir(os.path.dirname(out_reasoning_jsonl_path))

    n_sentences = len(src_devtest)

    if skip_existing_translations:
        complete, existing_translations, existing_reasoning = _prepare_resume(
            out_jsonl_path, out_reasoning_jsonl_path, n_sentences
        )
        if complete:
            # Nothing to redo: leave the published files untouched (no
            # truncate/rewrite window at all) and refresh their timestamps so
            # scratch-purge age sweeps never see them as stale.
            now = time.time()
            for p in (out_jsonl_path, out_reasoning_jsonl_path):
                try:
                    os.utime(p, (now, now))
                except OSError:
                    pass
            print(f"[skip-file] {out_jsonl_path} complete ({n_sentences}/{n_sentences}); left untouched.")
            return
    else:
        existing_translations = [None] * n_sentences
        existing_reasoning = [None] * n_sentences

    prompts: List[str] = []
    templates_for_each: List[Template] = []

    for i, src in enumerate(src_devtest):
        idxs = dev_indices_per_devtest[i]
        demos = [(src_dev[j], tgt_dev[j]) for j in idxs] if idxs else []
        prompt, template = generate_prompt_template11(
            src_sentence=src,
            demonstrations=demos,
            src_name=src_name,
            tgt_name=tgt_name,
            header=prompt_header,
        )
        prompts.append(prompt)
        templates_for_each.append(template)

    # All (re)writes go to sibling .tmp files; the previously published files
    # stay intact on disk until every sentence is written, then an atomic
    # os.replace publishes the new versions. A preemption/kill at any point
    # therefore never loses banked sentences — at worst the in-flight one,
    # and the partial .tmp is itself recovered by _prepare_resume next time.
    tmp_jsonl_path = out_jsonl_path + ".tmp"
    tmp_reasoning_jsonl_path = out_reasoning_jsonl_path + ".tmp"
    with open(tmp_jsonl_path, "w", encoding="utf-8") as f_out, \
         open(tmp_reasoning_jsonl_path, "w", encoding="utf-8") as f_r:

        for start in range(0, len(prompts), int(request_batch_size)):
            batch_prompts = prompts[start : start + int(request_batch_size)]

            for local_j, prompt in enumerate(batch_prompts):
                global_idx = start + local_j

                # Reuse the existing translation if it is non-empty and skipping is enabled.
                if skip_existing_translations:
                    stored = existing_translations[global_idx]
                    if stored is not None and stored != "":
                        print(f"[skip] sentence {global_idx} already translated.")
                        f_out.write(json.dumps({"translation": stored}, ensure_ascii=False) + "\n")
                        f_r.write(json.dumps({"reasoning": existing_reasoning[global_idx]}, ensure_ascii=False) + "\n")
                        # Flush per line so a killed attempt's .tmp always
                        # carries everything walked so far.
                        f_out.flush()
                        f_r.flush()
                        continue

                if local_j == 0:
                    print(prompt)

                messages: List[Dict[str, Any]] = []
                if system_prompt:
                    messages.append(system_prompt)
                messages.append({"role": "user", "content": prompt})

                template = templates_for_each[global_idx]

                # Initial attempt plus up to max_retries retries if the
                # translation comes back empty.
                attempt_fn = _attempt_translation_streaming if use_streaming else _attempt_translation
                translation, reasoning_val = attempt_fn(
                    client,
                    served_model_id=served_model_id,
                    messages=messages,
                    template=template,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stop_sequences,
                    extra_body=extra_body,
                )

                for retry_num in range(1, int(max_retries) + 1):
                    if translation != "":
                        break
                    print(f"[retry {retry_num}/{max_retries}] sentence {global_idx} returned empty translation, retrying...")
                    translation, reasoning_val = attempt_fn(
                        client,
                        served_model_id=served_model_id,
                        messages=messages,
                        template=template,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        stop_sequences=stop_sequences,
                        extra_body=extra_body,
                    )

                if translation == "":
                    print(f"[warning] sentence {global_idx} still empty after {max_retries} retries.")

                print(f"Reasoning Content for sentence {global_idx}: {reasoning_val is not None and reasoning_val != ''}")
                print(translation)
                f_out.write(json.dumps({"translation": translation}, ensure_ascii=False) + "\n")

                if log_reasoning_traces:
                    f_r.write(json.dumps({"reasoning": reasoning_val}, ensure_ascii=False) + "\n")
                else:
                    f_r.write(json.dumps({"reasoning": None}, ensure_ascii=False) + "\n")

                # Flush after every completed sentence so the .tmp survives a
                # SIGKILL with all regenerated work up to the mid-flight one.
                f_out.flush()
                f_r.flush()

    # Walk finished every sentence: publish atomically over the old files.
    os.replace(tmp_jsonl_path, out_jsonl_path)
    os.replace(tmp_reasoning_jsonl_path, out_reasoning_jsonl_path)
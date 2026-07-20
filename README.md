# omp-samplers

Per-prompt sampler control and model-selected sampling for [oh-my-pi](https://github.com/can1357/oh-my-pi) with a local llama.cpp server.

You can still build a sampler chain by hand, tune every knob, and save presets. Auto mode adds a sampling router: before omp asks the model to answer, the same model makes an internal schema-constrained tool call that chooses, orders, and configures the samplers for that prompt. The validated choice is applied to the answer from its first token onward.

![omp-samplers demo](assets/demo.gif)

## How it works

Manual mode subscribes to omp's `before_provider_request` event and writes the active `samplers` array and knob fields into the outgoing request body. llama.cpp rebuilds its sampler stack for that request.

Auto mode adds a routing prelude to the normal agent loop:

1. `before_agent_start` quickly activates the internal `route_samplers` tool and initializes routing state; it never waits for a model call.
2. The first provider request contains only the static sampler guidance, an extension-authored envelope holding the current task as inert JSON data, and that one forced tool. The raw task is never presented as an instruction to answer during routing; the normal agent context remains intact for the answer.
3. The compact tool schema contains every sampler ID and positional values. Its array order is execution order; a compact catalog maps those values to each sampler's knobs, and the local validator enforces types, bounds, and ownership. An optional one-sentence rationale can be removed from the schema entirely.
4. The tool validates and stores the route, then omp's normal tool follow-up becomes the answer request. Every token of that answer—and every later tool follow-up in the run—uses the selected route.

If the model somehow returns an invalid schema payload, generation continues with the saved manual profile as a fallback.

### Why routing is a separate request

llama.cpp fixes the sampler stack when an HTTP generation request starts. It cannot generate configuration tokens and then replace its own sampler stack halfway through that same request through the OpenAI-compatible API. Doing that literally would require a new stateful protocol and server-side llama.cpp changes.

The two-request design preserves the intended behavior: the structured route is generated first, then every answer token uses it. Both requests now run inside omp's normal agent loop, so routing is controlled by the agent's own cancellation lifecycle and cannot hit omp's 30-second extension-handler deadline.

## Requirements

Use oh-my-pi with a local llama.cpp provider configured as `api: openai-completions`. Auto routing is tested with omp 17.0.5.

The standard names `dry`, `top_k`, `top_p`, `min_p`, `typ_p`, `top_n_sigma`, `xtc`, `penalties`, and `temperature` work on recent stock llama.cpp builds. `hill`, `top_h`, `p_less`, `min_k`, `geo_mean`, `otsu`, `kneedle`, `top_gap`, and `robust_z` require a build that implements them, such as mink.

## Install

Clone the repository into the omp user extensions directory, then restart omp:

```sh
git clone https://github.com/Hellisotherpeople/omp-samplers.git ~/.omp/agent/extensions/omp-samplers
```

You can also load it for one run with `omp --extension /path/to/omp-samplers` or add it to the `extensions` list in `~/.omp/agent/config.yml`.

## Usage

Open `/samplers` for the interactive menu. The footer reports whether the override is off, manual, or auto and displays the latest selected route. `/samplers show` also reports the optional rationale plus routing latency and token/cache counters; every selection is written to omp's debug log.

Useful direct commands:

```text
/samplers auto          enable model-selected routing
/samplers manual        use the saved manual profile
/samplers show          show mode, route rationale, fallback, and storage path
/samplers rationale off omit rationale generation for lower latency
/samplers-auto          toggle auto/manual
/samplers-auto on|off   set auto/manual explicitly
/samplers-rationale     toggle rationale generation
/samplers-rationale on|off
/sampler-preset coding  apply a preset and switch to manual mode
/temp 0.7               set manual temperature and switch to manual mode
/samplers-off           toggle all request overrides
```

Editing the chain, tuning a knob, applying a preset, or using `/temp` switches to manual mode. Auto mode itself persists in `sampler-profile.json`, so it survives restarts.

For a non-interactive run, `OMP_SAMPLERS_AUTO=1` forces auto mode without changing the stored profile. The extension imposes no independent routing deadline; routing uses omp's normal agent request lifecycle.

## Router behavior

The router is deliberately conservative about structured output:

- It must choose between one and eight unique samplers.
- Sampler order and configuration are emitted together, preventing knobs from being attached to the wrong sampler.
- Numeric choices are schema-bounded; manual mode remains available for experiments outside those bounds.
- The router request itself uses greedy `top_k=1 → temperature=0` sampling and has thinking disabled. A route cannot apply to the tokens that generate that same route.
- User text is treated as classification input, not as authority to disable the schema or skip routing.

Auto mode adds one model inference and its latency to each top-level prompt. The internal tool turn reports the selected route, while the footer and `/samplers show` retain it afterward. Turning rationale off removes that field from both the provider schema and generated output; it saves decoding time, though the sampler configuration itself still has to be generated.

### Prompt caching and overhead

The routing request keeps a small, invariant instruction prefix and only the current user turn, while the following answer request restores the normal agent context. Both use omp's normal conversation cache key, and routing explicitly sends llama.cpp's `cache_prompt: true`, allowing recent llama.cpp/mink servers to retain both alternating prefixes in slot or host memory.

The routing request exposes only one compact tool, and its schema avoids a separate object grammar for every sampler; one compact catalog documents positional knob order. Thinking is disabled, routing uses a conservative fixed distribution, output is capped at 512 tokens with rationale or 384 without it, and one route is reused for every tool follow-up in the agent run. The unavoidable remaining cost is the second model decode: schema-constrained selection cannot be free while the LLM itself remains the router.

## Presets

Bundled manual presets include three high-temperature creative stacks, a balanced everyday stack, a low-entropy coding stack for reliable tool JSON, and greedy generation. Applying one replaces the manual chain and updates its relevant knobs.

Very high temperature makes coding agents and tool calls unreliable unless paired with a restrictive adaptive sampler. Prefer auto mode, `coding`, or `balanced` when omp needs to use tools.

## Development

Run the schema and validation tests with:

```sh
bun test
```

## License

MIT. See [LICENSE](LICENSE).

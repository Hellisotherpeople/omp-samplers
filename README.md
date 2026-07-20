# omp-samplers

Per-prompt sampler control and model-selected sampling for [oh-my-pi](https://github.com/can1357/oh-my-pi) with a local llama.cpp server.

You can still build a sampler chain by hand, tune every knob, and save presets. Auto mode adds a sampling router: before omp asks the model to answer, the same model makes a hidden schema-constrained tool call that chooses, orders, and configures the samplers for that prompt. The validated choice is applied to the answer from its first token onward.

![omp-samplers demo](assets/demo.gif)

## How it works

Manual mode subscribes to omp's `before_provider_request` event and writes the active `samplers` array and knob fields into the outgoing request body. llama.cpp rebuilds its sampler stack for that request.

Auto mode adds an awaited `before_agent_start` step:

1. The active model receives only the current task, sampler guidance, and a forced `route_samplers` tool.
2. The compact tool schema contains every sampler ID and positional values. Its array order is execution order; a compact catalog maps those values to each sampler's knobs, and the local validator enforces types, bounds, and ownership. An optional one-sentence rationale can be removed from the schema entirely.
3. The extension validates the result again locally: unsupported or repeated samplers, unknown settings, wrong types, and out-of-range values are rejected.
4. Every model request in the resulting agent run receives that route. Tool-call follow-ups therefore keep the same settings.

If routing times out or returns an invalid payload, generation continues with the saved manual profile as a fallback.

### Why routing is a separate request

llama.cpp fixes the sampler stack when an HTTP generation request starts. It cannot generate configuration tokens and then replace its own sampler stack halfway through that same request through the OpenAI-compatible API. Doing that literally would require a new stateful protocol and server-side llama.cpp changes.

The two-request design preserves the intended behavior: the structured route is generated first, then every answer token uses it. It also keeps routing tokens out of the visible conversation.

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

For a non-interactive run, `OMP_SAMPLERS_AUTO=1` forces auto mode without changing the stored profile. Routing has a 25-second timeout by default; set `OMP_SAMPLERS_ROUTER_TIMEOUT_MS` to a value from 1000 through 28000 to change it.

## Router behavior

The router is deliberately conservative about structured output:

- It must choose between one and eight unique samplers.
- Sampler order and configuration are emitted together, preventing knobs from being attached to the wrong sampler.
- Numeric choices are schema-bounded; manual mode remains available for experiments outside those bounds.
- The router request itself uses fixed `top_k → top_p → temperature` sampling at temperature 0.1 and has thinking disabled. A route cannot apply to the tokens that generate that same route.
- User text is treated as classification input, not as authority to disable the schema or skip routing.

Auto mode adds one model inference and its latency to each top-level prompt. The route call is hidden and is not appended to the chat transcript. Turning rationale off removes that field from both the schema and generated output; it saves decoding time, though the sampler configuration itself still has to be generated.

### Prompt caching and overhead

The invariant router instructions and tool schema precede the per-prompt task, providing a stable cacheable prefix. The request sets omp's long-retention/stable cache hints and explicitly sends llama.cpp's `cache_prompt: true`. Recent llama.cpp/mink servers can therefore restore the router prefix from their slot or host-memory prompt cache even though answer requests run between router requests.

The schema avoids a separate object grammar for every sampler; one compact catalog documents positional knob order. Thinking is disabled, routing uses a conservative fixed distribution, input tasks are capped at 12,000 characters, output is capped at 512 tokens with rationale or 384 without it, and one route is reused for every tool follow-up in the agent run. The unavoidable remaining cost is the second model decode: schema-constrained selection cannot be free while the LLM itself remains the router.

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

# omp-samplers

Per-prompt control over the llama.cpp sampler chain for [oh-my-pi](https://github.com/can1357/oh-my-pi), driven by dropdown menus inside the omp terminal UI. You pick which samplers run, in what order, and you tune each one's parameters, without memorizing any flags or request-body field names. Your choice is injected into every outgoing request, so the local server reconfigures its sampler stack on the fly.

![omp-samplers demo](assets/demo.gif)

## Why this exists

oh-my-pi forwards a fixed set of sampling parameters to a local llama.cpp server. Changing the sampler stack normally means editing `models.yml` and restarting, or rebuilding your launch script. This extension moves that control into the running session. You open a menu, choose a chain such as dry then hill then xtc then temperature, tune the knobs, and your next message uses it.

## How it works

oh-my-pi is built on pi, which exposes an extension API with lifecycle events. This extension subscribes to `before_provider_request`, the event that fires just before a request goes out to the model. It reads your active profile and writes a `samplers` array plus the matching knob fields directly into the request body. llama.cpp honors a per-request `samplers` list and per-sampler parameters on its OpenAI-compatible endpoint, so the server rebuilds its chain for that single request.

The write happens in place and synchronously, which matters because the OpenAI-completions provider inside omp reads the body the moment the hook returns. The practical result is that whatever you select in the menu takes effect on your very next prompt and stays in effect until you change it.

## Requirements

You need oh-my-pi installed and pointed at a local llama.cpp server through a provider with `api: openai-completions`. The custom sampler names such as hill, top_h, p_less, min_k, geo_mean, otsu, kneedle, top_gap, and robust_z require a llama.cpp build that implements them. The standard names dry, top_k, top_p, min_p, typ_p, top_n_sigma, xtc, penalties, and temperature work on any recent stock llama.cpp.

## Install

The simplest path is to clone the folder into the oh-my-pi user extensions directory, where it loads automatically on the next start.

```
git clone https://github.com/Hellisotherpeople/omp-samplers.git ~/.omp/agent/extensions/omp-samplers
```

Restart omp and type `/samplers` to confirm it loaded. You can also load it once for a single run with `omp --extension /path/to/omp-samplers`, or reference it from the `extensions` array in `~/.omp/agent/config.yml`.

## Usage

Type `/samplers` to open the main menu. From there you can edit the chain by adding, removing, or reordering samplers, tune any sampler's parameters through a dropdown of sensible values with a custom entry option, apply a ready-made preset, toggle the whole override off to fall back to your server defaults, or reset to the default chain of dry, hill, xtc, and temperature.

Three shortcuts cover the common actions. `/sampler-preset` jumps straight to the preset picker, and it also accepts a name, so `/sampler-preset coding` applies that preset with no menu at all. `/temp` sets temperature, either from an argument like `/temp 0.7` or through a dropdown when called bare. `/samplers-off` toggles the override.

A status line in the footer always shows the active chain and its live parameter values, so you can see at a glance what the next prompt will use. Everything you choose persists to a small JSON file under your agent directory, so your setup survives restarts.

## Presets

The bundled presets cover the common cases. There is a creative chain built around the hill sampler at temperature 10, two variants that swap in top_h or geo_mean, a balanced everyday chain, a low-entropy coding chain that keeps tool-call JSON well formed, and a near-deterministic greedy chain. Applying a preset replaces the chain and updates the relevant knobs in one step.

## A note on high temperature

Very high temperature paired with an aggressive creative chain makes a coding agent unreliable, because tool-call JSON tends to break under high-entropy sampling. Reach for the coding or balanced presets when you want omp to drive tools, and save the wild chains for creative writing and sampler research.

## License

MIT. See [LICENSE](LICENSE).

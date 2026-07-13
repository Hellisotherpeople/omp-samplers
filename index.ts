/**
 * Per-prompt sampler control for oh-my-pi + local llama.cpp (mink build).
 *
 * Lets you pick — via dropdowns, no commands to memorize — an ordered chain of
 * samplers and tune each one's knobs. The chosen chain is injected into every
 * outgoing llama.cpp request body (as `samplers: [...]` plus the relevant knob
 * fields) so the server reconfigures its sampler stack per request.
 *
 * How it works: llama.cpp's OpenAI-compatible endpoint honors a per-request
 * `samplers` array and individual sampler params. omp's `before_provider_request`
 * event hands us the outgoing request body by reference; we mutate it in place
 * (synchronously, before the body is serialized) to stamp in the active profile.
 *
 * Entry points (type `/` to see them):
 *   /samplers        - open the interactive menu (chain editor, tuning, presets)
 *   /sampler-preset  - jump straight to the preset picker
 *   /temp            - quick-set temperature (arg or dropdown)
 *   /samplers-off    - toggle the override on/off
 *
 * Config persists to  $PI_CODING_AGENT_DIR/sampler-profile.json  (default
 * ~/.omp/agent/sampler-profile.json) so your choice survives restarts.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

// ---------------------------------------------------------------------------
// Sampler catalog: the vocabulary the mink llama.cpp build accepts in the
// per-request `samplers` array, plus each sampler's tunable knob(s). Knob keys
// and defaults were read from the live server's /props.
// ---------------------------------------------------------------------------

type KnobKind = "float" | "int" | "bool";

interface Knob {
	key: string; // request-body field name
	label: string;
	kind: KnobKind;
	def: number | boolean; // default value
	presets: (number | boolean)[]; // quick-pick values for the dropdown
}

interface SamplerDef {
	id: string; // canonical name used in the `samplers` array
	label: string;
	blurb: string;
	knobs: Knob[];
}

const CATALOG: SamplerDef[] = [
	{
		id: "dry",
		label: "dry — DRY repetition penalty",
		blurb: "Penalizes verbatim n-gram repetition.",
		knobs: [
			{ key: "dry_multiplier", label: "multiplier", kind: "float", def: 0.8, presets: [0, 0.5, 0.8, 1.0] },
			{ key: "dry_base", label: "base", kind: "float", def: 1.75, presets: [1.5, 1.75, 2.0] },
			{ key: "dry_allowed_length", label: "allowed length", kind: "int", def: 15, presets: [2, 5, 10, 15] },
			{ key: "dry_penalty_last_n", label: "penalty last n", kind: "int", def: -1, presets: [-1, 0, 64, 256] },
		],
	},
	{
		id: "penalties",
		label: "penalties — repeat/freq/presence",
		blurb: "Classic repetition, frequency and presence penalties.",
		knobs: [
			{ key: "repeat_penalty", label: "repeat penalty", kind: "float", def: 1.0, presets: [1.0, 1.05, 1.1, 1.2] },
			{ key: "repeat_last_n", label: "repeat last n", kind: "int", def: 64, presets: [0, 64, 128, 256] },
			{ key: "frequency_penalty", label: "frequency penalty", kind: "float", def: 0, presets: [0, 0.5, 1.0] },
			{ key: "presence_penalty", label: "presence penalty", kind: "float", def: 0, presets: [0, 0.5, 1.0] },
		],
	},
	{
		id: "top_k",
		label: "top_k — keep K most likely",
		blurb: "Hard cap on candidate count. 0 disables.",
		knobs: [{ key: "top_k", label: "k", kind: "int", def: 20, presets: [0, 20, 40, 80, 100] }],
	},
	{
		id: "top_p",
		label: "top_p — nucleus",
		blurb: "Keep smallest set with cumulative prob ≥ p.",
		knobs: [{ key: "top_p", label: "p", kind: "float", def: 0.95, presets: [0.8, 0.9, 0.95, 0.99, 1.0] }],
	},
	{
		id: "min_p",
		label: "min_p — relative floor",
		blurb: "Drop tokens below p × top-token prob.",
		knobs: [{ key: "min_p", label: "p", kind: "float", def: 0.05, presets: [0, 0.02, 0.05, 0.1] }],
	},
	{
		id: "typ_p",
		label: "typ_p — locally typical",
		blurb: "Keep tokens near the distribution's entropy.",
		knobs: [{ key: "typical_p", label: "p", kind: "float", def: 1.0, presets: [0.5, 0.9, 0.95, 1.0] }],
	},
	{
		id: "top_n_sigma",
		label: "top_n_sigma — logit σ gate",
		blurb: "Keep logits within N standard deviations. -1 disables.",
		knobs: [{ key: "top_n_sigma", label: "n", kind: "float", def: -1, presets: [-1, 1, 2, 3] }],
	},
	{
		id: "xtc",
		label: "xtc — exclude top choices",
		blurb: "Probabilistically drops the most-predictable head (creativity engine).",
		knobs: [
			{ key: "xtc_probability", label: "probability", kind: "float", def: 0.5, presets: [0, 0.25, 0.5, 0.75, 1.0] },
			{ key: "xtc_threshold", label: "threshold", kind: "float", def: 0.1, presets: [0.05, 0.1, 0.15, 0.2] },
		],
	},
	{
		id: "min_k",
		label: "min_k — raw-logit cliff (custom)",
		blurb: "Temperature-invariant semantic cliff; tightest gate.",
		knobs: [{ key: "min_k_tau", label: "tau", kind: "float", def: 3.0, presets: [1, 2, 3, 4, 5] }],
	},
	{
		id: "p_less",
		label: "p_less — collision-prob gate (custom)",
		blurb: "Keep p ≥ Σpᵢ^q. Higher exponent → wider/more creative.",
		knobs: [
			{ key: "p_less_exponent", label: "exponent q", kind: "float", def: 2.0, presets: [1.5, 2, 3, 4] },
			{ key: "p_less_norm", label: "normalize", kind: "bool", def: false, presets: [true, false] },
		],
	},
	{
		id: "top_h",
		label: "top_h — entropy head (custom)",
		blurb: "Keep smallest head with entropy ≤ α·H(p). α≈0.45 is the temp-10 sweet spot.",
		knobs: [{ key: "top_h_alpha", label: "alpha", kind: "float", def: 0.4, presets: [0.3, 0.4, 0.45, 0.5, 0.6] }],
	},
	{
		id: "hill",
		label: "hill — Hill-number keep (custom)",
		blurb: "Keep ⌈D_q⌉ tokens. HIGHER q → FEWER tokens / more restrictive; q≈3 coherent+vivid at temp 10.",
		knobs: [{ key: "hill_order", label: "order q", kind: "float", def: 3.0, presets: [1.5, 2, 3, 4, 6, 10] }],
	},
	{
		id: "geo_mean",
		label: "geo_mean — geometric-mean gate (custom)",
		blurb: "Keep pᵢ ≥ coeff·e^{-H}. coeff≈0.7 is a good creative width.",
		knobs: [{ key: "geo_mean_coeff", label: "coeff", kind: "float", def: 1.0, presets: [0.5, 0.7, 1.0, 1.5] }],
	},
	{
		id: "robust_z",
		label: "robust_z — median+MAD gate (custom)",
		blurb: "Keep logits ≥ median + c·MAD.",
		knobs: [{ key: "robust_z_coeff", label: "coeff c", kind: "float", def: 3.0, presets: [1, 2, 3] }],
	},
	{ id: "otsu", label: "otsu — variance-max split (custom)", blurb: "Otsu logit bipartition. Wide; better at lower temp.", knobs: [] },
	{ id: "kneedle", label: "kneedle — elbow cut (custom)", blurb: "Max-curvature elbow. Wide; better at lower temp.", knobs: [] },
	{ id: "top_gap", label: "top_gap — largest logit gap (custom)", blurb: "Cut at the biggest logit gap. Tight/borderline.", knobs: [] },
	{
		id: "temperature",
		label: "temperature — softmax scaling",
		blurb: "Scales logits. Put LAST so truncators see cold logits.",
		knobs: [
			{ key: "temperature", label: "temperature", kind: "float", def: 10.0, presets: [0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0] },
			{ key: "dynatemp_range", label: "dynatemp range", kind: "float", def: 0, presets: [0, 0.5, 1.0] },
			{ key: "dynatemp_exponent", label: "dynatemp exponent", kind: "float", def: 1.0, presets: [1.0, 1.5] },
		],
	},
];

const BY_ID = new Map(CATALOG.map((s) => [s.id, s]));
const ALL_KNOBS: Knob[] = CATALOG.flatMap((s) => s.knobs);

// ---------------------------------------------------------------------------
// Presets: ready-made chains. First one is the user's stated default.
// ---------------------------------------------------------------------------

interface Preset {
	name: string;
	blurb: string;
	chain: string[];
	params: Record<string, number | boolean>;
}

const PRESETS: Preset[] = [
	{
		name: "hill-creative (default)",
		blurb: "dry → hill → xtc → temperature @ temp 10, hill q=3",
		chain: ["dry", "hill", "xtc", "temperature"],
		params: { temperature: 10.0, hill_order: 3.0, xtc_probability: 0.5, xtc_threshold: 0.1 },
	},
	{
		name: "toph-creative",
		blurb: "dry → top_h → xtc → temperature @ temp 10, α=0.45",
		chain: ["dry", "top_h", "xtc", "temperature"],
		params: { temperature: 10.0, top_h_alpha: 0.45, xtc_probability: 0.5, xtc_threshold: 0.1 },
	},
	{
		name: "geomean-creative",
		blurb: "dry → geo_mean → xtc → temperature @ temp 10, coeff 0.7",
		chain: ["dry", "geo_mean", "xtc", "temperature"],
		params: { temperature: 10.0, geo_mean_coeff: 0.7, xtc_probability: 0.5, xtc_threshold: 0.1 },
	},
	{
		name: "balanced",
		blurb: "penalties → top_k → top_p → min_p → temperature @ temp 0.7",
		chain: ["penalties", "top_k", "top_p", "min_p", "temperature"],
		params: { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.1 },
	},
	{
		name: "coding (reliable tool calls)",
		blurb: "top_k → top_p → temperature @ temp 0.3 — low-entropy, keeps tool JSON valid",
		chain: ["top_k", "top_p", "temperature"],
		params: { temperature: 0.3, top_k: 40, top_p: 0.95 },
	},
	{
		name: "greedy",
		blurb: "top_k → temperature @ temp 0 — near-deterministic",
		chain: ["top_k", "temperature"],
		params: { temperature: 0, top_k: 20 },
	},
];

// ---------------------------------------------------------------------------
// State + persistence
// ---------------------------------------------------------------------------

interface Profile {
	enabled: boolean;
	applyToAllModels: boolean; // if false, only apply to llama-ish providers
	chain: string[];
	params: Record<string, number | boolean>;
}

const AGENT_DIR = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".omp", "agent");
const STORE = path.join(AGENT_DIR, "sampler-profile.json");

function defaultProfile(): Profile {
	// Seed knob values with every catalog default, then apply the default preset.
	const params: Record<string, number | boolean> = {};
	for (const k of ALL_KNOBS) params[k.key] = k.def;
	Object.assign(params, PRESETS[0].params);
	return { enabled: true, applyToAllModels: false, chain: [...PRESETS[0].chain], params };
}

function loadProfile(): Profile {
	try {
		const raw = JSON.parse(fs.readFileSync(STORE, "utf8"));
		const base = defaultProfile();
		return {
			enabled: typeof raw.enabled === "boolean" ? raw.enabled : base.enabled,
			applyToAllModels: typeof raw.applyToAllModels === "boolean" ? raw.applyToAllModels : base.applyToAllModels,
			chain: Array.isArray(raw.chain) && raw.chain.every((x: unknown) => typeof x === "string" && BY_ID.has(x as string))
				? raw.chain
				: base.chain,
			params: { ...base.params, ...(raw.params && typeof raw.params === "object" ? raw.params : {}) },
		};
	} catch {
		return defaultProfile();
	}
}

let profile = loadProfile();

function save(): void {
	try {
		fs.mkdirSync(AGENT_DIR, { recursive: true });
		fs.writeFileSync(STORE, JSON.stringify(profile, null, 2));
	} catch {
		/* best effort */
	}
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

function fmt(v: number | boolean): string {
	if (typeof v === "boolean") return v ? "true" : "false";
	return Number.isInteger(v) ? String(v) : String(v);
}

/** Compact one-line summary of the active chain + its live knob values. */
function summarize(): string {
	if (!profile.enabled) return "samplers: OFF (server defaults)";
	const parts = profile.chain.map((id) => {
		const def = BY_ID.get(id);
		if (!def || def.knobs.length === 0) return id;
		const kv = def.knobs
			.filter((k) => k.key in profile.params)
			.map((k) => `${k.label.split(" ")[0]}=${fmt(profile.params[k.key])}`)
			.join(",");
		return kv ? `${id}(${kv})` : id;
	});
	return `samplers: ${parts.join(" → ")}`;
}

function refreshStatus(ctx: ExtensionContext): void {
	try {
		ctx.ui.setStatus("samplers", profile.enabled ? summarize() : "samplers: OFF");
	} catch {
		/* no UI (print/rpc mode) */
	}
}

// ---------------------------------------------------------------------------
// THE core: inject the active chain into every outgoing llama.cpp request.
// Handler is synchronous and mutates the body in place — the OpenAI-completions
// provider passes the request body by reference and reads it immediately after
// this call, so an in-place mutation lands before serialization. (Its return
// value is ignored by that provider, hence: mutate, don't just return.)
// ---------------------------------------------------------------------------

function looksLikeLlama(ctx: ExtensionContext): boolean {
	const p = (ctx.model?.provider ?? "").toLowerCase();
	const id = (ctx.model?.id ?? "").toLowerCase();
	return p.includes("llama") || p.includes("local") || id.includes("qwen") || id.includes("neo");
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI): void {
	pi.logger.debug("samplers extension loaded", { store: STORE, chain: profile.chain.join(",") });

	pi.on("before_provider_request", (event, ctx) => {
		if (!profile.enabled) return;
		const body = event.payload as Record<string, unknown> | undefined;
		if (!body || typeof body !== "object") return;
		if (!profile.applyToAllModels && !looksLikeLlama(ctx)) return;

		body.samplers = [...profile.chain];
		for (const [k, v] of Object.entries(profile.params)) body[k] = v;
		return body; // also return, for providers that honor the replacement path
	});

	pi.on("session_start", async (_e, ctx) => refreshStatus(ctx));

	// --- helpers bound to a command context (they use the dropdown UI) -------

	async function pickFromCatalog(ctx: ExtensionCommandContext, title: string): Promise<string | undefined> {
		const options = CATALOG.map((s) => ({ label: s.label, description: s.blurb }));
		const chosen = await ctx.ui.select(title, options);
		if (!chosen) return undefined;
		return CATALOG.find((s) => s.label === chosen)?.id;
	}

	/** Prompt for a single knob's value: preset dropdown + "Custom…". */
	async function editKnob(ctx: ExtensionCommandContext, knob: Knob): Promise<void> {
		const cur = profile.params[knob.key] ?? knob.def;
		if (knob.kind === "bool") {
			const pick = await ctx.ui.select(`${knob.label} (current: ${fmt(cur)})`, ["true", "false"]);
			if (pick === undefined) return;
			profile.params[knob.key] = pick === "true";
		} else {
			const presetLabels = knob.presets.map((v) => `${fmt(v)}${v === cur ? "  (current)" : ""}`);
			const CUSTOM = "Custom…";
			const pick = await ctx.ui.select(`${knob.label} (current: ${fmt(cur)})`, [...presetLabels, CUSTOM]);
			if (pick === undefined) return;
			let val: number;
			if (pick === CUSTOM) {
				const raw = await ctx.ui.input(`Enter ${knob.label}`, String(cur));
				if (raw === undefined || raw.trim() === "") return;
				val = Number(raw);
				if (!Number.isFinite(val)) {
					ctx.ui.notify(`Not a number: ${raw}`, "error");
					return;
				}
			} else {
				val = Number(pick.replace(/\s*\(current\)\s*$/, ""));
			}
			profile.params[knob.key] = knob.kind === "int" ? Math.round(val) : val;
		}
		save();
	}

	/** Tune every knob of one sampler, one after another. */
	async function tuneSampler(ctx: ExtensionCommandContext, id: string): Promise<void> {
		const def = BY_ID.get(id);
		if (!def) return;
		if (def.knobs.length === 0) {
			ctx.ui.notify(`${id} has no tunable parameters.`, "info");
			return;
		}
		for (;;) {
			const DONE = "‹ done›";
			const opts = def.knobs.map((k) => ({
				label: `${k.label}: ${fmt(profile.params[k.key] ?? k.def)}`,
				description: `default ${fmt(k.def)}`,
			}));
			const pick = await ctx.ui.select(`Tune ${id} — pick a parameter`, [...opts, DONE]);
			if (pick === undefined || pick === DONE) return;
			const knob = def.knobs.find((k) => pick.startsWith(`${k.label}:`));
			if (knob) await editKnob(ctx, knob);
		}
	}

	/** Interactive chain editor: add / remove / reorder, looping until done. */
	async function editChain(ctx: ExtensionCommandContext): Promise<void> {
		for (;;) {
			const ADD = "➕ Add a sampler";
			const REMOVE = "➖ Remove a sampler";
			const UP = "⬆️  Move a sampler earlier";
			const DOWN = "⬇️  Move a sampler later";
			const DONE = "✓ Done";
			const action = await ctx.ui.select(
				`Chain: ${profile.chain.join(" → ") || "(empty)"}`,
				[ADD, REMOVE, UP, DOWN, DONE],
			);
			if (action === undefined || action === DONE) {
				save();
				refreshStatus(ctx);
				return;
			}
			if (action === ADD) {
				const id = await pickFromCatalog(ctx, "Add which sampler? (appended to the end)");
				if (id) {
					profile.chain.push(id);
					save();
				}
			} else if (action === REMOVE) {
				if (profile.chain.length === 0) continue;
				const pick = await ctx.ui.select("Remove which?", profile.chain);
				if (pick) {
					const i = profile.chain.indexOf(pick);
					if (i >= 0) profile.chain.splice(i, 1);
					save();
				}
			} else if (action === UP || action === DOWN) {
				if (profile.chain.length < 2) continue;
				const pick = await ctx.ui.select(action === UP ? "Move earlier:" : "Move later:", profile.chain);
				if (!pick) continue;
				const i = profile.chain.indexOf(pick);
				const j = action === UP ? i - 1 : i + 1;
				if (i >= 0 && j >= 0 && j < profile.chain.length) {
					[profile.chain[i], profile.chain[j]] = [profile.chain[j], profile.chain[i]];
					save();
				}
			}
		}
	}

	async function applyPreset(ctx: ExtensionCommandContext): Promise<void> {
		const opts = PRESETS.map((p) => ({ label: p.name, description: p.blurb }));
		const pick = await ctx.ui.select("Apply a preset", opts);
		if (!pick) return;
		const preset = PRESETS.find((p) => p.name === pick);
		if (!preset) return;
		profile.chain = [...preset.chain];
		Object.assign(profile.params, preset.params);
		profile.enabled = true;
		save();
		refreshStatus(ctx);
		ctx.ui.notify(`Applied "${preset.name}". ${summarize()}`, "info");
	}

	// --- /samplers : the main menu ------------------------------------------

	pi.registerCommand("samplers", {
		description: "Configure llama.cpp sampler chain + parameters (dropdown menu)",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			if (!ctx.hasUI) {
				ctx.ui.notify(summarize(), "info");
				return;
			}
			// Allow a couple of direct subcommands, else open the menu.
			const arg = args.trim().toLowerCase();
			if (arg === "show") {
				ctx.ui.notify(summarize(), "info");
				return;
			}
			if (arg === "preset") return applyPreset(ctx);
			if (arg === "chain") return editChain(ctx);

			for (;;) {
				const CHAIN = "🔗 Edit chain (add / remove / reorder)";
				const TUNE = "🎚️  Tune a sampler's parameters";
				const PRESET = "⭐ Apply a preset";
				const SHOW = "👁️  Show current config";
				const TOGGLE = profile.enabled ? "⏻ Disable override (use server defaults)" : "⏻ Enable override";
				const RESET = "↺ Reset to default (dry → hill → xtc → temperature)";
				const DONE = "✓ Close";
				const choice = await ctx.ui.select(summarize(), [CHAIN, TUNE, PRESET, SHOW, TOGGLE, RESET, DONE], {
					helpText: "↑↓ move · enter select · esc close",
				});
				if (choice === undefined || choice === DONE) {
					refreshStatus(ctx);
					return;
				}
				if (choice === CHAIN) {
					await editChain(ctx);
				} else if (choice === TUNE) {
					const id = profile.chain.length
						? await ctx.ui.select("Tune which sampler?", profile.chain)
						: await pickFromCatalog(ctx, "Tune which sampler?");
					if (id) await tuneSampler(ctx, BY_ID.has(id) ? id : (id as string));
				} else if (choice === PRESET) {
					await applyPreset(ctx);
				} else if (choice === SHOW) {
					ctx.ui.notify(
						`${summarize()}\n(applies to ${profile.applyToAllModels ? "all models" : "local llama.cpp only"}; stored at ${STORE})`,
						"info",
					);
				} else if (choice === TOGGLE) {
					profile.enabled = !profile.enabled;
					save();
					refreshStatus(ctx);
				} else if (choice === RESET) {
					profile = defaultProfile();
					save();
					refreshStatus(ctx);
					ctx.ui.notify(`Reset. ${summarize()}`, "info");
				}
			}
		},
	});

	// --- /sampler-preset : jump straight to the preset picker ----------------

	pi.registerCommand("sampler-preset", {
		description: "Apply a ready-made sampler preset",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const arg = args.trim().toLowerCase();
			if (arg) {
				const preset = PRESETS.find((p) => p.name.toLowerCase().startsWith(arg));
				if (preset) {
					profile.chain = [...preset.chain];
					Object.assign(profile.params, preset.params);
					profile.enabled = true;
					save();
					refreshStatus(ctx);
					ctx.ui.notify(`Applied "${preset.name}". ${summarize()}`, "info");
					return;
				}
			}
			if (!ctx.hasUI) {
				ctx.ui.notify(`Presets: ${PRESETS.map((p) => p.name).join(", ")}`, "info");
				return;
			}
			await applyPreset(ctx);
		},
	});

	// --- /temp : quick temperature set --------------------------------------

	pi.registerCommand("temp", {
		description: "Quick-set sampling temperature",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const arg = args.trim();
			if (arg) {
				const v = Number(arg);
				if (!Number.isFinite(v)) {
					ctx.ui.notify(`Not a number: ${arg}`, "error");
					return;
				}
				profile.params.temperature = v;
				profile.enabled = true;
				save();
				refreshStatus(ctx);
				ctx.ui.notify(`temperature = ${v}. ${summarize()}`, "info");
				return;
			}
			if (!ctx.hasUI) return;
			const knob = ALL_KNOBS.find((k) => k.key === "temperature")!;
			await editKnob(ctx, knob);
			refreshStatus(ctx);
			ctx.ui.notify(summarize(), "info");
		},
	});

	// --- /samplers-off : toggle -------------------------------------------------

	pi.registerCommand("samplers-off", {
		description: "Toggle the sampler override on/off (off = server defaults)",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			profile.enabled = !profile.enabled;
			save();
			refreshStatus(ctx);
			ctx.ui.notify(profile.enabled ? `Override ON. ${summarize()}` : "Override OFF (server defaults).", "info");
		},
	});
}

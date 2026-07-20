import { describe, expect, test } from "bun:test";
import {
	applySamplerRoute,
	buildSamplerRouterSchema,
	parseSamplerRoute,
	prepareSamplerRouterRequest,
	type RouterSamplerDef,
	removeSamplerRouterTool,
} from "./router";

const catalog: RouterSamplerDef[] = [
	{
		id: "top_k",
		label: "top_k",
		blurb: "Keep K candidates.",
		knobs: [
			{
				key: "top_k",
				label: "k",
				kind: "int",
				def: 40,
				presets: [20, 40],
				min: 0,
				max: 1000,
			},
		],
	},
	{
		id: "temperature",
		label: "temperature",
		blurb: "Scale logits.",
		knobs: [
			{
				key: "temperature",
				label: "temperature",
				kind: "float",
				def: 0.7,
				presets: [0, 0.7, 1],
				min: 0,
				max: 20,
			},
		],
	},
];

describe("sampler router schema", () => {
	test("contains every sampler and knob in one compact ordered-item schema", () => {
		const schema = buildSamplerRouterSchema(catalog, 5) as {
			properties: {
				samplers: {
					minItems: number;
					maxItems: number;
					items: {
						properties: Record<string, { enum?: string[] }>;
						required: string[];
						additionalProperties: boolean;
					};
				};
			};
		};
		const list = schema.properties.samplers;
		expect(list.minItems).toBe(1);
		expect(list.maxItems).toBe(5);
		expect(list.items.properties.id.enum).toEqual(["top_k", "temperature"]);
		expect(Object.keys(list.items.properties)).toEqual(["id", "values"]);
		expect(list.items.required).toEqual(["id"]);
		expect(list.items.additionalProperties).toBe(false);
	});

	test("omits rationale entirely when disabled", () => {
		const schema = buildSamplerRouterSchema(catalog, 5, false) as {
			properties: Record<string, unknown>;
			required: string[];
		};
		expect(schema.properties.reason).toBeUndefined();
		expect(schema.required).toEqual(["samplers"]);
	});
});

describe("parseSamplerRoute", () => {
	test("preserves order and extracts only the selected settings", () => {
		const result = parseSamplerRoute(
			{
				samplers: [
					{ id: "top_k", values: [24] },
					{ id: "temperature", values: [0.25] },
				],
				reason: "Conservative code generation.",
			},
			catalog,
		);
		expect(result).toEqual({
			ok: true,
			route: {
				chain: ["top_k", "temperature"],
				params: { top_k: 24, temperature: 0.25 },
				reason: "Conservative code generation.",
			},
		});
	});

	test("accepts JSON-string arguments and fills a missing knob from its catalog default", () => {
		const result = parseSamplerRoute(
			JSON.stringify({ samplers: [{ id: "top_k" }], reason: "Default cap." }),
			catalog,
		);
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.route.params.top_k).toBe(40);
	});

	test("rejects duplicates, hallucinated settings, and out-of-range values", () => {
		expect(
			parseSamplerRoute(
				{
					samplers: [
						{ id: "top_k", values: [20] },
						{ id: "top_k", values: [30] },
					],
					reason: "duplicate",
				},
				catalog,
			),
		).toMatchObject({ ok: false, error: "duplicate sampler: top_k" });
		expect(
			parseSamplerRoute(
				{
					samplers: [{ id: "top_k", top_p: 0.9 }],
					reason: "bad field",
				},
				catalog,
			),
		).toMatchObject({ ok: false, error: "unknown setting for top_k: top_p" });
		expect(
			parseSamplerRoute(
				{
					samplers: [{ id: "temperature", values: [21] }],
					reason: "too hot",
				},
				catalog,
			),
		).toMatchObject({ ok: false, error: "temperature must be <= 20" });
	});

	test("does not require or retain a rationale when disabled", () => {
		const result = parseSamplerRoute(
			{ samplers: [{ id: "temperature", values: [0.2] }] },
			catalog,
			5,
			false,
		);
		expect(result).toEqual({
			ok: true,
			route: {
				chain: ["temperature"],
				params: { temperature: 0.2 },
				reason: "",
			},
		});
	});

	test("rejects positional setting counts that do not match the sampler", () => {
		expect(
			parseSamplerRoute(
				{ samplers: [{ id: "top_k", values: [20, 0.9] }], reason: "bad count" },
				catalog,
			),
		).toMatchObject({
			ok: false,
			error: "values for top_k must contain at most 1 entries",
		});
	});

	test("fills omitted trailing positional settings from defaults", () => {
		const multiKnobCatalog: RouterSamplerDef[] = [
			{
				id: "pair",
				label: "pair",
				blurb: "Two settings.",
				knobs: [
					{
						key: "first",
						label: "first",
						kind: "int",
						def: 1,
						presets: [1],
						min: 0,
						max: 10,
					},
					{
						key: "second",
						label: "second",
						kind: "bool",
						def: false,
						presets: [false, true],
					},
				],
			},
		];
		const result = parseSamplerRoute(
			{ samplers: [{ id: "pair", values: [2] }], reason: "partial" },
			multiKnobCatalog,
		);
		expect(result).toMatchObject({
			ok: true,
			route: { params: { first: 2, second: false } },
		});
	});
});

test("applySamplerRoute propagates the chain and settings without dropping request fields", () => {
	const body: Record<string, unknown> = {
		model: "local",
		stream: true,
		temperature: 99,
	};
	const result = applySamplerRoute(body, {
		chain: ["top_k", "temperature"],
		params: { top_k: 24, temperature: 0.25 },
		reason: "Conservative.",
	});
	expect(result).toBe(body);
	expect(body).toEqual({
		model: "local",
		stream: true,
		samplers: ["top_k", "temperature"],
		top_k: 24,
		temperature: 0.25,
	});
});

test("prepareSamplerRouterRequest isolates and forces the routing tool", () => {
	const body: Record<string, unknown> = {
		model: "local",
		tools: [{ type: "function", function: { name: "bash" } }],
		chat_template_kwargs: { existing: true },
	};
	const schema = buildSamplerRouterSchema(catalog, 5, false);
	prepareSamplerRouterRequest(body, "route_samplers", schema, 128);
	expect(body).toMatchObject({
		model: "local",
		tools: [
			{
				type: "function",
				function: {
					name: "route_samplers",
					parameters: schema,
					strict: false,
				},
			},
		],
		tool_choice: "required",
		parallel_tool_calls: false,
		max_tokens: 128,
		samplers: ["top_k", "top_p", "temperature"],
		top_k: 20,
		top_p: 0.9,
		temperature: 0.1,
		cache_prompt: true,
		chat_template_kwargs: { existing: true, enable_thinking: false },
	});
	removeSamplerRouterTool(body, "route_samplers");
	expect(body.tools).toBeUndefined();
	expect(body.tool_choice).toBeUndefined();
});

test("removeSamplerRouterTool keeps ordinary tools and clears a stale force", () => {
	const body: Record<string, unknown> = {
		tools: [
			{ type: "function", function: { name: "route_samplers" } },
			{ type: "function", function: { name: "bash" } },
		],
		tool_choice: {
			type: "function",
			function: { name: "route_samplers" },
		},
	};
	removeSamplerRouterTool(body, "route_samplers");
	expect(body.tools).toEqual([
		{ type: "function", function: { name: "bash" } },
	]);
	expect(body.tool_choice).toBeUndefined();
});

test("prepareSamplerRouterRequest respects the provider's output-token field", () => {
	const body: Record<string, unknown> = {
		max_completion_tokens: 32_768,
		grammar: 'root ::= "old"',
	};
	prepareSamplerRouterRequest(
		body,
		"route_samplers",
		buildSamplerRouterSchema(catalog),
		256,
	);
	expect(body.max_completion_tokens).toBe(256);
	expect(body.max_tokens).toBeUndefined();
	expect(body.grammar).toBeUndefined();
});

test("prepareSamplerRouterRequest isolates the current turn from agent history", () => {
	const current = {
		role: "user",
		content: [{ type: "text", text: "current task" }],
	};
	const body: Record<string, unknown> = {
		messages: [
			{ role: "system", content: "large agent prompt" },
			{ role: "user", content: "old task" },
			{ role: "assistant", content: "old answer" },
			current,
		],
	};
	prepareSamplerRouterRequest(
		body,
		"route_samplers",
		buildSamplerRouterSchema(catalog),
		256,
		"compact router prompt",
	);
	expect(body.messages).toEqual([
		{ role: "system", content: "compact router prompt" },
		current,
	]);
});

import { describe, expect, test } from "bun:test";
import {
	applySamplerRoute,
	buildSamplerRouterSchema,
	parseSamplerRoute,
	type RouterSamplerDef,
	truncateRouterPrompt,
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

test("truncateRouterPrompt keeps both ends within the requested size", () => {
	const input = `${"a".repeat(100)}${"z".repeat(100)}`;
	const output = truncateRouterPrompt(input, 100);
	expect(output.length).toBe(100);
	expect(output.startsWith("a")).toBe(true);
	expect(output.endsWith("z")).toBe(true);
	expect(output).toContain("middle omitted");
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

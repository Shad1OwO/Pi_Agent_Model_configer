import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

type JsonObject = Record<string, unknown>;

type PiApi =
  | "openai-completions"
  | "openai-responses"
  | "anthropic-messages"
  | "google-generative-ai";

type InputType = "text" | "image";

type ModelCost = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
};

type DiscoveredModel = {
  id: string;
  name: string;
  reasoning: boolean;
  input: InputType[];
  contextWindow: number;
  maxTokens: number;
  cost: ModelCost;
};

type ModelsJson = {
  providers?: Record<string, JsonObject>;
  [key: string]: unknown;
};

const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;

/* ============================================================================
 * Generic helpers
 * ========================================================================== */

function isObject(value: unknown): value is JsonObject {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function firstString(
  ...values: unknown[]
): string | undefined {
  for (const value of values) {
    if (
      typeof value === "string" &&
      value.trim().length > 0
    ) {
      return value.trim();
    }
  }

  return undefined;
}

function firstNumber(
  ...values: unknown[]
): number | undefined {
  for (const value of values) {
    let n: number;

    if (typeof value === "number") {
      n = value;
    } else if (
      typeof value === "string" &&
      value.trim().length > 0
    ) {
      n = Number(value);
    } else {
      continue;
    }

    if (Number.isFinite(n) && n > 0) {
      return n;
    }
  }

  return undefined;
}

function firstBoolean(
  ...values: unknown[]
): boolean | undefined {
  for (const value of values) {
    if (typeof value === "boolean") {
      return value;
    }
  }

  return undefined;
}

function getNested(
  object: JsonObject,
  ...paths: string[][]
): unknown {
  for (const parts of paths) {
    let current: unknown = object;

    for (const part of parts) {
      if (
        !isObject(current) ||
        !(part in current)
      ) {
        current = undefined;
        break;
      }

      current = current[part];
    }

    if (current !== undefined) {
      return current;
    }
  }

  return undefined;
}

/* ============================================================================
 * URL /provider helpers
 * ========================================================================== */

function normalizeBaseUrl(
  input: string,
): string {
  let url = input.trim().replace(/\/+$/, "");

  url = url.replace(
    /\/(?:v1\/)?models$/i,
    "",
  );

  return url;
}

function deriveProviderId(
  baseUrl: string,
): string {
  try {
    const hostname = new URL(baseUrl)
      .hostname
      .toLowerCase()
      .replace(/^www\./, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");

    return hostname || "custom-provider";
  } catch {
    return "custom-provider";
  }
}

function sanitizeProviderId(
  value: string,
): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function inferApi(
  baseUrl: string,
): PiApi {
  const url = baseUrl.toLowerCase();

  if (
    url.includes("anthropic") ||
    url.includes("/anthropic") ||
    url.includes("claude")
  ) {
    return "anthropic-messages";
  }

  if (
    url.includes(
      "generativelanguage.googleapis.com",
    ) ||
    url.includes("googleapis.com") ||
    url.includes("gemini")
  ) {
    return "google-generative-ai";
  }

  if (url.includes("/responses")) {
    return "openai-responses";
  }

  return "openai-completions";
}

/* ============================================================================
 * Context-window discovery
 * ========================================================================== */

function getContextWindow(
  model: JsonObject,
): number {
  return (
    firstNumber(
      model.context_window,
      model.contextWindow,

      model.context_length,
      model.contextLength,

      model.max_context_length,
      model.maxContextLength,

      model.input_token_limit,
      model.inputTokenLimit,

      model.max_input_tokens,
      model.maxInputTokens,

      model.context_size,
      model.contextSize,

      getNested(
        model,
        ["limits", "context_window"],
      ),

      getNested(
        model,
        ["limits", "contextWindow"],
      ),

      getNested(
        model,
        ["limits", "context_length"],
      ),

      getNested(
        model,
        ["limits", "max_context_length"],
      ),

      getNested(
        model,
        ["limits", "context_size"],
      ),

      getNested(
        model,
        ["capabilities", "context_window"],
      ),

      getNested(
        model,
        ["capabilities", "contextWindow"],
      ),

      getNested(
        model,
        ["capabilities", "context_length"],
      ),

      getNested(
        model,
        ["capabilities", "max_context_length"],
      ),

      getNested(
        model,
        ["metadata", "context_window"],
      ),

      getNested(
        model,
        ["metadata", "contextWindow"],
      ),

      getNested(
        model,
        ["metadata", "context_length"],
      ),
    ) ?? DEFAULT_CONTEXT_WINDOW
  );
}

/* ============================================================================
 * Maximum output token discovery
 * ========================================================================== */

function getMaxTokens(
  model: JsonObject,
): number {
  return (
    firstNumber(
      model.max_tokens,
      model.maxTokens,

      model.max_output_tokens,
      model.maxOutputTokens,

      model.max_completion_tokens,
      model.maxCompletionTokens,

      model.output_token_limit,
      model.outputTokenLimit,

      model.max_output_length,
      model.maxOutputLength,

      getNested(
        model,
        ["limits", "max_tokens"],
      ),

      getNested(
        model,
        ["limits", "maxTokens"],
      ),

      getNested(
        model,
        ["limits", "max_output_tokens"],
      ),

      getNested(
        model,
        ["limits", "maxOutputTokens"],
      ),

      getNested(
        model,
        ["limits", "max_completion_tokens"],
      ),

      getNested(
        model,
        ["capabilities", "max_tokens"],
      ),

      getNested(
        model,
        ["capabilities", "max_output_tokens"],
      ),

      getNested(
        model,
        ["metadata", "max_tokens"],
      ),

      getNested(
        model,
        ["metadata", "max_output_tokens"],
      ),
    ) ?? DEFAULT_MAX_TOKENS
  );
}

/* ============================================================================
 * Reasoning discovery
 * ========================================================================== */

function detectReasoning(
  model: JsonObject,
): boolean {
  return (
    firstBoolean(
      model.reasoning,
      model.supports_reasoning,
      model.supportsReasoning,

      model.reasoning_capable,
      model.reasoningCapable,

      model.thinking,
      model.supports_thinking,
      model.supportsThinking,

      getNested(
        model,
        ["capabilities", "reasoning"],
      ),

      getNested(
        model,
        ["capabilities", "thinking"],
      ),

      getNested(
        model,
        ["metadata", "reasoning"],
      ),
    ) ?? false
  );
}

/* ============================================================================
 * Input-modality discovery
 * ========================================================================== */

function getInputTypes(
  model: JsonObject,
): InputType[] {
  let imageSupport = false;

  const modalityValues = [
    model.input_modalities,
    model.inputModalities,
    model.modalities,

    getNested(
      model,
      ["capabilities", "modalities"],
    ),

    getNested(
      model,
      ["metadata", "modalities"],
    ),
  ];

  for (const value of modalityValues) {
    if (!Array.isArray(value)) {
      continue;
    }

    for (const item of value) {
      if (typeof item !== "string") {
        continue;
      }

      const normalized =
        item.toLowerCase();

      if (
        normalized === "image" ||
        normalized === "images" ||
        normalized === "vision" ||
        normalized === "multimodal"
      ) {
        imageSupport = true;
      }
    }
  }

  if (
    firstBoolean(
      model.vision,
      model.supports_vision,
      model.supportsVision,

      getNested(
        model,
        ["capabilities", "vision"],
      ),

      getNested(
        model,
        ["metadata", "vision"],
      ),
    ) === true
  ) {
    imageSupport = true;
  }

  return imageSupport
    ? ["text", "image"]
    : ["text"];
}

/* ============================================================================
 * Pricing discovery
 * ========================================================================== */

function getCost(
  model: JsonObject,
): ModelCost {
  const pricing =
    isObject(model.pricing)
      ? model.pricing
      : undefined;

  const cost =
    isObject(model.cost)
      ? model.cost
      : undefined;

  function normalizePrice(
    value: unknown,
  ): number {
    const n = firstNumber(value) ?? 0;

    if (n > 0 && n < 0.01) {
      return n * 1_000_000;
    }

    return n;
  }

  return {
    input: normalizePrice(
      model.input_cost ??
        model.inputCost ??
        pricing?.input ??
        pricing?.prompt ??
        cost?.input,
    ),

    output: normalizePrice(
      model.output_cost ??
        model.outputCost ??
        pricing?.output ??
        pricing?.completion ??
        cost?.output,
    ),

    cacheRead: normalizePrice(
      model.cache_read_cost ??
        model.cacheReadCost ??
        pricing?.cache_read ??
        pricing?.cacheRead ??
        cost?.cacheRead,
    ),

    cacheWrite: normalizePrice(
      model.cache_write_cost ??
        model.cacheWriteCost ??
        pricing?.cache_write ??
        pricing?.cacheWrite ??
        cost?.cacheWrite,
    ),
  };
}

/* ============================================================================
 * Provider response parsing
 * ========================================================================== */

function getModelArray(
  payload: unknown,
): unknown[] {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (!isObject(payload)) {
    return [];
  }

  for (
    const key of [
      "data",
      "models",
      "items",
      "results",
    ]
  ) {
    const value = payload[key];

    if (Array.isArray(value)) {
      return value;
    }
  }

  return [];
}

function normalizeModels(
  payload: unknown,
): DiscoveredModel[] {
  const models: DiscoveredModel[] = [];

  for (
    const value of getModelArray(payload)
  ) {
    if (!isObject(value)) {
      continue;
    }

    const id =
      firstString(
        value.id,
        value.model,
        value.slug,
      );

    if (!id) {
      continue;
    }

    const name =
      firstString(
        value.name,
        value.display_name,
        value.displayName,
        value.label,
      ) ?? id;

    models.push({
      id,
      name,

      reasoning:
        detectReasoning(value),

      input:
        getInputTypes(value),

      contextWindow:
        getContextWindow(value),

      maxTokens:
        getMaxTokens(value),

      cost:
        getCost(value),
    });
  }

  return [
    ...new Map(
      models.map(
        (model) => [
          model.id,
          model,
        ],
      ),
    ).values(),
  ];
}

/* ============================================================================
 * HTTP
 * ========================================================================== */

async function fetchJson(
  url: string,
  apiKey: string,
): Promise<unknown> {
  const headers: Record<
    string,
    string
  > = {
    Accept: "application/json",
  };

  if (apiKey.trim()) {
    headers.Authorization =
      `Bearer ${apiKey.trim()}`;
  }

  const response =
    await fetch(url, {
      method: "GET",
      headers,
    });

  const body =
    await response.text();

  if (!response.ok) {
    throw new Error(
      [
        `HTTP ${response.status} ${response.statusText}`,
        body.slice(0, 500),
      ].join("\n"),
    );
  }

  try {
    return JSON.parse(body);
  } catch {
    throw new Error(
      [
        "Provider returned invalid JSON:",
        body.slice(0, 500),
      ].join("\n"),
    );
  }
}

/* ============================================================================
 * Model discovery
 * ========================================================================== */

async function discoverModels(
  baseUrl: string,
  apiKey: string,
): Promise<{
  endpoint: string;
  models: DiscoveredModel[];
}> {
  const endpoints = [
    `${baseUrl}/models`,
    `${baseUrl}/v1/models`,
  ];

  let lastError:
    | unknown
    | undefined;

  for (
    const endpoint of [
      ...new Set(endpoints),
    ]
  ) {
    try {
      const payload =
        await fetchJson(
          endpoint,
          apiKey,
        );

      const models =
        normalizeModels(payload);

      if (models.length > 0) {
        return {
          endpoint,
          models,
        };
      }

      lastError =
        new Error(
          "The endpoint returned no usable models.",
        );
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    [
      "Could not discover provider models.",
      "",
      "Endpoints tried:",
      ...[
        ...new Set(endpoints),
      ].map(
        (url) => `  ${url}`,
      ),
      "",
      "Last error:",
      lastError instanceof Error
        ? lastError.message
        : String(lastError),
    ].join("\n"),
  );
}

/* ============================================================================
 * Manual model construction
 * ========================================================================== */

function createManualModel(
  id: string,
  supportsImage: boolean,
): DiscoveredModel {
  return {
    id,
    name: id,
    reasoning: false,
    input: supportsImage
      ? ["text", "image"]
      : ["text"],
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: DEFAULT_MAX_TOKENS,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    },
  };
}

/* ============================================================================
 * Model selection parsing
 * ========================================================================== */

function parseModelSelection(
  input: string,
  models: DiscoveredModel[],
): DiscoveredModel[] {
  const selected: DiscoveredModel[] = [];
  const seen = new Set<string>();

  const parts = input
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  for (const part of parts) {
    if (part.toLowerCase() === "all") {
      return models;
    }

    const rangeMatch =
      part.match(/^(\d+)-(\d+)$/);

    if (rangeMatch) {
      const start = parseInt(
        rangeMatch[1],
        10,
      );
      const end = parseInt(
        rangeMatch[2],
        10,
      );
      const lo = Math.min(start, end);
      const hi = Math.max(start, end);

      for (let i = lo; i <= hi; i++) {
        if (i >= 1 && i <= models.length) {
          const model = models[i - 1];

          if (!seen.has(model.id)) {
            seen.add(model.id);
            selected.push(model);
          }
        }
      }

      continue;
    }

    const num = Number(part);

    if (
      Number.isFinite(num) &&
      num >= 1 &&
      num <= models.length
    ) {
      const model = models[num - 1];

      if (!seen.has(model.id)) {
        seen.add(model.id);
        selected.push(model);
      }

      continue;
    }

    const model = models.find(
      (m) =>
        m.id.toLowerCase() ===
        part.toLowerCase(),
    );

    if (model && !seen.has(model.id)) {
      seen.add(model.id);
      selected.push(model);
    }
  }

  return selected;
}

/* ============================================================================
 * Pi paths
 * ========================================================================== */

function getPiAgentDirectory(): string {
  return path.join(
    os.homedir(),
    ".pi",
    "agent",
  );
}

function getModelsJsonPath(): string {
  return path.join(
    getPiAgentDirectory(),
    "models.json",
  );
}

/* ============================================================================
 * models.json
 * ========================================================================== */

async function readModelsJson(
  filePath: string,
): Promise<ModelsJson> {
  try {
    const text =
      await fs.readFile(
        filePath,
        "utf8",
      );

    const parsed =
      JSON.parse(text);

    if (!isObject(parsed)) {
      return {};
    }

    return parsed as ModelsJson;
  } catch (error) {
    const code =
      (error as NodeJS.ErrnoException)
        .code;

    if (code === "ENOENT") {
      return {};
    }

    throw error;
  }
}

async function writeProviderToModelsJson(
  providerId: string,
  provider: JsonObject,
): Promise<string> {
  const filePath =
    getModelsJsonPath();

  const existing =
    await readModelsJson(
      filePath,
    );

  const providers =
    isObject(existing.providers)
      ? {
          ...(existing.providers as Record<
            string,
            JsonObject
          >),
        }
      : {};

  providers[providerId] =
    provider;

  const output: ModelsJson = {
    ...existing,
    providers,
  };

  await fs.mkdir(
    getPiAgentDirectory(),
    {
      recursive: true,
    },
  );

  await fs.writeFile(
    filePath,
    JSON.stringify(
      output,
      null,
      2,
    ) + "\n",
    {
      encoding: "utf8",
    },
  );

  return filePath;
}

/* ============================================================================
 * Provider construction
 * ========================================================================== */

function buildProvider(
  providerId: string,
  baseUrl: string,
  apiKey: string,
  api: PiApi,
  models: DiscoveredModel[],
  reasoningOverride:
    | boolean
    | undefined,
  compatSettings?: JsonObject,
): JsonObject {
  const provider: JsonObject = {
    baseUrl,
    api,
    models: models.map(
      (model) => ({
        id: model.id,

        name: model.name,

        reasoning:
          reasoningOverride ??
          model.reasoning,

        input:
          model.input,

        cost:
          model.cost,

        contextWindow:
          model.contextWindow,

        maxTokens:
          model.maxTokens,
      }),
    ),
  };

  // Only add apiKey if it exists (omit for external auth configurations)
  if (apiKey) {
    provider.apiKey = apiKey;
  }

  if (compatSettings) {
    provider.compat = compatSettings;
  }

  return provider;
}

/* ============================================================================
 * Extension
 * ========================================================================== */

export default function (
  pi: ExtensionAPI,
) {
  pi.registerCommand(
    "setup-model-json",
    {
      description:
        "Discover provider models and generate/update ~/.pi/agent/models.json",

      handler:
        async (_args, ctx) => {
          if (!ctx.hasUI) {
            return;
          }

          try {
            /* ----------------------------------------------------------------
             * Provider URL
             * -------------------------------------------------------------- */

            const urlInput =
              await ctx.ui.input(
                "Provider URL",
                "https://api.example.com/v1",
              );

            if (
              !urlInput ||
              !urlInput.trim()
            ) {
              ctx.ui.notify(
                "Cancelled: provider URL is required.",
                "warning",
              );

              return;
            }

            const baseUrl =
              normalizeBaseUrl(
                urlInput,
              );

            /* ----------------------------------------------------------------
             * Infer API type early
             * -------------------------------------------------------------- */
            const api = inferApi(baseUrl);

            /* ----------------------------------------------------------------
             * API key
             * -------------------------------------------------------------- */
            const isLocal =
              baseUrl.includes("localhost") ||
              baseUrl.includes("127.0.0.1") ||
              baseUrl.includes("0.0.0.0");

            const defaultApiKey = isLocal
              ? "ollama"
              : "Enter API key";

            const apiKeyInput =
              await ctx.ui.input(
                "API Key (leave blank to omit)",
                defaultApiKey,
              );

            if (
              apiKeyInput === undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            const apiKey =
              apiKeyInput.trim() ||
              (isLocal ? "ollama" : "");

            /* ----------------------------------------------------------------
             * Provider ID
             * -------------------------------------------------------------- */

            const defaultProviderId =
              deriveProviderId(
                baseUrl,
              );

            const providerIdInput =
              await ctx.ui.input(
                "Provider ID",
                defaultProviderId,
              );

            if (
              providerIdInput ===
              undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            const providerId =
              sanitizeProviderId(
                providerIdInput ||
                  defaultProviderId,
              );

            if (!providerId) {
              throw new Error(
                "Provider ID is invalid.",
              );
            }

            /* ----------------------------------------------------------------
             * Compat Settings
             * -------------------------------------------------------------- */

            let compatSettings:
              | JsonObject
              | undefined;

            if (api === "openai-completions") {
              const defaultCompat = isLocal
                ? "yes"
                : "no";

              const compatInput =
                await ctx.ui.input(
                  "Disable 'developer' role & 'reasoning_effort' for compatibility? (Common for Ollama, vLLM, local servers)",
                  defaultCompat,
                );

              if (
                compatInput !== undefined
              ) {
                const choice =
                  compatInput
                    .trim()
                    .toLowerCase();

                if (
                  [
                    "yes",
                    "y",
                    "true",
                    "1",
                  ].includes(choice)
                ) {
                  compatSettings = {
                    supportsDeveloperRole: false,
                    supportsReasoningEffort: false,
                  };
                }
              }
            }

            /* ----------------------------------------------------------------
             * Reasoning support question
             * -------------------------------------------------------------- */

            const reasoningInput =
              await ctx.ui.input(
                "Reasoning support?",
                "yes / no / auto",
              );

            if (
              reasoningInput ===
              undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            const reasoningChoice =
              reasoningInput
                .trim()
                .toLowerCase();

            let reasoningOverride:
              | boolean
              | undefined;

            if (
              [
                "yes",
                "y",
                "true",
                "1",
              ].includes(
                reasoningChoice,
              )
            ) {
              reasoningOverride =
                true;
            } else if (
              [
                "no",
                "n",
                "false",
                "0",
              ].includes(
                reasoningChoice,
              )
            ) {
              reasoningOverride =
                false;
            } else if (
              reasoningChoice === "" ||
              reasoningChoice ===
                "auto" ||
              reasoningChoice ===
                "automatic"
            ) {
              reasoningOverride =
                undefined;
            } else {
              throw new Error(
                'Reasoning must be "yes", "no", or "auto".',
              );
            }

            /* ----------------------------------------------------------------
             * Auto-discover or manual entry
             * -------------------------------------------------------------- */

            const autoFetchInput =
              await ctx.ui.input(
                "Auto-discover models from provider?",
                "yes / no",
              );

            if (
              autoFetchInput ===
              undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            const autoFetchChoice =
              autoFetchInput
                .trim()
                .toLowerCase();

            const wantsAutoFetch = ![
              "no",
              "n",
              "false",
              "0",
            ].includes(
              autoFetchChoice,
            );

            let models: DiscoveredModel[];
            let discoveryEndpoint:
              | string
              | undefined;

            if (wantsAutoFetch) {
              ctx.ui.setWorkingMessage(
                "Fetching provider model metadata...",
              );

              const discovered =
                await discoverModels(
                  baseUrl,
                  apiKey,
                );

              discoveryEndpoint =
                discovered.endpoint;

              const listText =
                discovered.models
                  .map(
                    (
                      model,
                      index,
                    ) =>
                      `  ${index + 1}. ${model.id} (${model.name})`,
                  )
                  .join("\n");

              ctx.ui.notify(
                [
                  `Discovered ${discovered.models.length} models:`,
                  "",
                  listText,
                  "",
                  "Enter numbers (1,3,5), ranges (1-5), model IDs, or 'all'.",
                ].join("\n"),
                "info",
              );

              const selectionInput =
                await ctx.ui.input(
                  "Select models to add (or 'all')",
                  "all",
                );

              if (
                selectionInput ===
                undefined
              ) {
                ctx.ui.notify(
                  "Cancelled.",
                  "warning",
                );

                return;
              }

              const selection =
                selectionInput.trim();

              if (
                selection === "" ||
                selection.toLowerCase() ===
                  "all"
              ) {
                models =
                  discovered.models;
              } else {
                models =
                  parseModelSelection(
                    selection,
                    discovered.models,
                  );

                if (
                  models.length === 0
                ) {
                  throw new Error(
                    [
                      "No valid models selected.",
                      "",
                      "Use numbers (1,3,5), ranges (1-5), model IDs, or 'all'.",
                    ].join("\n"),
                  );
                }

                ctx.ui.notify(
                  `Selected ${models.length} of ${discovered.models.length} models.`,
                  "info",
                );
              }
            } else {
              const manualInput =
                await ctx.ui.input(
                  "Enter model IDs (comma-separated)",
                  "model-id-1, model-id-2",
                );

              if (
                manualInput ===
                undefined
              ) {
                ctx.ui.notify(
                  "Cancelled.",
                  "warning",
                );

                return;
              }

              const ids = manualInput
                .split(",")
                .map((s) =>
                  s.trim(),
                )
                .filter(Boolean);

              if (
                ids.length === 0
              ) {
                throw new Error(
                  "At least one model ID is required.",
                );
              }

              const imageInput =
                await ctx.ui.input(
                  "Do these models support image/vision input?",
                  "yes / no",
                );

              if (
                imageInput ===
                undefined
              ) {
                ctx.ui.notify(
                  "Cancelled.",
                  "warning",
                );

                return;
              }

              const imageChoice =
                imageInput
                  .trim()
                  .toLowerCase();

              const supportsImage = [
                "yes",
                "y",
                "true",
                "1",
              ].includes(
                imageChoice,
              );

              models = ids.map(
                (id) =>
                  createManualModel(
                    id,
                    supportsImage,
                  ),
              );

              discoveryEndpoint =
                undefined;

              ctx.ui.notify(
                `Created ${models.length} manual model entr${models.length === 1 ? "y" : "ies"}.`,
                "info",
              );
            }

            /* ----------------------------------------------------------------
             * Build provider
             * -------------------------------------------------------------- */

            const provider =
              buildProvider(
                providerId,
                baseUrl,
                apiKey,
                api,
                models,
                reasoningOverride,
                compatSettings,
              );

            /* ----------------------------------------------------------------
             * Write ~/.pi/agent/models.json
             * -------------------------------------------------------------- */

            const modelsPath =
              await writeProviderToModelsJson(
                providerId,
                provider,
              );

            /* ----------------------------------------------------------------
             * Register immediately
             * -------------------------------------------------------------- */

            pi.registerProvider(
              providerId,
              provider as any,
            );

            ctx.ui.setWorkingMessage(
              "",
            );

            /* ----------------------------------------------------------------
             * Result display
             * -------------------------------------------------------------- */

            const reasoningMode =
              reasoningOverride ===
              true
                ? "forced ON"
                : reasoningOverride ===
                    false
                  ? "forced OFF"
                  : "automatic";

            const compatLabel =
              compatSettings
                ? "Compat: developer role & reasoning_effort disabled"
                : "Compat: default";

            const sourceLabel =
              discoveryEndpoint
                ? `Discovery endpoint: ${discoveryEndpoint}`
                : "Source: manual entry";

            const preview =
              models
                .slice(0, 12)
                .map(
                  (model) =>
                    [
                      `  ${model.id}`,
                      `    contextWindow: ${model.contextWindow.toLocaleString()}`,
                      `    maxTokens: ${model.maxTokens.toLocaleString()}`,
                      `    reasoning: ${
                        reasoningOverride ??
                        model.reasoning
                      }`,
                    ].join("\n"),
                )
                .join("\n");

            const remaining =
              Math.max(
                0,
                models.length -
                  12,
              );

            ctx.ui.notify(
              [
                "Pi model setup complete.",
                "",
                `Provider: ${providerId}`,
                `API: ${api}`,
                `Models: ${models.length}`,
                sourceLabel,
                `Reasoning setting: ${reasoningMode}`,
                compatLabel,
                "",
                "Models:",
                preview,

                remaining > 0
                  ? `  ... and ${remaining} more`
                  : "",

                "",
                "Saved to:",
                modelsPath,

                "",
                "No environment variables were created or modified.",
              ]
                .filter(Boolean)
                .join("\n"),
              "info",
            );
          } catch (error) {
            ctx.ui.setWorkingMessage(
              "",
            );

            ctx.ui.notify(
              [
                "Pi model setup failed.",
                "",
                error instanceof Error
                  ? error.message
                  : String(error),
                "",
                "No environment variables were created or modified.",
              ].join("\n"),
              "error",
            );
          }
        },
    },
  );
}
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

type JsonObject = Record<string, unknown>;

type PiApi =
  | "openai-completions"
  | "openai-responses"
  | "anthropic-messages"
  | "google-generative-ai";

type PiInput = "text" | "image";

type PiCost = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
};

type DiscoveredModel = {
  id: string;
  name: string;
  reasoning: boolean;
  input: PiInput[];
  cost: PiCost;
  contextWindow: number;
  maxTokens: number;
};

type ModelsJson = {
  providers?: Record<string, JsonObject>;
  [key: string]: unknown;
};

const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;

/* -------------------------------------------------------------------------- */
/* Generic helpers                                                           */
/* -------------------------------------------------------------------------- */

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
    let numberValue: number;

    if (typeof value === "number") {
      numberValue = value;
    } else if (
      typeof value === "string" &&
      value.trim()
    ) {
      numberValue = Number(value);
    } else {
      continue;
    }

    if (
      Number.isFinite(numberValue) &&
      numberValue > 0
    ) {
      return numberValue;
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

/* -------------------------------------------------------------------------- */
/* URL / provider helpers                                                     */
/* -------------------------------------------------------------------------- */

function normalizeBaseUrl(
  value: string,
): string {
  let url = value.trim().replace(/\/+$/, "");

  /*
   * Let the user paste either:
   *
   * https://provider.example/v1
   * https://provider.example/v1/models
   * https://provider.example/models
   */
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
    url.includes("gemini")
  ) {
    return "google-generative-ai";
  }

  if (url.includes("/responses")) {
    return "openai-responses";
  }

  return "openai-completions";
}

/* -------------------------------------------------------------------------- */
/* Model metadata discovery                                                   */
/* -------------------------------------------------------------------------- */

function getContextWindow(
  model: JsonObject,
): number {
  /*
   * Different providers use different names.
   *
   * We try all common variants before falling
   * back to Pi's default.
   */
  return (
    firstNumber(
      // Common top-level names
      model.context_window,
      model.contextWindow,
      model.context_length,
      model.contextLength,

      // More variants
      model.max_context_length,
      model.maxContextLength,
      model.input_token_limit,
      model.inputTokenLimit,
      model.max_input_tokens,
      model.maxInputTokens,
      model.context_size,
      model.contextSize,

      // Nested limits
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

      // Nested capabilities
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

      // Nested limits under metadata
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

function getMaxTokens(
  model: JsonObject,
): number {
  return (
    firstNumber(
      // Common names
      model.max_tokens,
      model.maxTokens,
      model.max_output_tokens,
      model.maxOutputTokens,
      model.max_completion_tokens,
      model.maxCompletionTokens,

      // Other common names
      model.output_token_limit,
      model.outputTokenLimit,
      model.max_output_length,
      model.maxOutputLength,

      // Nested limits
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

      // Nested capabilities
      getNested(
        model,
        ["capabilities", "max_tokens"],
      ),
      getNested(
        model,
        ["capabilities", "max_output_tokens"],
      ),

      // Metadata
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

function getReasoning(
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

function getInputTypes(
  model: JsonObject,
): PiInput[] {
  let imageSupport = false;

  const modalityFields = [
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

  for (const value of modalityFields) {
    if (!Array.isArray(value)) {
      continue;
    }

    for (const item of value) {
      if (
        typeof item !== "string"
      ) {
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

/* -------------------------------------------------------------------------- */
/* Pricing discovery                                                          */
/* -------------------------------------------------------------------------- */

function getPricing(
  model: JsonObject,
): PiCost {
  const pricing =
    isObject(model.pricing)
      ? model.pricing
      : undefined;

  const cost =
    isObject(model.cost)
      ? model.cost
      : undefined;

  /*
   * Prices in Pi are USD / million tokens.
   *
   * A provider may return:
   *
   *   0.003
   *
   * meaning USD/token.
   *
   * Or:
   *
   *   3
   *
   * meaning USD/million tokens.
   *
   * We handle both common styles.
   */
  const normalizePrice = (
    value: unknown,
  ): number => {
    const numberValue =
      firstNumber(value) ?? 0;

    if (
      numberValue > 0 &&
      numberValue < 0.01
    ) {
      return numberValue * 1_000_000;
    }

    return numberValue;
  };

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

/* -------------------------------------------------------------------------- */
/* Provider response parsing                                                  */
/* -------------------------------------------------------------------------- */

function getModelArray(
  payload: unknown,
): unknown[] {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (!isObject(payload)) {
    return [];
  }

  /*
   * OpenAI:
   *   { "data": [...] }
   *
   * Other gateways:
   *   { "models": [...] }
   *   { "items": [...] }
   *   { "results": [...] }
   */
  for (
    const property of [
      "data",
      "models",
      "items",
      "results",
    ]
  ) {
    const value =
      payload[property];

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
    const item of getModelArray(payload)
  ) {
    if (!isObject(item)) {
      continue;
    }

    const id =
      firstString(
        item.id,
        item.model,
        item.slug,
      );

    if (!id) {
      continue;
    }

    const name =
      firstString(
        item.name,
        item.display_name,
        item.displayName,
        item.label,
      ) ?? id;

    models.push({
      id,
      name,

      reasoning:
        getReasoning(item),

      input:
        getInputTypes(item),

      cost:
        getPricing(item),

      contextWindow:
        getContextWindow(item),

      maxTokens:
        getMaxTokens(item),
    });
  }

  /*
   * Remove duplicate IDs.
   */
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

/* -------------------------------------------------------------------------- */
/* HTTP                                                                       */
/* -------------------------------------------------------------------------- */

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

  /*
   * Most OpenAI-compatible APIs
   * use Authorization: Bearer.
   *
   * For some APIs that don't require
   * authentication this is simply omitted.
   */
  if (apiKey.trim()) {
    headers.Authorization =
      `Bearer ${apiKey.trim()}`;
  }

  const response =
    await fetch(url, {
      method: "GET",
      headers,
    });

  const text =
    await response.text();

  if (!response.ok) {
    throw new Error(
      `HTTP ${response.status} ${response.statusText}\n` +
      text.slice(0, 500),
    );
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new Error(
      `Provider returned invalid JSON:\n` +
      text.slice(0, 500),
    );
  }
}

/* -------------------------------------------------------------------------- */
/* Model endpoint discovery                                                   */
/* -------------------------------------------------------------------------- */

async function discoverModels(
  baseUrl: string,
  apiKey: string,
): Promise<{
  endpoint: string;
  models: DiscoveredModel[];
}> {
  const candidates = [
    `${baseUrl}/models`,
    `${baseUrl}/v1/models`,
  ];

  let lastError:
    | unknown
    | undefined;

  for (
    const endpoint of [
      ...new Set(candidates),
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
          "Endpoint returned no model entries.",
        );
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    [
      "Could not discover models.",
      "",
      "Tried:",
      ...[
        ...new Set(candidates),
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

/* -------------------------------------------------------------------------- */
/* models.json                                                                */
/* -------------------------------------------------------------------------- */

function getModelsJsonPath(): string {
  /*
   * Windows:
   *   C:\Users\User\.pi\agent\models.json
   *
   * Linux:
   *   /home/user/.pi/agent/models.json
   *
   * macOS:
   *   /Users/user/.pi/agent/models.json
   */
  return path.join(
    os.homedir(),
    ".pi",
    "agent",
    "models.json",
  );
}

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

    return isObject(parsed)
      ? (parsed as ModelsJson)
      : {};
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

async function writeProvider(
  filePath: string,
  providerId: string,
  provider: JsonObject,
): Promise<void> {
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

  /*
   * IMPORTANT:
   *
   * Only this provider is replaced.
   * Existing providers stay untouched.
   */
  providers[providerId] =
    provider;

  const output: ModelsJson = {
    ...existing,
    providers,
  };

  await fs.mkdir(
    path.dirname(filePath),
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
}

/* -------------------------------------------------------------------------- */
/* Provider construction                                                      */
/* -------------------------------------------------------------------------- */

function buildProvider(
  providerId: string,
  baseUrl: string,
  apiKey: string,
  api: PiApi,
  models: DiscoveredModel[],
): JsonObject {
  return {
    name: providerId,

    baseUrl,

    /*
     * Literal API key.
     *
     * No Windows environment variable is
     * created or modified.
     */
    apiKey,

    api,

    models: models.map(
      (model) => ({
        id: model.id,
        name: model.name,

        reasoning:
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
}

/* -------------------------------------------------------------------------- */
/* Extension                                                                   */
/* -------------------------------------------------------------------------- */

export default function (
  pi: ExtensionAPI,
) {
  pi.registerCommand(
    "setup-model-json",
    {
      description:
        "Fetch provider models and generate/update ~/.pi/agent/models.json",

      handler:
        async (_args, ctx) => {
          if (!ctx.hasUI) {
            return;
          }

          try {
            /* -------------------------------------------------------------- */
            /* Ask for URL                                                     */
            /* -------------------------------------------------------------- */

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

            /* -------------------------------------------------------------- */
            /* Ask for API key                                                 */
            /* -------------------------------------------------------------- */

            const apiKey =
              await ctx.ui.input(
                "API Key",
                "Enter API key (leave empty if not required)",
              );

            if (
              apiKey === undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            /* -------------------------------------------------------------- */
            /* Provider ID                                                      */
            /* -------------------------------------------------------------- */

            const defaultId =
              deriveProviderId(
                baseUrl,
              );

            const idInput =
              await ctx.ui.input(
                "Provider ID",
                defaultId,
              );

            if (
              idInput === undefined
            ) {
              ctx.ui.notify(
                "Cancelled.",
                "warning",
              );

              return;
            }

            const providerId =
              sanitizeProviderId(
                idInput ||
                  defaultId,
              );

            if (!providerId) {
              throw new Error(
                "Provider ID is empty or invalid.",
              );
            }

            /* -------------------------------------------------------------- */
            /* Discover models                                                  */
            /* -------------------------------------------------------------- */

            ctx.ui.setWorkingMessage(
              "Fetching model catalog...",
            );

            const discovered =
              await discoverModels(
                baseUrl,
                apiKey,
              );

            /* -------------------------------------------------------------- */
            /* Infer provider API                                               */
            /* -------------------------------------------------------------- */

            const api =
              inferApi(
                baseUrl,
              );

            /* -------------------------------------------------------------- */
            /* Build Pi configuration                                           */
            /* -------------------------------------------------------------- */

            const provider =
              buildProvider(
                providerId,
                baseUrl,
                apiKey,
                api,
                discovered.models,
              );

            /* -------------------------------------------------------------- */
            /* Write models.json                                                */
            /* -------------------------------------------------------------- */

            const modelsPath =
              getModelsJsonPath();

            await writeProvider(
              modelsPath,
              providerId,
              provider,
            );

            /* -------------------------------------------------------------- */
            /* Register immediately                                             */
            /* -------------------------------------------------------------- */

            /*
             * Pi applies registerProvider calls made
             * after extension load immediately.
             *
             * models.json is still written so the
             * configuration survives Pi restarts.
             */
            pi.registerProvider(
              providerId,
              provider as any,
            );

            ctx.ui.setWorkingMessage(
              "",
            );

            /* -------------------------------------------------------------- */
            /* Result                                                           */
            /* -------------------------------------------------------------- */

            const preview =
              discovered.models
                .slice(0, 12)
                .map(
                  (model) =>
                    [
                      `  ${model.id}`,
                      `    context: ${model.contextWindow.toLocaleString()}`,
                      `    max output: ${model.maxTokens.toLocaleString()}`,
                    ].join("\n"),
                )
                .join("\n");

            const hiddenCount =
              Math.max(
                0,
                discovered.models.length -
                  12,
              );

            ctx.ui.notify(
              [
                "Model setup complete.",
                "",
                `Provider: ${providerId}`,
                `API: ${api}`,
                `Models found: ${discovered.models.length}`,
                `Endpoint: ${discovered.endpoint}`,
                "",
                "Discovered models:",
                preview,

                hiddenCount > 0
                  ? `\n  ... and ${hiddenCount} more`
                  : "",

                "",
                `Saved to:`,
                modelsPath,

                "",
                "Windows environment variables were not modified.",
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
                "Model setup failed.",
                "",
                error instanceof Error
                  ? error.message
                  : String(error),
                "",
                "No environment variables were modified.",
              ].join("\n"),
              "error",
            );
          }
        },
    },
  );
}
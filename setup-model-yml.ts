import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
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
      value.trim()
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
/* URL helpers                                                               */
/* -------------------------------------------------------------------------- */

function normalizeBaseUrl(
  value: string,
): string {
  let url = value.trim().replace(/\/+$/, "");

  /*
   * Allow:
   *
   * https://example.com/v1
   * https://example.com/v1/models
   * https://example.com/models
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
    url.includes("generativelanguage.googleapis.com") ||
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

/* -------------------------------------------------------------------------- */
/* Context window discovery                                                  */
/* -------------------------------------------------------------------------- */

function getContextWindow(
  model: JsonObject,
): number {
  return (
    firstNumber(
      /*
       * Common names
       */
      model.context_window,
      model.contextWindow,
      model.context_length,
      model.contextLength,

      /*
       * Other provider variants
       */
      model.max_context_length,
      model.maxContextLength,
      model.input_token_limit,
      model.inputTokenLimit,
      model.max_input_tokens,
      model.maxInputTokens,
      model.context_size,
      model.contextSize,

      /*
       * Nested limits
       */
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

      /*
       * Nested capabilities
       */
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

      /*
       * Metadata
       */
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

/* -------------------------------------------------------------------------- */
/* Max output tokens                                                         */
/* -------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------- */
/* Reasoning / input support                                                 */
/* -------------------------------------------------------------------------- */

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
): InputType[] {
  let hasImageSupport = false;

  const modalities = [
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

  for (const value of modalities) {
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
        hasImageSupport = true;
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
    hasImageSupport = true;
  }

  return hasImageSupport
    ? ["text", "image"]
    : ["text"];
}

/* -------------------------------------------------------------------------- */
/* Pricing                                                                   */
/* -------------------------------------------------------------------------- */

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

  const normalizePrice = (
    value: unknown,
  ): number => {
    const numberValue =
      firstNumber(value) ?? 0;

    /*
     * Some catalogs expose USD/token:
     *
     *   0.000003
     *
     * while Pi expects USD/million tokens:
     *
     *   3
     */
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
/* Model list parsing                                                        */
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
        getReasoning(value),

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
      `HTTP ${response.status} ${response.statusText}\n` +
      body.slice(0, 500),
    );
  }

  try {
    return JSON.parse(body);
  } catch {
    throw new Error(
      `Provider returned invalid JSON:\n` +
      body.slice(0, 500),
    );
  }
}

/* -------------------------------------------------------------------------- */
/* Model discovery                                                           */
/* -------------------------------------------------------------------------- */

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

      if (models.length) {
        return {
          endpoint,
          models,
        };
      }

      lastError =
        new Error(
          "No models were found in the response.",
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

/* -------------------------------------------------------------------------- */
/* OMP models.yml path                                                       */
/* -------------------------------------------------------------------------- */

function getOmpAgentDirectory(): string {
  /*
   * Windows:
   *
   * C:\Users\User\.omp\agent
   *
   * Linux:
   *
   * /home/user/.omp/agent
   *
   * macOS:
   *
   * /Users/user/.omp/agent
   */
  return path.join(
    os.homedir(),
    ".omp",
    "agent",
  );
}

function getModelsYamlPath(): string {
  return path.join(
    getOmpAgentDirectory(),
    "models.yml",
  );
}

/* -------------------------------------------------------------------------- */
/* YAML helpers                                                              */
/* -------------------------------------------------------------------------- */

/*
 * We intentionally write YAML ourselves rather than requiring a third-party
 * YAML package in the extension.
 *
 * Existing models.yml content is preserved and the provider block is
 * replaced/added by provider ID.
 */

function quoteYaml(
  value: string,
): string {
  /*
   * JSON string quoting is valid YAML for strings.
   */
  return JSON.stringify(value);
}

function yamlScalar(
  value: string | number | boolean,
): string {
  if (typeof value === "string") {
    return quoteYaml(value);
  }

  return String(value);
}

function indent(
  text: string,
  spaces: number,
): string {
  const prefix =
    " ".repeat(spaces);

  return text
    .split("\n")
    .map(
      (line) =>
        line.length
          ? prefix + line
          : line,
    )
    .join("\n");
}

function modelToYaml(
  model: DiscoveredModel,
  level = 8,
): string {
  const pad =
    " ".repeat(level);

  const child =
    " ".repeat(level + 2);

  const lines: string[] = [];

  lines.push(
    `${pad}- id: ${quoteYaml(model.id)}`,
  );

  lines.push(
    `${child}name: ${quoteYaml(model.name)}`,
  );

  lines.push(
    `${child}reasoning: ${model.reasoning}`,
  );

  lines.push(
    `${child}input:`,
  );

  for (
    const input of model.input
  ) {
    lines.push(
      `${child}  - ${input}`,
    );
  }

  lines.push(
    `${child}contextWindow: ${model.contextWindow}`,
  );

  lines.push(
    `${child}maxTokens: ${model.maxTokens}`,
  );

  lines.push(
    `${child}cost:`,
  );

  lines.push(
    `${child}  input: ${model.cost.input}`,
  );

  lines.push(
    `${child}  output: ${model.cost.output}`,
  );

  lines.push(
    `${child}  cacheRead: ${model.cost.cacheRead}`,
  );

  lines.push(
    `${child}  cacheWrite: ${model.cost.cacheWrite}`,
  );

  return lines.join("\n");
}

function providerToYaml(
  providerId: string,
  baseUrl: string,
  apiKey: string,
  api: PiApi,
  models: DiscoveredModel[],
): string {
  const lines: string[] = [];

  lines.push(
    `  ${providerId}:`,
  );

  lines.push(
    `    baseUrl: ${quoteYaml(baseUrl)}`,
  );

  lines.push(
    `    api: ${api}`,
  );

  /*
   * OMP supports a literal apiKey.
   *
   * IMPORTANT:
   * This is deliberately NOT an environment variable.
   */
  lines.push(
    `    apiKey: ${quoteYaml(apiKey)}`,
  );

  lines.push(
    `    authHeader: true`,
  );

  lines.push(
    `    models:`,
  );

  for (const model of models) {
    lines.push(
      modelToYaml(
        model,
        6,
      ),
    );
  }

  return lines.join("\n");
}

/*
 * Locate:
 *
 * providers:
 *   existing-provider:
 *     ...
 *   our-provider:
 *     ...
 *
 * and replace only our provider.
 *
 * We intentionally don't rewrite unrelated settings.
 */
function updateProvidersYaml(
  existing: string,
  providerId: string,
  providerYaml: string,
): string {
  const newline =
    existing.includes("\r\n")
      ? "\r\n"
      : "\n";

  const normalized =
    existing.replace(/\r\n/g, "\n");

  const lines =
    normalized.split("\n");

  /*
   * Find top-level "providers:"
   */
  let providersIndex =
    lines.findIndex(
      (line) =>
        /^providers:\s*$/.test(
          line.trimEnd(),
        ) &&
        !line.startsWith(" "),
    );

  /*
   * No providers section yet.
   */
  if (providersIndex === -1) {
    const prefix =
      normalized.trim().length
        ? normalized.replace(
            /\s*$/,
            "",
          ) + "\n\n"
        : "";

    return (
      prefix +
      `providers:\n` +
      providerYaml.replace(
        /^ {2}/gm,
        "  ",
      ) +
      "\n"
    ).replace(
      /\n/g,
      newline,
    );
  }

  /*
   * Find the next top-level property.
   */
  let providersEnd =
    lines.length;

  for (
    let i = providersIndex + 1;
    i < lines.length;
    i++
  ) {
    if (
      lines[i].length > 0 &&
      !lines[i].startsWith(" ") &&
      !lines[i].startsWith("\t") &&
      /^\S[^:]*:\s*/.test(
        lines[i],
      )
    ) {
      providersEnd = i;
      break;
    }
  }

  /*
   * Find the provider under providers:
   *
   *     provider-id:
   */
  let providerStart = -1;

  for (
    let i = providersIndex + 1;
    i < providersEnd;
    i++
  ) {
    const line = lines[i];

    const match =
      /^  ([^:#][^:]*):\s*$/.exec(
        line,
      );

    if (
      match &&
      match[1] === providerId
    ) {
      providerStart = i;
      break;
    }
  }

  /*
   * Provider doesn't exist.
   * Insert it at the end of providers.
   */
  if (providerStart === -1) {
    const insertAt =
      providersEnd;

    const newLines = [
      ...lines.slice(
        0,
        insertAt,
      ),
      providerYaml,
      ...lines.slice(
        insertAt,
      ),
    ];

    return newLines
      .join("\n")
      .replace(
        /\n/g,
        newline,
      );
  }

  /*
   * Provider exists.
   *
   * Remove it until the next provider-level
   * indentation or end of providers.
   */
  let providerEnd =
    providersEnd;

  for (
    let i =
      providerStart + 1;
    i < providersEnd;
    i++
  ) {
    const line = lines[i];

    if (
      /^  ([^:#][^:]*):\s*$/.test(
        line,
      )
    ) {
      providerEnd = i;
      break;
    }
  }

  const newLines = [
    ...lines.slice(
      0,
      providerStart,
    ),
    providerYaml,
    ...lines.slice(
      providerEnd,
    ),
  ];

  return newLines
    .join("\n")
    .replace(
      /\n/g,
      newline,
    );
}

/* -------------------------------------------------------------------------- */
/* File writing                                                              */
/* -------------------------------------------------------------------------- */

async function writeProviderToModelsYaml(
  providerId: string,
  baseUrl: string,
  apiKey: string,
  api: PiApi,
  models: DiscoveredModel[],
): Promise<string> {
  const filePath =
    getModelsYamlPath();

  let existing = "";

  try {
    existing =
      await fs.readFile(
        filePath,
        "utf8",
      );
  } catch (error) {
    const code =
      (error as NodeJS.ErrnoException)
        .code;

    if (code !== "ENOENT") {
      throw error;
    }
  }

  const providerYaml =
    providerToYaml(
      providerId,
      baseUrl,
      apiKey,
      api,
      models,
    );

  const result =
    updateProvidersYaml(
      existing,
      providerId,
      providerYaml,
    );

  await fs.mkdir(
    getOmpAgentDirectory(),
    {
      recursive: true,
    },
  );

  await fs.writeFile(
    filePath,
    result.trimEnd() + "\n",
    {
      encoding: "utf8",
    },
  );

  return filePath;
}

/* -------------------------------------------------------------------------- */
/* Extension command                                                         */
/* -------------------------------------------------------------------------- */

export default function (
  pi: ExtensionAPI,
) {
  pi.registerCommand(
    "setup-model-json",
    {
      description:
        "Discover a provider and generate/update OMP models.yml",

      handler:
        async (_args, ctx) => {
          if (!ctx.hasUI) {
            return;
          }

          try {
            /*
             * --------------------------------------------------------------
             * Provider URL
             * --------------------------------------------------------------
             */

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

            /*
             * --------------------------------------------------------------
             * API key
             * --------------------------------------------------------------
             */

            const apiKey =
              await ctx.ui.input(
                "API Key",
                "Enter API key",
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

            /*
             * --------------------------------------------------------------
             * Provider ID
             * --------------------------------------------------------------
             */

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
                "Invalid provider ID.",
              );
            }

            /*
             * --------------------------------------------------------------
             * Discover models
             * --------------------------------------------------------------
             */

            ctx.ui.setWorkingMessage(
              "Fetching provider model metadata...",
            );

            const result =
              await discoverModels(
                baseUrl,
                apiKey,
              );

            /*
             * --------------------------------------------------------------
             * Infer protocol
             * --------------------------------------------------------------
             */

            const api =
              inferApi(
                baseUrl,
              );

            /*
             * --------------------------------------------------------------
             * Write models.yml
             * --------------------------------------------------------------
             */

            const modelsPath =
              await writeProviderToModelsYaml(
                providerId,
                baseUrl,
                apiKey,
                api,
                result.models,
              );

            ctx.ui.setWorkingMessage(
              "",
            );

            /*
             * --------------------------------------------------------------
             * Register in runtime
             * --------------------------------------------------------------
             *
             * OMP supports extension-registered providers.
             * The models.yml write is still the persistent source.
             */

            try {
              pi.registerProvider(
                providerId,
                {
                  baseUrl,
                  api,
                  apiKey,

                  models:
                    result.models.map(
                      (model) => ({
                        id: model.id,
                        name: model.name,
                        reasoning:
                          model.reasoning,
                        input:
                          model.input,
                        contextWindow:
                          model.contextWindow,
                        maxTokens:
                          model.maxTokens,
                        cost:
                          model.cost,
                      }),
                    ),
                } as any,
              );
            } catch {
              /*
               * Persisting models.yml is the important operation.
               *
               * Some OMP versions may already have loaded the provider
               * registry and reject duplicate runtime registration.
               */
            }

            /*
             * --------------------------------------------------------------
             * Display discovered models
             * --------------------------------------------------------------
             */

            const preview =
              result.models
                .slice(0, 12)
                .map(
                  (model) =>
                    [
                      `  ${model.id}`,
                      `    contextWindow: ${model.contextWindow.toLocaleString()}`,
                      `    maxTokens: ${model.maxTokens.toLocaleString()}`,
                    ].join("\n"),
                )
                .join("\n");

            const remaining =
              Math.max(
                0,
                result.models.length -
                  12,
              );

            ctx.ui.notify(
              [
                "OMP model setup complete.",
                "",
                `Provider: ${providerId}`,
                `API: ${api}`,
                `Models discovered: ${result.models.length}`,
                `Discovery endpoint: ${result.endpoint}`,
                "",
                "Discovered metadata:",
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
                "OMP model setup failed.",
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
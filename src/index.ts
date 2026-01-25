/**
 * =============================================================================
 * ai.xaostech.io - AI Hub Worker
 * =============================================================================
 * 
 * Centralised AI inference with:
 * - Neuron budget management (10k free/day)
 * - LoRA fine-tune support for domain-specific models
 * - Training data collection for continuous improvement
 * - Fallback hierarchy: cached → fine-tuned → base model
 * 
 * Model Strategy:
 * - Fast tasks: @cf/meta/llama-3.2-3b-instruct (~4.6k neurons/M input)
 * - Quality tasks: @cf/meta/llama-3.1-8b-instruct-fp8-fast (~4.1k neurons/M input)
 * - Reasoning: @cf/qwen/qwq-32b (~60k neurons/M input, use sparingly)
 * - Embeddings: @cf/baai/bge-m3 (~1k neurons/M input, very cheap)
 * 
 * =============================================================================
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { secureHeaders } from 'hono/secure-headers';

// Types
interface Env {
    AI: any; // Cloudflare Workers AI binding
    DB: D1Database;
    CACHE_KV: KVNamespace;
    TRAINING_KV: KVNamespace;
    EDU?: Fetcher;
    LINGUA?: Fetcher;
    DATA?: Fetcher;
    // Config
    DEFAULT_MODEL: string;
    REASONING_MODEL: string;
    EMBEDDING_MODEL: string;
    LINGUA_LORA_ID?: string;
    EDU_LORA_ID?: string;
    RATE_LIMIT_PER_MINUTE: string;
    DAILY_NEURON_BUDGET: string;
    RESERVED_NEURONS_PCT: string;
}

interface InferenceRequest {
    prompt: string;
    systemPrompt?: string;
    model?: 'fast' | 'quality' | 'reasoning';
    domain?: 'lingua' | 'edu' | 'general';
    maxTokens?: number;
    temperature?: number;
    stream?: boolean;
    // For training data collection
    taskType?: string;
    metadata?: Record<string, any>;
}

interface EmbeddingRequest {
    texts: string[];
    model?: string;
}

const app = new Hono<{ Bindings: Env }>();

// =============================================================================
// MIDDLEWARE
// =============================================================================

app.use('*', logger());
app.use('*', secureHeaders());
app.use('*', cors({
    origin: [
        'https://ai.xaostech.io',
        'https://edu.xaostech.io',
        'https://lingua.xaostech.io',
        'https://api.xaostech.io',
        'https://xaostech.io',
        'http://localhost:3000',
        'http://localhost:8787',
    ],
    allowMethods: ['GET', 'POST', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
    maxAge: 86400,
}));

// =============================================================================
// NEURON BUDGET TRACKING
// =============================================================================

async function getNeuronUsageToday(env: Env): Promise<number> {
    const today = new Date().toISOString().split('T')[0];
    const cacheKey = `neurons:${today}`;

    // Check cache first
    const cached = await env.CACHE_KV.get(cacheKey);
    if (cached) return parseInt(cached, 10);

    // Query database
    const result = await env.DB.prepare(`
    SELECT SUM(neurons_used) as total 
    FROM neuron_usage 
    WHERE date = ?
  `).bind(today).first<{ total: number }>();

    const total = result?.total || 0;
    await env.CACHE_KV.put(cacheKey, total.toString(), { expirationTtl: 60 });
    return total;
}

async function recordNeuronUsage(
    env: Env,
    model: string,
    taskType: string,
    neurons: number,
    latencyMs: number,
    error: boolean = false
): Promise<void> {
    const today = new Date().toISOString().split('T')[0];
    const hour = new Date().getUTCHours();

    try {
        await env.DB.prepare(`
      INSERT INTO neuron_usage (date, hour, model, task_type, request_count, neurons_used, avg_latency_ms, error_count)
      VALUES (?, ?, ?, ?, 1, ?, ?, ?)
      ON CONFLICT (date, hour, model, task_type) DO UPDATE SET
        request_count = request_count + 1,
        neurons_used = neurons_used + excluded.neurons_used,
        avg_latency_ms = (avg_latency_ms * request_count + excluded.avg_latency_ms) / (request_count + 1),
        error_count = error_count + excluded.error_count
    `).bind(today, hour, model, taskType, neurons, latencyMs, error ? 1 : 0).run();

        // Invalidate cache
        await env.CACHE_KV.delete(`neurons:${today}`);
    } catch (err) {
        console.error('[AI] Failed to record neuron usage:', err);
    }
}

function estimateNeurons(model: string, inputTokens: number, outputTokens: number): number {
    // Approximate neuron costs per model (from CF pricing)
    const costs: Record<string, { input: number; output: number }> = {
        '@cf/meta/llama-3.2-1b-instruct': { input: 2.5, output: 18.3 },
        '@cf/meta/llama-3.2-3b-instruct': { input: 4.6, output: 30.5 },
        '@cf/meta/llama-3.1-8b-instruct-fp8-fast': { input: 4.1, output: 34.9 },
        '@cf/qwen/qwq-32b': { input: 60, output: 90.9 },
        '@cf/baai/bge-m3': { input: 1.1, output: 0 },
    };

    const cost = costs[model] || { input: 10, output: 20 };
    return Math.ceil((inputTokens * cost.input + outputTokens * cost.output) / 1000);
}

// =============================================================================
// MODEL SELECTION
// =============================================================================

function selectModel(
    env: Env,
    request: InferenceRequest
): { model: string; loraId?: string } {
    // Check for domain-specific LoRA
    if (request.domain === 'lingua' && env.LINGUA_LORA_ID) {
        return {
            model: '@cf/mistralai/mistral-7b-instruct-v0.2-lora',
            loraId: env.LINGUA_LORA_ID
        };
    }
    if (request.domain === 'edu' && env.EDU_LORA_ID) {
        return {
            model: '@cf/mistralai/mistral-7b-instruct-v0.2-lora',
            loraId: env.EDU_LORA_ID
        };
    }

    // Select base model by speed/quality preference
    switch (request.model) {
        case 'reasoning':
            return { model: env.REASONING_MODEL || '@cf/qwen/qwq-32b' };
        case 'quality':
            return { model: '@cf/meta/llama-3.1-8b-instruct-fp8-fast' };
        case 'fast':
        default:
            return { model: env.DEFAULT_MODEL || '@cf/meta/llama-3.2-3b-instruct' };
    }
}

// =============================================================================
// ROUTES
// =============================================================================

app.get('/', (c) => {
    return c.json({
        service: 'ai.xaostech.io',
        version: '1.0.0',
        status: 'operational',
        endpoints: {
            'POST /infer': 'Run inference with automatic model selection',
            'POST /embed': 'Generate embeddings',
            'GET /usage': 'Get neuron usage statistics',
            'GET /models': 'List available models',
            'POST /training/exercise': 'Submit exercise training data',
            'POST /training/translation': 'Submit translation training data',
        },
    });
});

// Health check
app.get('/health', async (c) => {
    const usage = await getNeuronUsageToday(c.env);
    const budget = parseInt(c.env.DAILY_NEURON_BUDGET) || 10000;
    const remaining = budget - usage;

    return c.json({
        status: remaining > 0 ? 'healthy' : 'budget_exceeded',
        neurons: {
            used: usage,
            budget,
            remaining,
            percentUsed: Math.round((usage / budget) * 100),
        },
    });
});

// List available models
app.get('/models', (c) => {
    return c.json({
        models: {
            fast: {
                id: c.env.DEFAULT_MODEL || '@cf/meta/llama-3.2-3b-instruct',
                description: 'Fast responses, good for simple tasks',
                neuronsPerMInput: 4625,
                neuronsPerMOutput: 30475,
            },
            quality: {
                id: '@cf/meta/llama-3.1-8b-instruct-fp8-fast',
                description: 'Higher quality, balanced speed',
                neuronsPerMInput: 4119,
                neuronsPerMOutput: 34868,
            },
            reasoning: {
                id: c.env.REASONING_MODEL || '@cf/qwen/qwq-32b',
                description: 'Complex reasoning, expensive',
                neuronsPerMInput: 60000,
                neuronsPerMOutput: 90909,
            },
            embedding: {
                id: c.env.EMBEDDING_MODEL || '@cf/baai/bge-m3',
                description: 'Text embeddings, very cheap',
                neuronsPerMInput: 1075,
            },
        },
        finetunes: {
            lingua: c.env.LINGUA_LORA_ID ? 'available' : 'not_configured',
            edu: c.env.EDU_LORA_ID ? 'available' : 'not_configured',
        },
    });
});

// Get usage statistics
app.get('/usage', async (c) => {
    const days = parseInt(c.req.query('days') || '7');

    const results = await c.env.DB.prepare(`
    SELECT date, model, task_type, 
           SUM(request_count) as requests,
           SUM(neurons_used) as neurons,
           AVG(avg_latency_ms) as avg_latency,
           SUM(error_count) as errors
    FROM neuron_usage
    WHERE date >= date('now', '-' || ? || ' days')
    GROUP BY date, model, task_type
    ORDER BY date DESC, neurons DESC
  `).bind(days).all();

    const todayUsage = await getNeuronUsageToday(c.env);
    const budget = parseInt(c.env.DAILY_NEURON_BUDGET) || 10000;

    return c.json({
        today: {
            used: todayUsage,
            budget,
            remaining: budget - todayUsage,
        },
        history: results.results,
    });
});

// Main inference endpoint
app.post('/infer', async (c) => {
    const request = await c.req.json<InferenceRequest>();

    if (!request.prompt) {
        return c.json({ error: 'prompt is required' }, 400);
    }

    // Check budget
    const usage = await getNeuronUsageToday(c.env);
    const budget = parseInt(c.env.DAILY_NEURON_BUDGET) || 10000;
    const reserved = (parseInt(c.env.RESERVED_NEURONS_PCT) || 20) / 100;
    const available = budget * (1 - reserved) - usage;

    if (available <= 0) {
        return c.json({
            error: 'Daily neuron budget exceeded',
            usage: { used: usage, budget, available: 0 },
        }, 429);
    }

    const { model, loraId } = selectModel(c.env, request);
    const startTime = Date.now();

    try {
        const messages = [
            ...(request.systemPrompt ? [{ role: 'system', content: request.systemPrompt }] : []),
            { role: 'user', content: request.prompt },
        ];

        const aiRequest: any = {
            messages,
            max_tokens: request.maxTokens || 1024,
            temperature: request.temperature ?? 0.7,
        };

        // Add LoRA if available
        if (loraId) {
            aiRequest.lora = loraId;
            aiRequest.raw = true; // Skip default chat template for LoRA
        }

        const response = await c.env.AI.run(model, aiRequest);
        const latencyMs = Date.now() - startTime;

        // Estimate tokens (rough: 4 chars per token)
        const inputTokens = Math.ceil((request.systemPrompt?.length || 0 + request.prompt.length) / 4);
        const outputTokens = Math.ceil((response.response?.length || 0) / 4);
        const neuronsUsed = estimateNeurons(model, inputTokens, outputTokens);

        // Record usage
        await recordNeuronUsage(c.env, model, request.taskType || 'inference', neuronsUsed, latencyMs);

        return c.json({
            response: response.response,
            model,
            loraId,
            usage: {
                inputTokens,
                outputTokens,
                neuronsEstimate: neuronsUsed,
                latencyMs,
            },
        });
    } catch (err: any) {
        const latencyMs = Date.now() - startTime;
        await recordNeuronUsage(c.env, model, request.taskType || 'inference', 0, latencyMs, true);

        console.error('[AI] Inference error:', err);
        return c.json({
            error: 'Inference failed',
            message: err.message,
            model,
        }, 500);
    }
});

// Embeddings endpoint
app.post('/embed', async (c) => {
    const request = await c.req.json<EmbeddingRequest>();

    if (!request.texts || !Array.isArray(request.texts) || request.texts.length === 0) {
        return c.json({ error: 'texts array is required' }, 400);
    }

    if (request.texts.length > 100) {
        return c.json({ error: 'Maximum 100 texts per request' }, 400);
    }

    const model = request.model || c.env.EMBEDDING_MODEL || '@cf/baai/bge-m3';
    const startTime = Date.now();

    try {
        const response = await c.env.AI.run(model, { text: request.texts });
        const latencyMs = Date.now() - startTime;

        // Estimate neurons (embeddings are cheap)
        const totalChars = request.texts.reduce((sum, t) => sum + t.length, 0);
        const inputTokens = Math.ceil(totalChars / 4);
        const neuronsUsed = estimateNeurons(model, inputTokens, 0);

        await recordNeuronUsage(c.env, model, 'embedding', neuronsUsed, latencyMs);

        return c.json({
            embeddings: response.data,
            model,
            usage: {
                textCount: request.texts.length,
                inputTokens,
                neuronsEstimate: neuronsUsed,
                latencyMs,
            },
        });
    } catch (err: any) {
        const latencyMs = Date.now() - startTime;
        await recordNeuronUsage(c.env, model, 'embedding', 0, latencyMs, true);

        console.error('[AI] Embedding error:', err);
        return c.json({ error: 'Embedding failed', message: err.message }, 500);
    }
});

// =============================================================================
// TRAINING DATA COLLECTION
// =============================================================================

// Submit exercise training data
app.post('/training/exercise', async (c) => {
    const data = await c.req.json();

    const id = crypto.randomUUID();

    try {
        await c.env.DB.prepare(`
      INSERT INTO training_exercises (
        id, subject, category, difficulty, exercise_type, topic,
        prompt_input, model_output, parsed_exercise, source_model,
        generation_time_ms, neurons_used
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
            id,
            data.subject,
            data.category,
            data.difficulty,
            data.exerciseType,
            data.topic,
            data.promptInput,
            data.modelOutput,
            JSON.stringify(data.parsedExercise),
            data.sourceModel,
            data.generationTimeMs,
            data.neuronsUsed
        ).run();

        return c.json({ success: true, id });
    } catch (err: any) {
        console.error('[AI] Training data error:', err);
        return c.json({ error: 'Failed to store training data' }, 500);
    }
});

// Submit translation training data
app.post('/training/translation', async (c) => {
    const data = await c.req.json();

    const id = crypto.randomUUID();

    try {
        await c.env.DB.prepare(`
      INSERT INTO training_translations (
        id, source_lang, target_lang, source_text,
        model_translation, model_used, context, domain, neurons_used
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
            id,
            data.sourceLang,
            data.targetLang,
            data.sourceText,
            data.modelTranslation,
            data.modelUsed,
            data.context,
            data.domain,
            data.neuronsUsed
        ).run();

        return c.json({ success: true, id });
    } catch (err: any) {
        console.error('[AI] Training data error:', err);
        return c.json({ error: 'Failed to store training data' }, 500);
    }
});

// Update training data with quality signals
app.patch('/training/exercise/:id', async (c) => {
    const id = c.req.param('id');
    const updates = await c.req.json();

    const fields: string[] = [];
    const values: any[] = [];

    if (updates.correctRate !== undefined) {
        fields.push('correct_rate = ?');
        values.push(updates.correctRate);
    }
    if (updates.avgTimeSeconds !== undefined) {
        fields.push('avg_time_seconds = ?');
        values.push(updates.avgTimeSeconds);
    }
    if (updates.qualityScore !== undefined) {
        fields.push('quality_score = ?');
        values.push(updates.qualityScore);
    }
    if (updates.approved !== undefined) {
        fields.push('approved_for_training = ?');
        values.push(updates.approved ? 1 : 0);
        if (updates.approved) {
            fields.push('reviewed_at = datetime("now")');
        }
    }

    if (fields.length === 0) {
        return c.json({ error: 'No valid update fields' }, 400);
    }

    values.push(id);

    try {
        await c.env.DB.prepare(`
      UPDATE training_exercises 
      SET ${fields.join(', ')}
      WHERE id = ?
    `).bind(...values).run();

        return c.json({ success: true });
    } catch (err: any) {
        return c.json({ error: 'Update failed' }, 500);
    }
});

// Export approved training data for LoRA fine-tuning
app.get('/training/export/:type', async (c) => {
    const type = c.req.param('type');
    const limit = parseInt(c.req.query('limit') || '1000');

    if (type === 'exercises') {
        const results = await c.env.DB.prepare(`
      SELECT prompt_input, model_output, quality_score
      FROM training_exercises
      WHERE approved_for_training = 1
      ORDER BY quality_score DESC
      LIMIT ?
    `).bind(limit).all();

        // Format for LoRA training (instruction/response pairs)
        const trainingData = results.results?.map((r: any) => ({
            instruction: r.prompt_input,
            output: r.model_output,
        }));

        return c.json({ count: trainingData?.length || 0, data: trainingData });
    }

    if (type === 'translations') {
        const results = await c.env.DB.prepare(`
      SELECT source_text, COALESCE(edited_translation, model_translation) as translation,
             source_lang, target_lang
      FROM training_translations
      WHERE approved_for_training = 1
      ORDER BY rating DESC NULLS LAST
      LIMIT ?
    `).bind(limit).all();

        return c.json({ count: results.results?.length || 0, data: results.results });
    }

    return c.json({ error: 'Invalid type. Use "exercises" or "translations"' }, 400);
});

export default app;

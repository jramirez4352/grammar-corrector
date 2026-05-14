import 'dotenv/config';
import express from 'express';
import rateLimit from 'express-rate-limit';
import Anthropic from '@anthropic-ai/sdk';
import Groq from 'groq-sdk';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(join(__dirname, 'public')));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please wait a few minutes and try again.' },
});
app.use('/api/', limiter);

const useGroq = !!process.env.GROQ_API_KEY;
const useAnthropic = !!process.env.ANTHROPIC_API_KEY;

if (!useGroq && !useAnthropic) {
  console.error('Error: Set GROQ_API_KEY (free at groq.com) or ANTHROPIC_API_KEY in your .env file.');
  process.exit(1);
}

const groqClient = useGroq ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;
const anthropicClient = useAnthropic ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
console.log(`Provider: ${useGroq ? 'Groq (llama-3.3-70b-versatile)' : 'Anthropic (claude-opus-4-7)'}`);

async function streamCompletion(res, systemPrompt, userContent) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  if (useGroq) {
    const stream = await groqClient.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      max_tokens: 4096,
      stream: true,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ],
    });
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || '';
      if (delta) res.write(`data: ${JSON.stringify({ text: delta })}\n\n`);
    }
  } else {
    const stream = anthropicClient.messages.stream({
      model: 'claude-opus-4-7',
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userContent }],
    });
    stream.on('text', delta => res.write(`data: ${JSON.stringify({ text: delta })}\n\n`));
    await stream.finalMessage();
  }

  res.write('data: [DONE]\n\n');
  res.end();
}

function validate(req, res) {
  const { text } = req.body;
  if (!text || typeof text !== 'string' || !text.trim())
    return res.status(400).json({ error: 'Text is required.' });
  if (text.length > 50_000)
    return res.status(400).json({ error: 'Text is too long (max 50,000 characters).' });
  return null;
}

function handleError(res, err, label) {
  console.error(`${label} error:`, err.message);
  try { res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`); res.end(); } catch {}
}

// ── Grammar correction ────────────────────────────────────────────────────────
const CORRECT_PROMPT = `You are a professional grammar and spelling editor. \
When given text, return ONLY the corrected version with proper grammar, spelling, and punctuation. \
Do not add any commentary, explanations, preamble, or formatting changes. \
Preserve the original meaning, tone, and structure as closely as possible. \
If the text has no errors, return it exactly as given.`;

app.post('/api/correct', async (req, res) => {
  if (validate(req, res)) return;
  try { await streamCompletion(res, CORRECT_PROMPT, req.body.text); }
  catch (err) { handleError(res, err, 'Correct'); }
});

// ── Rephrase ──────────────────────────────────────────────────────────────────
const REPHRASE_BASE = 'Fix all grammar, spelling and punctuation errors, and at the same time ';
const REPHRASE_PROMPTS = {
  formal:       REPHRASE_BASE + 'rewrite the text in a formal, professional tone. Return ONLY the rewritten text, no explanations.',
  casual:       REPHRASE_BASE + 'rewrite the text in a casual, conversational tone. Return ONLY the rewritten text, no explanations.',
  concise:      REPHRASE_BASE + 'rewrite the text more concisely, removing unnecessary words while keeping the core meaning. Return ONLY the rewritten text, no explanations.',
  expanded:     REPHRASE_BASE + 'expand and enrich the text with more detail and description. Return ONLY the rewritten text, no explanations.',
  professional: REPHRASE_BASE + 'rewrite the text for a business/professional context. Return ONLY the rewritten text, no explanations.',
  simple:       REPHRASE_BASE + 'rewrite the text using simple, easy-to-understand language. Return ONLY the rewritten text, no explanations.',
};

app.post('/api/rephrase', async (req, res) => {
  if (validate(req, res)) return;
  const { style } = req.body;
  if (!REPHRASE_PROMPTS[style]) return res.status(400).json({ error: 'Invalid style.' });
  try { await streamCompletion(res, REPHRASE_PROMPTS[style], req.body.text); }
  catch (err) { handleError(res, err, 'Rephrase'); }
});

// ── Translate ─────────────────────────────────────────────────────────────────
const VALID_LANGS = ['English','Spanish','French','German','Italian','Portuguese','Chinese','Japanese','Arabic','Russian','Korean','Dutch'];

app.post('/api/translate', async (req, res) => {
  if (validate(req, res)) return;
  const { targetLang } = req.body;
  if (!VALID_LANGS.includes(targetLang)) return res.status(400).json({ error: 'Invalid language.' });
  const prompt = `Fix any grammar, spelling and punctuation errors in the text, then translate it to ${targetLang}. Return ONLY the translated text, no explanations or commentary.`;
  try { await streamCompletion(res, prompt, req.body.text); }
  catch (err) { handleError(res, err, 'Translate'); }
});

// ── Summarize ─────────────────────────────────────────────────────────────────
const SUMMARIZE_PROMPT = `Summarize the following text as briefly as possible — aim for 20-30% of the original length. \
Extract only the most essential points, cutting all filler, examples, and repetition. \
Fix any grammar or spelling errors. Return ONLY the summary, no explanations or commentary.`;

app.post('/api/summarize', async (req, res) => {
  if (validate(req, res)) return;
  try { await streamCompletion(res, SUMMARIZE_PROMPT, req.body.text); }
  catch (err) { handleError(res, err, 'Summarize'); }
});

// ── Explain changes ───────────────────────────────────────────────────────────
app.post('/api/explain', async (req, res) => {
  const { original, corrected } = req.body;
  if (!original || !corrected) return res.status(400).json({ error: 'Original and corrected text are required.' });

  if (original.trim() === corrected.trim()) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    res.write(`data: ${JSON.stringify({ text: 'No changes were made — the text had no grammar or spelling errors.' })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
    return;
  }

  const prompt = `You are a grammar teacher. Compare the original and corrected versions of a text and explain each change made. Be concise and educational. Format your response as bullet points.`;
  const userContent = `Original:\n${original}\n\nCorrected:\n${corrected}`;
  try { await streamCompletion(res, prompt, userContent); }
  catch (err) { handleError(res, err, 'Explain'); }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Grammar Corrector running at http://localhost:${PORT}`));

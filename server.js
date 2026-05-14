import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import Groq from 'groq-sdk';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(join(__dirname, 'public')));

const useGroq = !!process.env.GROQ_API_KEY;
const useAnthropic = !!process.env.ANTHROPIC_API_KEY;

if (!useGroq && !useAnthropic) {
  console.error('Error: Set GROQ_API_KEY (free at groq.com) or ANTHROPIC_API_KEY in your .env file.');
  process.exit(1);
}

const groqClient = useGroq ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;
const anthropicClient = useAnthropic ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;

console.log(`Provider: ${useGroq ? 'Groq (llama-3.3-70b-versatile)' : 'Anthropic (claude-opus-4-7)'}`);

const SYSTEM_PROMPT = `You are a professional grammar and spelling editor. \
When given text, return ONLY the corrected version with proper grammar, spelling, and punctuation. \
Do not add any commentary, explanations, preamble, or formatting changes. \
Preserve the original meaning, tone, and structure as closely as possible. \
If the text has no errors, return it exactly as given.`;

app.post('/api/correct', async (req, res) => {
  const { text } = req.body;

  if (!text || typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'Text is required.' });
  }
  if (text.length > 50_000) {
    return res.status(400).json({ error: 'Text is too long (max 50,000 characters).' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  try {
    if (useGroq) {
      const stream = await groqClient.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        max_tokens: 4096,
        stream: true,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: text },
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
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content: text }],
      });

      stream.on('text', (delta) => {
        res.write(`data: ${JSON.stringify({ text: delta })}\n\n`);
      });

      await stream.finalMessage();
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    const provider = useGroq ? 'Groq' : 'Anthropic';
    console.error(`${provider} error:`, err.message);
    try {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    } catch {}
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Grammar Corrector running at http://localhost:${PORT}`);
});

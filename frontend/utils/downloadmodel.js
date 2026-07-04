import fs from 'fs';
import path from 'path';
import axios from 'axios';

// Vercel hanya mengizinkan operasi 'write' di folder /tmp
const MODEL_DIR = '/tmp';
const MODEL_PATH = path.join(MODEL_DIR, 'v4_wav2vec_3f.pt');
const MODEL_URL = 'https://huggingface.co/Mufids/DLMindVoice/resolve/main/v4_wav2vec_3f.pt'; // Jalur URL dari Langkah 1

export async function initModel() {
  // 1. Cek apakah model sudah terunduh di serverless instance ini (biar tidak download berkali-kali)
  if (fs.existsSync(MODEL_PATH)) {
    console.log('Model sudah siap di /tmp.');
    return MODEL_PATH;
  }

  console.log('Model belum ada. Memulai unduhan dari Hugging Face...');

  // 2. Download file besar menggunakan streaming
  const response = await axios({
    method: 'GET',
    url: MODEL_URL,
    responseType: 'stream',
  });

  const writer = fs.createWriteStream(MODEL_PATH);
  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => {
      console.log('Model berhasil diunduh ke Vercel!');
      resolve(MODEL_PATH);
    });
    writer.on('error', (err) => {
      console.error('Gagal mengunduh model:', err);
      reject(err);
    });
  });
}
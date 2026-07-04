import type { NextApiRequest, NextApiResponse } from 'next';
import { initModel } from '../../utils/downloadmodel';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const modelPath = await initModel();

    return res.status(200).json({
      success: true,
      message: 'Model siap digunakan dari path: ' + modelPath,
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

export const config = {
  maxDuration: 60,
};

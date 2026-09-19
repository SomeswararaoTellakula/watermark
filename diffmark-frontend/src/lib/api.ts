import axios from 'axios'

const apiBase = (process.env.NEXT_PUBLIC_API_URL || '/api').replace(/\/$/, '')

export const api = axios.create({
  baseURL: apiBase,
  headers: {
    'Accept': 'application/json'
  }
})

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('dm_token')
    if (token) {
      config.headers = config.headers || {}
      ;(config.headers as any)['Authorization'] = `Bearer ${token}`
    }
  }
  return config
})

export async function embedWatermark(image: File, watermark: string) {
  const form = new FormData()
  form.append('image', image)
  form.append('watermark', watermark)
  const res = await api.post('/embed', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { watermarked_image_url: string, id?: string }
}

export async function embedWatermarkRobust(image: File, watermark: string) {
  const form = new FormData()
  form.append('image', image)
  form.append('watermark', watermark)
  const res = await api.post('/robust/embed', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { watermarked_image_url: string, id?: string, robust?: boolean }
}

export async function robustStatus() {
  const res = await api.get('/robust/status')
  return res.data as { available: boolean, checkpoint: string | null }
}

export async function robustReload() {
  const res = await api.post('/robust/reload')
  return res.data as { available: boolean, checkpoint: string | null }
}

export async function health() {
  const res = await api.get('/health')
  return res.data as { ok: boolean, db: 'up' | 'down', ffmpeg?: 'up' | 'down' }
}

export async function verifyImage(image: File) {
  const form = new FormData()
  form.append('image', image)
  const res = await api.post('/verify', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function getHistory() {
  const res = await api.get('/history')
  return res.data as { items: Array<{ id: string, filename?: string, watermarked_image_url: string, wm_text?: string, robust?: boolean, uploadDate?: string }> }
}

export async function verifyById(id: string) {
  const form = new FormData()
  form.append('id', id)
  const res = await api.post('/verify', form)
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function verifyRobustById(id: string) {
  const form = new FormData()
  form.append('id', id)
  const res = await api.post('/robust/verify', form)
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function verifyRobustFile(image: File) {
  const form = new FormData()
  form.append('image', image)
  const res = await api.post('/robust/verify', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function embedVideo(video: File, watermark: string) {
  const form = new FormData()
  form.append('video', video)
  form.append('watermark', watermark)
  const res = await api.post('/video/embed', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { watermarked_video_url: string, id?: string }
}

export async function verifyVideoById(id: string) {
  const form = new FormData()
  form.append('id', id)
  const res = await api.post('/video/verify', form)
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function verifyVideoFile(video: File) {
  const form = new FormData()
  form.append('video', video)
  const res = await api.post('/video/verify', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string }
}

export async function embedVideoRobust(video: File, watermark: string) {
  const form = new FormData()
  form.append('video', video)
  form.append('watermark', watermark)
  const res = await api.post('/video/robust/embed', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data as { watermarked_video_url: string, id?: string, robust?: boolean }
}

export async function verifyVideoRobustById(id: string) {
  const form = new FormData()
  form.append('id', id)
  const res = await api.post('/video/robust/verify', form)
  return res.data as { detected: boolean, accuracy: number, confidence: number, watermark_message?: string, robust?: boolean }
}

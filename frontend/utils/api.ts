export const getApiUrl = (path: string): string => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  
  // Use environment variable if set, otherwise default to localhost or relative path
  const productionApiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (productionApiUrl) {
    return `${productionApiUrl}${cleanPath}`;
  }
  
  if (typeof window !== "undefined") {
    // If running on localhost, connect directly to FastAPI dev server on port 8000
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return `http://localhost:8000${cleanPath}`;
    }
    // In production (Vercel), use relative path routed via vercel.json rewrite
    return `/_backend${cleanPath}`;
  }
  
  return `http://localhost:8000${cleanPath}`;
};

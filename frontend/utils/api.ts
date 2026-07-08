export const getApiUrl = (path: string): string => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  
  // Use environment variable if set, otherwise default to localhost or relative path
  const productionApiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (productionApiUrl) {
    return `${productionApiUrl}${cleanPath}`;
  }
  
  if (typeof window !== "undefined") {
    return `https://mindvoice-api.onrender.com${cleanPath}`;
  }
  
  return `https://mindvoice-api.onrender.com${cleanPath}`;
};

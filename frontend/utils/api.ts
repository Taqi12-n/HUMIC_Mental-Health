export const getApiUrl = (path: string): string => {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;

  // Development
  if (process.env.NODE_ENV === "development") {
    return `http://localhost:8000${cleanPath}`;
  }

  // Production
  return `https://mindvoice-api.onrender.com${cleanPath}`;
};
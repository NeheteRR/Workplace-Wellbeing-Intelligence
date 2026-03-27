export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchWithAuth(
  endpoint: string,
  options: RequestInit = {}
) {
  let token = null;
  if (typeof window !== "undefined") {
    token = localStorage.getItem("token");
  }

  const headers = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (token) {
    (headers as any)["Authorization"] = `Bearer ${token}`;
    // The backend seems to expect x-org-id or x-source in some places, but for now we'll pass token.
    // wait, the backend doesn't fully validate tokens in all endpoints (e.g., depends on employee_id from query).
    // Let's also pass a default org_id for mock purposes to x-org-id
    let org_id = null;
    if (typeof window !== "undefined") {
      org_id = localStorage.getItem("org_id");
    }
    if (org_id) {
        (headers as any)["x-org-id"] = org_id;
    }
  }

  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let message = "An error occurred";
    try {
      const data = await res.json();
      message = data.detail || data.message || message;
    } catch (e) {
      // Ignore JSON parse error, use default message
    }
    throw new Error(message);
  }

  return res.json();
}

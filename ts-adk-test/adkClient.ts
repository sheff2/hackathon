// adkClient.ts
const ADK_BASE   = "http://127.0.0.1:8000";
const APP_NAME   = "multichatter";
const USER_ID    = "u_test";
const SESSION_ID = "s_test";

type Part = { text?: string; functionResponse?: { response?: any } };
type Event = { message?: { role?: string; parts?: Part[] }; content?: { parts?: Part[] } };

export async function startSession(): Promise<void> {
  const url = `${ADK_BASE}/apps/${APP_NAME}/users/${USER_ID}/sessions/${SESSION_ID}`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state: {} }),
  });
  if (!r.ok) throw new Error(`ensure session failed: ${r.status} ${await r.text()}`);
}

function pickAssistantText(events: Event[]): string | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const m = events[i]?.message;
    if (m?.role === "assistant") {
      const t = m.parts?.map(p => p.text).filter(Boolean) as string[] | undefined;
      if (t?.length) return t.join("\n");
    }
    const c = events[i]?.content?.parts?.map(p => p.text).filter(Boolean) as string[] | undefined;
    if (c?.length) return c.join("\n");
  }
  return undefined;
}

function pickToolError(events: Event[]): string | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const parts = events[i]?.message?.parts ?? events[i]?.content?.parts ?? [];
    for (const p of parts) {
      const resp = p.functionResponse?.response;
      if (resp && typeof resp === "object" && resp.status === "error") {
        return resp.error_message || JSON.stringify(resp);
      }
    }
  }
  return undefined;
}

export async function ask(text: string): Promise<string> {
  const r = await fetch(`${ADK_BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      app_name: APP_NAME,
      user_id: USER_ID,
      session_id: SESSION_ID,
      new_message: { role: "user", parts: [{ text }] },
    }),
  });
  if (!r.ok) throw new Error(`POST /run failed: ${r.status} ${await r.text()}`);

  const events = (await r.json()) as Event[];
  return pickAssistantText(events) || pickToolError(events) || "(no assistant text)";
}

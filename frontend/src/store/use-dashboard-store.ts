import { create } from "zustand";
import type { DgaPayload, ChatMessage } from "@/types/dga";
import { runPredictionFromFile, runPredictionFromJson, askChatBackend, resetDataset, type ChatHistoryTurn } from "@/lib/api";

// How many prior turns to send as conversation memory — enough for
// pronoun/follow-up resolution ("what about that one?", "compare it with
// the previous transformer") without letting the prompt grow unbounded.
const CHAT_HISTORY_TURNS = 8;

interface DashboardState {
  payload: DgaPayload | null;
  loading: boolean;
  error: string | null;
  selectedTransformerId: string | null;
  bannerDismissed: boolean;

  chatOpen: boolean;
  chatMessages: ChatMessage[];
  chatSending: boolean;

  dataPanelOpen: boolean;
  setDataPanelOpen: (open: boolean) => void;

  uploadFile: (file: File) => Promise<void>;
  uploadJson: (rows: unknown[]) => Promise<void>;
  clearData: () => Promise<void>;
  setSelectedTransformer: (id: string | null) => void;
  dismissBanner: () => void;

  toggleChat: (open?: boolean) => void;
  sendChatMessage: (question: string) => Promise<void>;
  clearChat: () => void;
}

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi! I can look up transformer status, explain a diagnostic result, or compare two units. What would you like to know?",
  createdAt: Date.now(),
};

export const useDashboardStore = create<DashboardState>((set, get) => ({
  payload: null,
  loading: false,
  error: null,
  selectedTransformerId: null,
  bannerDismissed: false,

  chatOpen: false,
  chatMessages: [WELCOME_MESSAGE],
  chatSending: false,

  dataPanelOpen: false,
  setDataPanelOpen: (open) => set({ dataPanelOpen: open }),

  uploadFile: async (file: File) => {
    set({ loading: true, error: null });
    try {
      const payload = await runPredictionFromFile(file);
      set({ payload, loading: false, bannerDismissed: false });
    } catch (e) {
      set({ loading: false, error: e instanceof Error ? e.message : "Prediction failed." });
      throw e;
    }
  },

  uploadJson: async (rows: unknown[]) => {
    set({ loading: true, error: null });
    try {
      const payload = await runPredictionFromJson(rows);
      set({ payload, loading: false, bannerDismissed: false });
    } catch (e) {
      set({ loading: false, error: e instanceof Error ? e.message : "Prediction failed." });
      throw e;
    }
  },

  clearData: async () => {
    set({ payload: null, selectedTransformerId: null });
    // Also wipes the backend's accumulated dataset (dataset_accumulator.py)
    // — otherwise the next upload would silently merge into whatever was
    // cleared here, which isn't what "Clear data" implies.
    await resetDataset();
  },

  setSelectedTransformer: (id) => set({ selectedTransformerId: id }),
  dismissBanner: () => set({ bannerDismissed: true }),

  toggleChat: (open) => set((state) => ({ chatOpen: open ?? !state.chatOpen })),

  sendChatMessage: async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;

    // Snapshot the conversation so far (before adding this new question) as
    // memory for the backend — lets follow-ups like "what about that one?"
    // resolve against what was actually discussed in this session.
    const history: ChatHistoryTurn[] = get()
      .chatMessages.filter((m) => m.id !== "welcome")
      .slice(-CHAT_HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: trimmed,
      createdAt: Date.now(),
    };
    set((state) => ({ chatMessages: [...state.chatMessages, userMsg], chatSending: true }));

    const { payload, selectedTransformerId } = get();

    if (!payload) {
      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: "Upload a DGA dataset first (Data Source panel) so I have something to analyze.",
        createdAt: Date.now(),
      };
      set((state) => ({ chatMessages: [...state.chatMessages, assistantMsg], chatSending: false }));
      return;
    }

    const context = {
      ...payload.chat_context_payload,
      transformer_summary: selectedTransformerId
        ? [
            ...payload.chat_context_payload.transformer_summary.filter(
              (s) => s.transformer_id === selectedTransformerId
            ),
            ...payload.chat_context_payload.transformer_summary.filter(
              (s) => s.transformer_id !== selectedTransformerId
            ),
          ]
        : payload.chat_context_payload.transformer_summary,
    };

    try {
      const answer = await askChatBackend(trimmed, context, history);
      const sourceChip = extractSourceChip(trimmed, context);
      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: answer,
        sourceChip,
        createdAt: Date.now(),
      };
      set((state) => ({ chatMessages: [...state.chatMessages, assistantMsg], chatSending: false }));
    } catch (e) {
      const errMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: "assistant",
        content: e instanceof Error ? e.message : "Something went wrong answering that.",
        createdAt: Date.now(),
      };
      set((state) => ({ chatMessages: [...state.chatMessages, errMsg], chatSending: false }));
    }
  },

  clearChat: () => set({ chatMessages: [WELCOME_MESSAGE] }),
}));

function extractSourceChip(
  question: string,
  context: { transformer_summary: { transformer_id: string; latest_sample_day: string }[] } | null
): string | undefined {
  if (!context) return undefined;
  const q = question.toLowerCase();
  const match = context.transformer_summary.find((s) => q.includes(s.transformer_id.toLowerCase()));
  const target = match ?? context.transformer_summary[0];
  if (!target) return undefined;
  const date = target.latest_sample_day ? new Date(target.latest_sample_day) : null;
  const dateStr = date && !Number.isNaN(date.getTime()) ? formatDate(date) : target.latest_sample_day;
  return `${target.transformer_id} · ${dateStr}`;
}

function formatDate(d: Date): string {
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${mm}/${dd}/${d.getUTCFullYear()}`;
}

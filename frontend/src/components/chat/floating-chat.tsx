"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, X, Info, Mic, Send, Trash2, ChevronDown, GripVertical, GripHorizontal } from "lucide-react";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "Which transformers are Critical right now?",
  "Why is this transformer flagged?",
  "What does acetylene (C₂H₂) indicate?",
  "How is the severity score calculated?",
  "Compare Duval Triangle and IEC 60599",
  "Show me the H₂ and CO trend for the current transformer over the last 2 years.",
];

const DEFAULT_WIDTH = 384;
const DEFAULT_HEIGHT = 32 * 16; // 32rem in px (approx)
const MIN_WIDTH = 320;
const MAX_WIDTH = 640;
const MIN_HEIGHT = 300; // px
const MAX_HEIGHT = 600; // px
const WIDTH_STORAGE_KEY = "dga-chat-width";
const HEIGHT_STORAGE_KEY = "dga-chat-height";
const SUGGESTIONS_OPEN_STORAGE_KEY = "dga-chat-suggestions-open";

function readStoredWidth(): number {
  if (typeof window === "undefined") return DEFAULT_WIDTH;
  const raw = Number(window.localStorage.getItem(WIDTH_STORAGE_KEY));
  return raw >= MIN_WIDTH && raw <= MAX_WIDTH ? raw : DEFAULT_WIDTH;
}

function readStoredHeight(): number {
  if (typeof window === "undefined") return DEFAULT_HEIGHT;
  const raw = Number(window.localStorage.getItem(HEIGHT_STORAGE_KEY));
  return raw >= MIN_HEIGHT && raw <= MAX_HEIGHT ? raw : DEFAULT_HEIGHT;
}

function readStoredSuggestionsOpen(): boolean {
  if (typeof window === "undefined") return true;
  const raw = window.localStorage.getItem(SUGGESTIONS_OPEN_STORAGE_KEY);
  return raw === null ? true : raw === "1";
}

export function FloatingChat() {
  const open = useDashboardStore((s) => s.chatOpen);
  const toggleChat = useDashboardStore((s) => s.toggleChat);
  const messages = useDashboardStore((s) => s.chatMessages);
  const sending = useDashboardStore((s) => s.chatSending);
  const sendChatMessage = useDashboardStore((s) => s.sendChatMessage);
  const clearChat = useDashboardStore((s) => s.clearChat);
  const payload = useDashboardStore((s) => s.payload);
  const selectedTransformerId = useDashboardStore((s) => s.selectedTransformerId);

  const [input, setInput] = useState("");
  const [listening, setListening] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [width, setWidth] = useState(readStoredWidth);
  const [height, setHeight] = useState(readStoredHeight);
  const [resizing, setResizing] = useState(false);
  const resizeStartRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);
  const [suggestionsOpen, setSuggestionsOpen] = useState(readStoredSuggestionsOpen);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  // Resize logic (cả chiều rộng và chiều cao)
  useEffect(() => {
    if (!resizing) return;
    function onMove(e: MouseEvent) {
      if (!resizeStartRef.current) return;
      const { x: startX, y: startY, width: startW, height: startH } = resizeStartRef.current;
      // Chiều rộng: kéo cạnh trái sang trái -> tăng width (vì panel neo phải)
      const deltaX = startX - e.clientX;
      const nextWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startW + deltaX));
      setWidth(nextWidth);
      // Chiều cao: kéo cạnh trên lên trên -> tăng height (vì panel neo dưới)
      const deltaY = startY - e.clientY;
      const nextHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, startH + deltaY));
      setHeight(nextHeight);
    }
    function onUp() {
      setResizing(false);
      resizeStartRef.current = null;
      setWidth((w) => {
        window.localStorage.setItem(WIDTH_STORAGE_KEY, String(w));
        return w;
      });
      setHeight((h) => {
        window.localStorage.setItem(HEIGHT_STORAGE_KEY, String(h));
        return h;
      });
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    document.body.style.cursor = "nw-resize"; // thay đổi con trỏ
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [resizing]);

  function toggleSuggestions() {
    setSuggestionsOpen((v) => {
      const next = !v;
      window.localStorage.setItem(SUGGESTIONS_OPEN_STORAGE_KEY, next ? "1" : "0");
      return next;
    });
  }

  function submit(text: string) {
    if (!text.trim()) return;
    sendChatMessage(text);
    setInput("");
  }

  function startVoiceInput() {
    type SpeechRecognitionCtor = new () => {
      lang: string;
      interimResults: boolean;
      onresult: (e: { results: { transcript: string }[][] }) => void;
      onend: () => void;
      start: () => void;
    };
    const w = window as typeof window & {
      webkitSpeechRecognition?: SpeechRecognitionCtor;
      SpeechRecognition?: SpeechRecognitionCtor;
    };
    const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
    if (!Ctor) {
      setInput((prev) => prev || "");
      return;
    }
    const recognition = new Ctor();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.onresult = (e) => {
      const transcript = e.results[0]?.[0]?.transcript ?? "";
      setInput(transcript);
    };
    recognition.onend = () => setListening(false);
    setListening(true);
    recognition.start();
  }

  const contextLabel = selectedTransformerId ?? "All transformers";

  return (
    <div className="no-print">
      <motion.button
        onClick={() => toggleChat()}
        whileTap={{ scale: 0.94 }}
        className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-teal-800 text-white shadow-lg ring-4 ring-teal-800/10 transition-colors hover:bg-teal-700 cursor-pointer"
        aria-label="Open DGA Assistant"
      >
        <Bot className="h-6 w-6" />
        {payload && payload.transformer_summary.some((s) => s.latest_score >= 13) && (
          <span className="absolute right-1 top-1 h-2.5 w-2.5 rounded-full bg-status-critical ring-2 ring-teal-800" />
        )}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.97 }}
            transition={resizing ? { duration: 0 } : { type: "spring", damping: 26, stiffness: 300 }}
            style={{ width, height }}
            className="fixed bottom-24 right-6 z-50 flex max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-2xl border border-teal-800/10 bg-white shadow-2xl"
          >
            {/* Handle resize cạnh trái (chiều rộng) */}
            <div
              onMouseDown={(e) => {
                resizeStartRef.current = { x: e.clientX, y: e.clientY, width, height };
                setResizing(true);
              }}
              className={cn(
                "group absolute bottom-0 left-0 top-0 z-10 flex w-2 cursor-ew-resize items-center justify-center hover:bg-teal-500/15",
                resizing && "bg-teal-500/20"
              )}
              aria-hidden
            >
              <GripVertical className="h-4 w-4 text-teal-800/0 group-hover:text-teal-800/40" />
            </div>

            {/* Handle resize cạnh trên (chiều cao) */}
            <div
              onMouseDown={(e) => {
                resizeStartRef.current = { x: e.clientX, y: e.clientY, width, height };
                setResizing(true);
              }}
              className={cn(
                "group absolute left-0 right-0 top-0 z-10 flex h-2 cursor-n-resize items-center justify-center hover:bg-teal-500/15",
                resizing && "bg-teal-500/20"
              )}
              aria-hidden
            >
              <GripHorizontal className="h-4 w-4 text-teal-800/0 group-hover:text-teal-800/40 rotate-90" />
            </div>

            <div className="flex items-start justify-between gap-3 bg-gradient-to-b from-teal-900 to-teal-800 px-4 py-3.5 text-white">
              <div className="flex items-center gap-2.5">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-copper-500/90">
                  <Bot className="h-4 w-4 text-teal-950" />
                </span>
                <div>
                  <div className="text-sm font-bold">DGA Assistant</div>
                  <div className="text-[11px] text-teal-200">Ask about any transformer in plain language</div>
                </div>
              </div>
              <button
                onClick={() => toggleChat(false)}
                className="rounded-full p-1 text-teal-200 hover:bg-white/10 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex items-start gap-2 border-b border-status-watch-border bg-status-watch-soft px-3.5 py-2 text-[11px] text-status-watch">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              This assistant only looks up and explains data already in the system — it does not replace
              an engineer&apos;s judgment.
            </div>

            <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto scrollbar-thin px-3.5 py-3">
              {messages.map((m) => (
                <div key={m.id} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                  <div
                    className={cn(
                      "max-w-[85%] rounded-2xl px-3.5 py-2 text-[13px] leading-relaxed whitespace-pre-line",
                      m.role === "user"
                        ? "bg-teal-800 text-white rounded-br-sm"
                        : "bg-cream-100 text-teal-900 rounded-bl-sm"
                    )}
                  >
                    {m.content}
                    {m.sourceChip && (
                      <div className="mt-1.5">
                        <span className="inline-flex items-center gap-1 rounded-full bg-teal-100 px-2 py-0.5 text-[10px] font-semibold text-teal-700">
                          {m.sourceChip}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {sending && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-cream-100 px-3.5 py-2.5">
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        className="h-1.5 w-1.5 rounded-full bg-teal-400"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.15 }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-cream-200 px-3.5 py-2">
              <div className="mb-2 flex items-center justify-between">
                <span className="rounded-full bg-teal-50 px-2 py-0.5 text-[10px] font-semibold text-teal-600">
                  Context: {contextLabel}
                </span>
                <button
                  onClick={clearChat}
                  className="flex items-center gap-1 text-[11px] text-teal-400 hover:text-status-critical cursor-pointer"
                >
                  <Trash2 className="h-3 w-3" /> Clear
                </button>
              </div>
              <button
                onClick={toggleSuggestions}
                className="mb-1.5 flex items-center gap-1 text-[11px] font-medium text-teal-400 hover:text-teal-600 cursor-pointer"
              >
                <ChevronDown className={cn("h-3 w-3 transition-transform", !suggestionsOpen && "-rotate-90")} />
                Quick suggestions
              </button>
              {suggestionsOpen && (
                <div className="flex flex-wrap gap-1.5">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => submit(s)}
                      className="rounded-full border border-teal-200 bg-white px-2.5 py-1 text-[11px] text-teal-700 hover:bg-teal-50 cursor-pointer"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                submit(input);
              }}
              className="flex items-center gap-1.5 border-t border-cream-200 p-2.5"
            >
              <button
                type="button"
                onClick={startVoiceInput}
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border cursor-pointer transition-colors",
                  listening
                    ? "border-status-critical text-status-critical animate-pulse"
                    : "border-teal-200 text-teal-500 hover:bg-teal-50"
                )}
                aria-label="Voice input"
              >
                <Mic className="h-3.5 w-3.5" />
              </button>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a question, e.g. explain C2H2…"
                className="h-8 flex-1 rounded-full border border-teal-200 bg-cream-50 px-3.5 text-[13px] outline-none focus:border-teal-500"
              />
              <button
                type="submit"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-teal-800 text-white hover:bg-teal-700 cursor-pointer disabled:opacity-40"
                disabled={!input.trim()}
                aria-label="Send"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

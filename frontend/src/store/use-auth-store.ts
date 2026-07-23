import { create } from "zustand";
import { ApiError, fetchCurrentUser, getAuthToken, loginAccount, logoutAccount, type AuthUser } from "@/lib/api";

const AUTH_TOKEN_KEY = "dga-auth-token";

interface AuthState {
  user: AuthUser | null;
  status: "loading" | "authenticated" | "unauthenticated";
  error: string | null;

  init: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

function storeToken(token: string) {
  window.localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function clearToken() {
  window.localStorage.removeItem(AUTH_TOKEN_KEY);
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  status: "loading",
  error: null,

  init: async () => {
    // Both AuthGuard and the login page call init() on mount. If a login
    // just resolved (status already "authenticated"), skip re-verifying —
    // a slow or failed /auth/me check here would otherwise clear the token
    // that was just stored and bounce the user straight back to /login
    // right after a successful sign-in.
    if (get().status === "authenticated") return;
    if (!getAuthToken()) {
      set({ status: "unauthenticated", user: null });
      return;
    }
    const user = await fetchCurrentUser();
    if (user) {
      set({ status: "authenticated", user });
    } else {
      clearToken();
      set({ status: "unauthenticated", user: null });
    }
  },

  login: async (email, password) => {
    set({ error: null });
    try {
      const { user, token } = await loginAccount(email, password);
      storeToken(token);
      set({ user, status: "authenticated" });
    } catch (e) {
      set({ error: e instanceof ApiError ? e.message : "Login failed." });
      throw e;
    }
  },

  logout: async () => {
    await logoutAccount();
    clearToken();
    set({ user: null, status: "unauthenticated" });
  },

  clearError: () => set({ error: null }),
}));
